import torch
import torch.nn as nn
import peft
from transformers import SegformerForSemanticSegmentation, PretrainedConfig
from monai.networks.nets import SwinUNETR

def get_model(model_name, hp):
    if 'segformer' in hp.model[model_name]['model_type']:
        return Segformer_segmentation(model_name, hp)
    elif 'swinunetr' in hp.model[model_name]['model_type']:
        return Swinunetr_segmentation(model_name, hp)
    elif 'unet3d' in hp.model[model_name]['model_type']:
        return Unet3d_segmentation(model_name, hp)
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
class CatLayer(torch.nn.Module):
    def __init__(self, repeat: int) -> None:
        super().__init__()
        self.repeat = repeat

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cat([input]*self.repeat, dim=1)
    
class BaseModel(nn.Module):
    def __init__(self, model_name, hp):
        super().__init__()
        self.in_channels = hp.in_channels
        self.num_classes = hp.num_classes+1    # +1 for background
        self.model_specs = hp.model[model_name]['model_specs'] 

        'Step 1: Load the original model'
        self.base_model = self.create_model(hp.model[model_name]['model_type'], pretrained=self.model_specs['pretrained'])
        self.get_params_from_base()

        'Step 2: Add input layers'
        if 'in' in self.model_specs:
            if self.model_specs['in']=='repeat':
                self.input_model = self.repeat_input_channel()
            elif self.model_specs['in']=='linear':
                self.input_model = self.add_input_layers()
            elif self.model_specs['in']=='none':
                self.input_model = nn.Identity()
            else:
                raise ValueError(f"Input layer {self.model_specs['in']} not recognized")
        else:
            pass
        
        'Step 3: Add output decoder layers'
        self.output_model = self.modify_output_layers()

        'Step 4: Freeze the original model'
        if 'base' in self.model_specs:
            if self.model_specs['base']=='freeze':
                self.freeze_module(self.base_model)
            elif self.model_specs['base']=='train':
                pass
            elif self.model_specs['base']=='lora':
                self.base_model = self.lora(self.base_model)
            else:
                raise ValueError(f"Base model {self.model_specs['base']} not recognized")
        else:
            pass
        
    def get_params_from_base(self):
        pass

    def freeze_module(self, module):
        for name, param in module.named_parameters():
            param.requires_grad = False

    def create_model(self, model_type, pretrained=True):
        pass

    def add_input_layers(self):
        input_layer = nn.Sequential(nn.Conv2d(self.in_channels, 3, kernel_size=1, stride=1, padding=0))

        input_layer[0].weight.data.fill_(1)
        input_layer[0].bias.data.fill_(0)

        return input_layer
    
    def repeat_input_channel(self):
        return nn.Sequential(CatLayer(3))

    def modify_output_layers(self):
        pass

    def lora(self, model):
        # Config for the LoRA Injection via PEFT, check their documentation for details
        peft_config = peft.LoraConfig(
            r=self.model_specs['rank'], # rank dimension of the LoRA injected matrices
            lora_alpha=self.model_specs['alpha'], # parameter for scaling
            target_modules=['qkv', 'attn.proj', 'fc'], # the name would vary depending on the model
            modules_to_save=["norm"], 
            lora_dropout=0.1, 
            bias="all", 
        )

        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model
    

class Segformer_segmentation(BaseModel):
    def __init__(self, model_name, hp):
        super().__init__(model_name, hp)
       

    def create_model(self, model_type, pretrained=True):
        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(model_type)
        else:
            config = PretrainedConfig.from_pretrained(model_type)
            model = SegformerForSemanticSegmentation(config)

        model.decode_head.classifier = nn.Identity()

        return model
    
    def add_input_layers(self):
        middle_channels = 16
        input_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, middle_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, 3, kernel_size=1, stride=1, padding=0),
            )

        input_layer[0].weight.data.fill_(1)
        input_layer[0].bias.data.fill_(0)
        input_layer[2].weight.data.fill_(1/middle_channels)
        input_layer[2].bias.data.fill_(0)
        input_layer[4].weight.data.fill_(1/middle_channels)
        input_layer[4].bias.data.fill_(0)

        return input_layer
    
    def get_params_from_base(self):
        self.num_features = self.base_model.config.decoder_hidden_size                  #256

    def freeze_module(self, module):
        for name, param in module.named_parameters():
            param.requires_grad = False
    
    def modify_output_layers(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.num_features+3, self.num_classes, kernel_size=4, stride=4, padding=0),
        )
    
    def lora(self, model):
        # Config for the LoRA Injection via PEFT
        peft_config = peft.LoraConfig(
            r=self.model_specs['rank'], # rank dimension of the LoRA injected matrices
            lora_alpha=self.model_specs['alpha'], # parameter for scaling
            target_modules=['query', 'key', 'value', 'dense', 'dense1', 'dense2'], # be precise about dense because classifier has dense too
            modules_to_save=["0.layer_norm", 
                             "1.layer_norm",
                             "2.layer_norm",
                             "3.layer_norm",
                             "self.layer_norm",
                             "layer_norm_1",
                             "layer_norm_2",
                             "layer_norm.0",
                             "layer_norm.1",
                             "layer_norm.2",
                             "layer_norm.3",
                             ], 
            lora_dropout=0.1, 
            bias="all", 
        )

        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model
    
    def forward(self, x):
        x_in = self.input_model(x) # B,C,H,W
        x_out = self.base_model(x_in)

        x_final = torch.cat((x_out.logits,x_in[:,:,::4,::4]), dim=1)
        x_final = self.output_model(x_final)

        return x_final
    


class Unet3d_segmentation(BaseModel):
    def __init__(self, model_name, hp):
        super().__init__(model_name, hp)
        try:
            self.base_n_filter = hp.model[model_name]['model_specs']['base_n_filter']
        except:
            print('UNet 3d no base_n_filter specified, defaulting to 32')
            self.base_n_filter = 32

        self.lrelu = nn.LeakyReLU()
        
        self.conv2d_d1 = self.conv_block(hp.in_channels, self.base_n_filter)
        self.conv2d_d2 = self.conv_block(self.base_n_filter, self.base_n_filter*2)
        self.conv2d_d3 = self.conv_block(self.base_n_filter*2, self.base_n_filter*4)
        self.conv2d_d4 = self.conv_block(self.base_n_filter*4, self.base_n_filter*8)
        self.conv2d_d5 = self.conv_block(self.base_n_filter*8, self.base_n_filter*16)
        
        self.up_1 = self.up_sample_block(self.base_n_filter*16, self.base_n_filter*8)
        self.conv2d_u1 = self.conv_block(self.base_n_filter*16, self.base_n_filter*8)
        self.up_2 = self.up_sample_block(self.base_n_filter*8, self.base_n_filter*4)
        self.conv2d_u2 = self.conv_block(self.base_n_filter*8, self.base_n_filter*4)
        self.up_3 = self.up_sample_block(self.base_n_filter*4, self.base_n_filter*2)
        self.conv2d_u3 = self.conv_block(self.base_n_filter*4, self.base_n_filter*2)
        self.up_4 = self.up_sample_block(self.base_n_filter*2, self.base_n_filter)
        self.conv2d_u4 = self.conv_block(self.base_n_filter*2, self.base_n_filter)


        self.pred_final = nn.Sequential(
                nn.Conv3d(self.base_n_filter, hp.num_classes+1, kernel_size=3, stride=1, padding=1),
        )

        
        
    def conv_block(self, feat_in, feat_out):
        return nn.Sequential(
                nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(feat_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(feat_out, feat_out, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(feat_out),
                nn.ReLU(inplace=True)
            )
    
    def up_sample_block(self, feat_in, feat_out):
        return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(feat_out),
                nn.ReLU(inplace=True),
            )


    def forward(self, x):
        down1 = self.conv2d_d1(x)
        pool1 = nn.MaxPool3d(kernel_size=2, stride=2)(down1)
        
        down2 = self.conv2d_d2(pool1)
        pool2 = nn.MaxPool3d(kernel_size=2, stride=2)(down2)
        
        down3 = self.conv2d_d3(pool2)
        pool3 = nn.MaxPool3d(kernel_size=2, stride=2)(down3)
        
        down4 = self.conv2d_d4(pool3)
        pool4 = nn.MaxPool3d(kernel_size=2, stride=2)(down4)
        
        down5 = self.conv2d_d5(pool4)
        
        up1 = self.up_1(down5)
        up1 = torch.cat((down4,up1), dim=1)
        up1 = self.conv2d_u1(up1)
        
        up2 = self.up_2(up1)
        up2 = torch.cat((down3,up2), dim=1)
        up2 = self.conv2d_u2(up2)
        
        up3 = self.up_3(up2)
        up3 = torch.cat((down2,up3), dim=1)
        up3 = self.conv2d_u3(up3)
        
        up4 = self.up_4(up3)
        up4 = torch.cat((down1,up4), dim=1)
        up4 = self.conv2d_u4(up4)
        
        prediction = self.pred_final(up4)
            
         
        return prediction
    

    
class Swinunetr_segmentation(BaseModel):
    def __init__(self, model_name, hp):
        super().__init__(model_name, hp)
        
        self.model = SwinUNETR(
            img_size=tuple(hp.tform_testing['crop']['crop_size']),
            in_channels=hp.in_channels,
            out_channels=hp.num_classes+1,
            feature_size=36,
            use_checkpoint=True,
        )
    
    def forward(self, x):
        x = self.model(x) 

        return x