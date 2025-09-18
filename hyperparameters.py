import sys
import torch
import os
import jsonpickle
import copy

current_dir = os.path.dirname(os.path.realpath(__file__))
'''
nih_pancreas
target_vox_space = [0.85,0.85,0.75]

la
target_vox_space = [0.625, 0.625, 0.625]
'''

def fold_distribution(fold):
    if fold==1:
        data_split = {
            'training':[0, 0.75],
            'validation':[0.75, 0.8],
            'testing':[0.8, 1],
        }
    elif fold==2:
        data_split = {
            'training':[0.2, 0.95],
            'validation':[0.95, 1],
            'testing':[0, 0.2],
        }
    elif fold==3:
        data_split = {
            'training':[0.4, 1, 0, 0.15],
            'validation':[0.15, 0.2],
            'testing':[0.2, 0.4],
        }
    elif fold==4:
        data_split = {
            'training':[0.6, 1, 0, 0.35],
            'validation':[0.35, 0.4],
            'testing':[0.4, 0.6],
        }
    elif fold==5:
        data_split = {
            'training':[0.8, 1, 0, 0.55],
            'validation':[0.55, 0.6],
            'testing':[0.6, 0.8],
        }
    else:
        raise ValueError('Unknown fold')
    
    return data_split

class Hyperparameters_base:
    def __init__(self, **kwargs) -> None:
        self.system(**kwargs)
        
    def system(self, **kwargs):
        'Cluster or local'
        if sys.platform=='darwin':
            # Have not tested on Mac though, please expect some debugging if you train/test on Mac
            self.os_system = 'mac'
        elif sys.platform=='linux':
            self.os_system = 'local'
        else:
            raise ValueError('Unknown platform')
        
        'Device for PyTorch'
        if self.os_system=='mac':
            self.device = torch.device("mps")
            self.use_cuda = False
        
    def set(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                v = self.special_key(k, v)
                setattr(self, k, v)
    
    def special_key(self, key, value):
        if key == 'save_path':
            if value is None:
                value = os.path.join(current_dir, 'experiments', 'no_name')
            elif '\\' in value or '/' in value:
                value = value
            else:
                value = os.path.join(current_dir, 'experiments', value)
        elif key == 'device_num':
            if self.os_system=='mac':
                self.device = torch.device("mps")
                self.use_cuda = False
            else:
                self.use_cuda = torch.cuda.is_available()
                self.device = torch.device("cuda:%d"%(value) if self.use_cuda else "cpu")
        elif key == 'slice_num':
            try:
                self.tform_training['slicing']['slice_num'] = value
            except:
                print('slice_num not found in tform_training')
        elif key == 'fold':
            self.data_split.update(fold_distribution(value))
        
        return value

    def print(self):
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print('')

    def save_json(self, path=None):
        if path is None and self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            path = os.path.join(self.save_path, self.hp_name)
        elif path is None:
            raise ValueError('No path specified for saving hyperparameters')
        
        json_data = jsonpickle.encode(self, indent=4)

        with open(path, 'w') as f:
            f.write(json_data)

    def load_json(self, path=None):
        if path is None:
            path = os.path.join(self.save_path, self.hp_name)

        with open(path, 'r') as f:
            json_data = f.read()
        
        hp = jsonpickle.decode(json_data)
        self.set(**hp.__dict__)

        self.system()


class Hyperparameters(Hyperparameters_base):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        'Dataset'
        self.dataset_type = 'la'   # 'la' or 'nih_pancreas'

        'Datasplit'
        self.fold = 1
        self.data_split = {
            'training_use':4,   # number of labelled volumes used
            'validation_use':'all',
            'testing_use':'all',
        } 
        if kwargs.get('fold') is not None:
            self.data_split = self.special_key('fold', kwargs.get('fold'))
        
        'Dataloader'
        self.num_workers = 4 if kwargs.get('num_workers') is None else kwargs.get('num_workers')
        self.batch_size = 5      # number of volumes in each batch

        'Data Transformation'
        crop_size = [160,160,64]
        target_vox_space = [0.625, 0.625, 0.625]    # varies for different datasets, check line 9-13 for the default values we used in our experiments
        self.tform_training = {
            'patch':{'crop_size':crop_size, 'prob':0.5, 'target_vox_space': target_vox_space, 'aug_range':0.1},
            # 'rotate_90':{'prob':0.2},
            # 'flip':{'prob':0.2},
            'gamma':{'prob':0.1, 'gamma':[0.8, 1.2]},
        }
        self.tform_training_pseudo = copy.deepcopy(self.tform_training) # for augmentation of pseudo-labelled data
        self.tform_training_pseudo['patch']['prob'] = 0.5
        
        self.tform_testing = {
            'crop':{'crop_size':crop_size},
            'resample':{'target_vox_space': target_vox_space, 'aug_range':0},
        }

        'Device for PyTorch'
        self.device_num = self.special_key('device_num', kwargs.get('device_num') if kwargs.get('device_num') is not None else 0)
        
        'Save'
        self.save_path = self.special_key('save_path', kwargs.get('save_path'))
        self.hp_name = self.special_key('hyperparameters.json', kwargs.get('hp_name'))

        'Training schedule (Check train.py for details)'
        self.max_epochs = 4000
        self.warmup_epochs = 50
        self.transition_epochs = 500
        self.restart_epoch = None
        self.freq_switch = 1

        'Save pseudo labels'
        self.save_pseudo = [1000, 1500, 2000, 3000]  # epochs to save pseudo labels. It will save once after transition_epochs

        'Models'
        self.model = {}

        # 2D natural image pretrained model
        self.model.update(
            {'model_2d':{
                # Check train.py for details
                'training_specs':{
                    'optimizer':'adamw',
                    'learning_rate':0.001,
                    'max_epochs':self.max_epochs,
                    'warmup_epochs':self.warmup_epochs,
                    'patience':20,
                },
                # Check model.py for details
                'model_type':'nvidia/segformer-b2-finetuned-ade-512-512',
                'model_specs':{
                    'pretrained':True,
                    'in':'repeat',
                    'base':'lora',
                    'out':'',
                    'rank':64,
                    'alpha':64,
                },
            }}
        )
        
        # 3D medical segmentation model from scratch
        self.model.update(
            {'model_3d':{
                'training_specs':{
                    'optimizer':'adamw',
                    'learning_rate':0.0001,
                    'max_epochs':self.max_epochs,
                    'warmup_epochs':self.warmup_epochs,
                    'patience':10,
                },
                'model_type':'unet3d',
                'model_specs':{
                    'pretrained':False,
                    'dropout':0,
                    'base_n_filter':32,
                },
            }}
        )



from hyperparameters import Hyperparameters as Hyperparameters
    
if __name__ ==  '__main__':    
    hp = Hyperparameters()
    ### Can either set some important parameters here or directly edit in the Hyperparameters class
    hp.set(**{'fold':1})
    hp.set(**{'save_path':'test'})
    hp.set(**{'hp_name':'hyperparameters.json'})
    ###
    hp.print()
    hp.save_json()
    

    
    