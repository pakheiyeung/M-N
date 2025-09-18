import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np
import argparse
import math
from torch.utils.tensorboard import SummaryWriter
import random

current_dir = os.path.dirname(os.path.realpath(__file__))


from dataset import pick_dataset, Dataset_test
from dataset import Dataset_3D as Dataset
from model import get_model
from losses import DiceLoss
from utilities import Split_dataset, write_tensorboard, Test_runner, Results, Convert_2D_3D
from hyperparameters import Hyperparameters

torch.manual_seed(101)
torch.cuda.manual_seed(101)
np.random.seed(101)
random.seed(101)

parser = argparse.ArgumentParser(description="training pipeline")
parser.add_argument("--save_path", default='test', type=str, help="Path (or folder) to save the model")
parser.add_argument("--hp_name", default='hyperparameters.json', type=str, help="Name of the hyperparameter file")
parser.add_argument("--device_num", default=0, type=int, help="CUDA device number")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")


def create_loaders(hp, epoch):
    num_training = hp.data_split['training_use']

    hp.data_split['training_use'] = 'all'
    split_dataset = Split_dataset(hp, print_stat=False)
    files_split = split_dataset.split()
    files_training_pseudo = sum(files_split.values(), [])
    # files_training_pseudo = sum([v for k, v in files_split.items() if k != 'extra'], [])
    
    hp.data_split['training_use'] = num_training
    split_dataset = Split_dataset(hp, print_stat=False)
    files_split = split_dataset.split()
    files_training = files_split['training']

    files_training_pseudo = [f for f in files_training_pseudo if f not in files_training]

    total_round = 8 # just a hyperparameter to control the number of iterations per epoch

    if epoch<hp.transition_epochs:
        num_pseudo = 0
    else:
        num_pseudo = math.floor((epoch-hp.transition_epochs)/(hp.max_epochs-hp.transition_epochs)*(hp.batch_size)) #124/2000 * 6

    num_label = hp.batch_size - num_pseudo #6

    # for debugging
    # num_label = 3
    # num_pseudo = hp.batch_size - num_label 

    'Dataset and dataloader for labelled data'
    if num_label!=0:
        random.shuffle(files_training)
        files_training = files_training[:num_label*total_round]
        params = {
            'batch_size': num_label,
            'shuffle': True,
            'num_workers': hp.num_workers,
            'drop_last':False,
        }     # for data generator
        ds_training = pick_dataset(files_training)
        training_set = Dataset(ds_training, transform=hp.tform_training)
        training_generator = data.DataLoader(training_set, **params)
    else:
        training_generator = ((None, None) for _ in range(total_round))


    'Dataset and dataloader for pseudo-labelled data'
    if num_pseudo!=0:
        random.shuffle(files_training_pseudo)
        files_training_pseudo = files_training_pseudo[:num_pseudo*total_round]
        params = {
            'batch_size': num_pseudo,
            'shuffle': True,
            'num_workers': hp.num_workers,
            'drop_last':False,
        }     # for data generator
        ds_training_pseudo = pick_dataset(files_training_pseudo)
        ds_training_pseudo.load_pseudo_label_list()
        training_pseudo_set = Dataset(ds_training_pseudo, transform=hp.tform_training_pseudo)
        training_pseudo_generator = data.DataLoader(training_pseudo_set, **params)
    else:
        training_pseudo_generator = ((None, None) for _ in range(total_round))

    return training_pseudo_generator, training_generator


def create_model(model, model_name, hp):
    'model'
    model = model.to(hp.device)

    'Path and files for saving the models'
    save_path_model = os.path.join(hp.save_path, model_name)
    saved_model = os.path.join(save_path_model,'best_model.pth')
    current_model = os.path.join(save_path_model, 'current_model.pth')
    
    if not os.path.exists(save_path_model):
        os.makedirs(save_path_model)

    'load current weight'
    try:
        model.load_state_dict(torch.load(hp.model[model_name]['model_specs']['pretrained']))
    except:
        print('No pretrained model loaded')

    'training specs'
    training_specs = hp.model[model_name]['training_specs']

    'Loss'
    criterion = {
        'dice':DiceLoss(normalization='softmax'),
        'distri':nn.KLDivLoss(reduction='mean'),
    }

    'Optimizer'
    if training_specs['optimizer']=='adamw':
        optimizer = optim.AdamW(model.parameters(), lr=training_specs['learning_rate'])
    elif training_specs['optimizer']=='adam':
        optimizer = optim.Adam(model.parameters(), lr=training_specs['learning_rate'])
    elif training_specs['optimizer']=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=training_specs['learning_rate'], momentum=training_specs['momentum'])
    else:
        raise ValueError('Optimizer not recognized')
    
    'Scheduler'
    scheduler_list = []
    milestone_list = [hp.warmup_epochs, hp.transition_epochs]
    scheduler_list.append(optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=hp.warmup_epochs))
    scheduler_list.append(optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hp.transition_epochs-hp.warmup_epochs, eta_min=0))
    if hp.restart_epoch is not None:
        scheduler_list.append(optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hp.restart_epoch, eta_min=0))
    else:
        scheduler_list.append(optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hp.max_epochs-hp.transition_epochs, eta_min=0))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, 
                                                schedulers=scheduler_list, 
                                                milestones=milestone_list)
    
    'Early stop when loss plateu'
    stop_count = 0
    
    'Evaluation metrics'
    loss = {
        'dice':[],
        'distri':[],
        'overall':[]
    }
    best = None
    matrix = {}
    
    
    'Group all'
    model_group = {
        'model':model,
        'name':model_name,
        'saved_model':saved_model,
        'current_model':current_model,
        'optimizer': optimizer,
        'learning_rate':training_specs['learning_rate'],
        'scheduler': scheduler,
        'patience':training_specs['patience'],
        'criterion':criterion,
        'loss': loss,
        'best': best,
        'stop_count':stop_count,
        'stop':False,
        'matrix':matrix,
        'remark':''
    }
    
    return model_group


        


def run_warmup(model_group, data_label, hp, train=None):
    for m in model_group:
        m['model'] = m['model'].train()
        m['optimizer'].zero_grad()
    
    'Data'
    img_label, y_label = data_label     #B, Ci, H, W, D

    'Loop over the models'
    for m in model_group:
        'Generate local batch'
        if '2d' in m['name']:
            converter = Convert_2D_3D()

            '3D to 2D'
            local_batch = converter.convert_3D_to_2D(img_label).to(device=hp.device, dtype=torch.float)    #Bp*D, Ci, H, W

            'Pass through the model'
            pred = m['model'](local_batch)   #Bp*D, Cm, H, W

            '2D to 3D'
            pred = converter.convert_2D_to_3D(pred)     #Bp, Cm, H, W, D

        elif '3d' in m['name']:
            local_batch = img_label.to(device=hp.device, dtype=torch.float)    #Bp, Ci, H, W, D

            'pass through the model'
            pred = m['model'](local_batch)   #B, Cm, H, W, D
        
        else:
            raise ValueError('2D or 3D not recognized')
        

        'Generate groundtruth'
        y = y_label.to(device=hp.device, dtype=torch.float) #B, Cm, H, W, D

        'Loss'
        loss_dice = m['criterion']['dice'](pred, y)
        loss_distri = m['criterion']['distri'](F.log_softmax(pred, dim=1), y)

        loss = loss_dice + loss_distri

        if train:
            loss.backward()
            m['optimizer'].step()
            m['optimizer'].zero_grad()

        m['loss']['overall'].append(loss.item())
        m['loss']['dice'].append(loss_dice.item())
        m['loss']['distri'].append(loss_distri.item())




def run(model_pseudo, model_label, data_pseudo, data_label, hp, train=None, pseudo_mode=None):
    if pseudo_mode is None:
        raise ValueError('Pseudo mode not defined')
    if train:
        model_label['model'] = model_label['model'].train()
        model_label['optimizer'].zero_grad()
        
    'Data'
    img_pseudo, _ = data_pseudo     #B, Ci, H, W, D
    img_label, y_label = data_label

    
    'Pseudo'
    if img_pseudo!=None:
        if pseudo_mode=='3D':
            converter = Convert_2D_3D()

            '3D to 2D'
            local_batch = converter.convert_3D_to_2D(img_pseudo).to(device=hp.device, dtype=torch.float)    #Bp*D, Ci, H, W

            with torch.no_grad():
                y = model_pseudo['model'](local_batch)   #Bp*D, Cm, H, W

            '2D to 3D'
            y = converter.convert_2D_to_3D(y)     #Bp, Cm, H, W, D

        elif pseudo_mode=='2D':
            local_batch = img_pseudo.to(device=hp.device, dtype=torch.float)    #Bp, Ci, H, W, D
           
            with torch.no_grad():
                y = model_pseudo['model'](local_batch)   #Bp, Cm, H, W, D
        
        y = F.softmax(y, dim=1)   #Bp, Cm, H, W, D

    'Generate groundtruth'
    if img_pseudo!=None and img_label!=None:
        y = torch.cat((y, y_label.to(device=hp.device, dtype=torch.float)), dim=0)   #B, Cm, H, W, D
    elif img_label!=None:
        y = y_label.to(device=hp.device, dtype=torch.float) #B, Cm, H, W, D
    
    'Training'
    if img_label!=None and img_pseudo!=None:
        local_batch = torch.cat((img_pseudo, img_label), dim=0) #B, Ci, H, W, D
    elif img_label!=None:
        local_batch = img_label
    elif img_pseudo!=None:
        local_batch = img_pseudo    #B, Ci, H, W, D

    if pseudo_mode=='2D':
        converter = Convert_2D_3D()

        '3D to 2D'
        local_batch = converter.convert_3D_to_2D(local_batch).to(device=hp.device, dtype=torch.float)    #Bp*D, Ci, H, W

        'Pass through the model'
        pred = model_label['model'](local_batch)   #Bp*D, Cm, H, W

        '2D to 3D'
        pred = converter.convert_2D_to_3D(pred)     #Bp, Cm, H, W, D


    elif pseudo_mode=='3D':
        local_batch = local_batch.to(device=hp.device, dtype=torch.float)

        pred = model_label['model'](local_batch)   #B, Cm, H, W, D 

        
    'Loss'
    loss_dice = model_label['criterion']['dice'](pred, y)
    loss_distri = model_label['criterion']['distri'](F.log_softmax(pred, dim=1), y)
    
    loss = loss_dice + loss_distri

    if train:
        loss.backward()
        model_label['optimizer'].step()
        model_label['optimizer'].zero_grad()
    
        
    model_label['loss']['overall'].append(loss.item())
    model_label['loss']['dice'].append(loss_dice.item())   
    model_label['loss']['distri'].append(loss_distri.item())



def main():
    args = parser.parse_args()

    'Intialize hyperparameters'
    hp = Hyperparameters(**vars(args))
    hp.load_json()
    hp.set(**vars(args))

    'Import validation and pseudo-labeled (for saving) dataset'
    split_dataset = Split_dataset(hp)
    files_split = split_dataset.split()
    files_validation = files_split['validation']
    files_pseudo = [f for f in split_dataset.files_all if f not in files_split['training']]

    'Dataset and dataloader for validation and pseudo-labeled data'
    params_val = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0,
        'drop_last':False,
    }
    ds_validation = pick_dataset(files_validation)
    validation_set = Dataset_test(ds_validation, transform=hp.tform_testing)
    validation_generator = data.DataLoader(validation_set, **params_val)
    hp.set(**{'num_classes':ds_validation.num_classes,
              'in_channels':ds_validation.in_channels,
              })
    
    ds_pseudo = pick_dataset(files_pseudo)
    pseudo_set = Dataset_test(ds_pseudo, transform=hp.tform_testing)
    pseudo_generator = data.DataLoader(pseudo_set, **params_val)
    
    'Initialize the models'
    for model_name, model_dict in hp.model.items():
        m = get_model(model_name, hp)
        if '2d' in model_name:
            model_2d = create_model(m, model_name, hp)
        elif '3d' in model_name:
            model_3d = create_model(m, model_name, hp)
        else:
            raise ValueError('Model not recognized')

    'Create dict to save results'
    matrix_names = ['dice',
                    'jaccard',
                    # 'hd95',
                    # 'asd',
                    ]
    results = Results(models_list=list(hp.model.keys()), 
                      matrix_list=matrix_names, 
                      mask_dict=ds_validation.mask_dict
                      )
    results_pseudo = Results(models_list=list(hp.model.keys()), 
                             matrix_list=matrix_names, 
                             mask_dict=ds_pseudo.mask_dict
                             )
        
    'Log'
    writer = SummaryWriter(os.path.join(hp.save_path, 'log'))

    'Early stop when loss plateu'
    def stop_early(best, current, count, patience):
        stop = False
        if current>=best:
            count=0
        else:
            count+=1
            if count>=2*patience+1:
                stop=True

        return stop, count
    
    'evaluation metrics'
    count_2d = 0
    count_3d = 0
    epoch_2d = 0
    epoch_3d = 0

    'Print hyperparameters'
    hp.save_json()
    hp.print()

    '''
    Start training and testing
    '''

    'Stage 1) Training  with just label'
    while epoch_2d<hp.transition_epochs and epoch_3d<hp.transition_epochs:
        if epoch_2d==0 and epoch_3d==0:
            print(hp.save_path)
        _, training_generator = create_loaders(hp, epoch_2d)
        for i, data_label in enumerate(training_generator):
            run_warmup([model_2d, model_3d], data_label, hp, train=True)
            for mg in [model_2d, model_3d]:
                if epoch_2d==0 :
                    print(mg['name'], '[%d, %5d]'%(epoch_2d + 1, i + 1), ''.join(['{0}:{1}   '.format(k, np.mean(v)) for k,v in mg['loss'].items()]))
            count_2d+=1
            count_3d+=1
        epoch_2d+=1
        epoch_3d+=1

        'Update the scheduler if not reduce on plateau'
        for mg in [model_2d, model_3d]:
            if not isinstance(mg['scheduler'], optim.lr_scheduler.ReduceLROnPlateau):
                mg['scheduler'].step()
                
            
        'After 1 epoch of training'
        write_tensorboard([model_2d, model_3d], count_2d, writer, 'training')
        if epoch_2d%50==0:
            for mg in [model_2d, model_3d]:
                torch.save(mg['model'].state_dict(), mg['current_model'])

    'Save pseudo labels (first time)'
    with torch.set_grad_enabled(False):
        model_3d['model'] = model_3d['model'].eval()
        for input_group in pseudo_generator:
            input_group['img'] = input_group['img'].to(dtype=torch.float).squeeze(0)
            input_group['corner'] = input_group['corner'].squeeze(0)
            test_runner = Test_runner(input_group, hp, results_pseudo)

            # Run the model
            results_pseudo, mask_pred = test_runner.test(model_3d['model'], model_3d['name'], return_pred=True)
            ds_pseudo.save_pseudo_label(mask_pred, input_group['img_path'][0])
    

    'Stage 2) Training label and pseudo'
    while epoch_2d<hp.max_epochs and epoch_3d<hp.max_epochs:
        'Train 2D'
        model_3d['model'] = model_3d['model'].eval()
        for _ in range(hp.freq_switch):
            training_pseudo_generator, training_generator = create_loaders(hp, epoch_2d)
            for i, (data_pseudo, data_label) in enumerate(zip(training_pseudo_generator, training_generator)):
                count_2d+=1
                run(model_3d, model_2d, data_pseudo, data_label, hp, train=True, pseudo_mode='2D')
                if epoch_2d==0:
                    print(model_2d['name'], '[%d, %5d]'%(epoch_2d + 1, i + 1), ''.join(['{0}:{1}   '.format(k, np.mean(v)) for k,v in model_2d['loss'].items()]))
            epoch_2d+=1

            '(2D) Update the scheduler if not reduce on plateau'
            if not isinstance(model_2d['scheduler'], optim.lr_scheduler.ReduceLROnPlateau):
                model_2d['scheduler'].step()
                    
                
            '(2D) After 1 epoch of training'
            write_tensorboard([model_2d], count_2d, writer, 'training')
            if epoch_2d%50==0:
                torch.save(model_2d['model'].state_dict(), model_2d['current_model'])

            if epoch_2d>=model_dict['training_specs']['max_epochs']:
                break
        
        'Train 3D'
        model_2d['model'] = model_2d['model'].eval()
        for _ in range(hp.freq_switch):
            training_pseudo_generator, training_generator = create_loaders(hp, epoch_3d)
            for i, (data_pseudo, data_label) in enumerate(zip(training_pseudo_generator, training_generator)):
                count_3d+=1
                run(model_2d, model_3d, data_pseudo, data_label, hp, train=True, pseudo_mode='3D')
                if epoch_3d==0:
                    print(model_3d['name'], '[%d, %5d]'%(epoch_3d + 1, i + 1), ''.join(['{0}:{1}   '.format(k, np.mean(v)) for k,v in model_3d['loss'].items()]))
            epoch_3d+=1

            '(3D) Update the scheduler if not reduce on plateau'
            if not isinstance(model_3d['scheduler'], optim.lr_scheduler.ReduceLROnPlateau):
                model_3d['scheduler'].step()
                
            '(3D) After 1 epoch of training'
            write_tensorboard([model_3d], count_3d, writer, 'training')
            if epoch_3d%50==0:
                torch.save(model_3d['model'].state_dict(), model_3d['current_model'])

            'Save pseudo labels'
            if epoch_3d in hp.save_pseudo:
                with torch.set_grad_enabled(False):
                    model_3d['model'] = model_3d['model'].eval()
                    for input_group in pseudo_generator:
                        input_group['img'] = input_group['img'].to(dtype=torch.float).squeeze(0)
                        input_group['corner'] = input_group['corner'].squeeze(0)
                        test_runner = Test_runner(input_group, hp, results_pseudo)

                        # Run the model
                        results_pseudo, mask_pred = test_runner.test(model_3d['model'], model_3d['name'], return_pred=True)
                        ds_pseudo.save_pseudo_label(mask_pred, input_group['img_path'][0])
            

            'Validation (whole volume)'
            with torch.set_grad_enabled(False):
                if epoch_3d%50==0:
                    model_3d['model'] = model_3d['model'].eval()
                    for input_group in validation_generator:
                        input_group['img'] = input_group['img'].to(dtype=torch.float).squeeze(0)
                        input_group['mask_original'] = input_group['mask_original']#.to(dtype=torch.float)
                        input_group['corner'] = input_group['corner'].squeeze(0)
                        test_runner = Test_runner(input_group, hp, results)

                        # Run the model
                        results = test_runner.test(model_3d['model'], model_3d['name'], return_pred=False)

                    'Extract the metrics'
                    model_3d['matrix'] = results.return_validation(model_3d['name'])

                    'Save and display results after test'
                    current_metrics = model_3d['matrix'][matrix_names[0]][0]
                    if model_3d['best'] is None or current_metrics > model_3d['best']:
                        model_3d['best'] = current_metrics
                        torch.save(model_3d['model'].state_dict(), model_3d['saved_model'])
                        model_3d['remark'] = '  ***'
                        
                    # Update the scheduler if reduce on plateau
                    if isinstance(model_3d['scheduler'], optim.lr_scheduler.ReduceLROnPlateau):
                        model_3d['scheduler'].step(torch.tensor([current_metrics]).to(device=hp.device, dtype=torch.float))
                        # Stop early
                        model_3d['stop'], model_3d['stop_count'] = stop_early(model_3d['best'], current_metrics, model_3d['stop_count'], model_3d['patience'])

                    write_tensorboard([model_3d], epoch_3d, writer, 'validation', progress='(validation %d)'%(epoch_3d), loss='matrix')
                    results.reset()

            if epoch_2d>=model_dict['training_specs']['max_epochs']:
                break




if __name__ ==  '__main__':
    main()