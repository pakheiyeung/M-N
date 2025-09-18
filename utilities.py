import math
import torch

import glob
import os
import numpy as np
from itertools import cycle, islice
from scipy import ndimage
from medpy import metric
import statistics
import pandas as pd
import pickle
import nibabel as nib

current_dir = os.path.dirname(os.path.realpath(__file__))


def write_tensorboard(models_list, num, writer, category, progress=False, loss='loss'):
    dict_loss = {}
    dict_lr = {}
    for mg in models_list:
        for key in mg[loss]:
            # check if the loss is empty
            if mg[loss][key]==[]:
                return
            else:
                break 
        
        for param_group in mg['optimizer'].param_groups:
            current_lr = param_group['lr']
        for k,v in mg[loss].items():
            temp = dict_loss[k] if k in dict_loss else {}
            temp.update({mg['name']:np.mean(v)})
            dict_loss.update({k:temp})
        dict_lr.update({mg['name']:current_lr})
        
        if progress:
            print(mg['name'], progress, ''.join(['{0}:{1}   '.format(k, np.mean(v)) for k,v in mg[loss].items()]), 'learning rate: %.6e' %(current_lr), mg['remark'])
        
        for key in mg[loss]:
            mg[loss].update({key:[]})
        mg['remark']=''
        
    for k,v in dict_loss.items():
        writer.add_scalars(os.path.join(category, k), v, num)
    writer.add_scalars(os.path.join(category, 'lr'), dict_lr, num)


def one_hot_encoding(mask, num_classes=None):
    if num_classes is None:
        num_classes = int(mask.max())

    mask_onehot = torch.moveaxis(torch.nn.functional.one_hot(torch.from_numpy(mask.copy()).type(torch.LongTensor), num_classes+1), -1, 0)    #+1 because add the background
    mask_onehot = mask_onehot.numpy()

    
    return mask_onehot


class Split_dataset:
    def __init__(self, hp, print_stat=True):
        self.hp = hp
        self.print_stat = print_stat
        self.import_dataset()

    def import_dataset(self):
        
        'Get the files'
        if self.hp.dataset_type=='nih_pancreas':
            if self.hp.os_system == 'local':
                self.hp.img_path = '/home/hugoyeung/Desktop/Datasets/NIH_Pancreas/Pancreas-CT'
            else:
                raise ValueError('Unknown platform')
            files = glob.glob(os.path.join(self.hp.img_path, '*', '*', '*'))
            files_extra = []
        elif self.hp.dataset_type=='la':
            if self.hp.os_system == 'local':
                self.hp.img_path = '/home/hugoyeung/Desktop/Datasets/LA/Training Set'
            else:
                raise ValueError('Unknown platform')
            files = glob.glob(os.path.join(self.hp.img_path, '*', 'lgemri.nrrd'))
            files_extra = glob.glob(os.path.join(self.hp.img_path.replace('Training', 'Testing'), '*', 'lgemri.nrrd'))
        else:
            raise ValueError('Unknown dataset type')
        
        'Sort the files'
        self.files = sorted(files)
        self.files_extra = sorted(files_extra) 
        self.files_all = self.files + self.files_extra
    
    def split(self):
        '''
        Return a dictionary of files for training, validation, and testing
        data_split_dict = {'training':[...], 'validation':[...], 'testing':[...], 'extra':[...]}
        '''

        'Split the files'
        data_split_dict = {}
        num_files = len(self.files)
        for data_type in ['training', 'validation', 'testing']:
            fs = []
            for s, e in zip(self.hp.data_split[data_type][0::2], self.hp.data_split[data_type][1::2]):
                'if both are integers, use them as indices, otherwise use them as portion'
                if isinstance(s, int) and isinstance(e, int):
                    fs += self.files[s:e]
                else:
                    fs += self.files[int(s*num_files):int(e*num_files)]
                
            
            'Add to the dictionary'
            if self.hp.data_split[data_type+'_use']=='all':
                pass
            elif isinstance(self.hp.data_split[data_type+'_use'], int):
                if data_type == 'training':
                    'if training, repeat the files to match the number of the whole training set'
                    fs_len = len(fs)
                    fs = list(islice(cycle(fs[0:self.hp.data_split[data_type+'_use']]), fs_len))
                else:
                    fs = fs[0:self.hp.data_split[data_type+'_use']]
            else:
                raise ValueError('Unknown data split portion')
            
            if len(fs)==0:
                raise ValueError(f"Empty {data_type} set")
            
            data_split_dict.update({data_type:fs})
        
        'Check if they overlap'
        for k1, v1 in data_split_dict.items():
            for k2, v2 in data_split_dict.items():
                if k1!=k2:
                    if bool(set(v1) & set(v2)):
                        raise ValueError(f"Overlap between {k1} and {k2}")
        
        'Print statistics'
        if self.print_stat:
            print(self.hp.dataset_type)
            for k, v in data_split_dict.items():
                print(f"{k}: {len(v)}")
            for k, v in data_split_dict.items():
                print(f"{k}: {v[0]}")
            print("")

        'Add extra'
        data_split_dict.update({'extra':self.files_extra})

        return data_split_dict
    
    


class Results:
    def __init__(self, models_list=None, matrix_list=None, mask_dict=None, results_path=None):
        if models_list is not None and matrix_list is not None and mask_dict is not None:
            self.results = self.initialize(models_list, matrix_list, mask_dict)
            self.models_list = models_list
            self.matrix_list = matrix_list
        elif results_path is not None:
            self.load(results_path)
            self.models_list = [m for m in self.results.keys()]
            self.matrix_list = [m for m in self.results[self.models_list[0]].keys()]

    def initialize(self, models_list, matrix_list, mask_dict):
        '''
        results = {'model1':{
                        'matrix1':{
                                    -1:['overall'],
                                    1:['pancreas'],
                                    2:['liver'],
        '''
        results = {}
        for model in models_list:
            results_model = {}
            for matrix in matrix_list:
                results_matrix = {-1:['overall']}
                for c, v in mask_dict.items():
                    if c!=0:
                        results_matrix.update({c:[v]})
                results_model.update({matrix:results_matrix})
            results.update({model:results_model})
        return results
    
    def reset(self):
        for model in self.results:
            for matrix in self.results[model]:
                for c in self.results[model][matrix].keys():
                    self.results[model][matrix][c] = self.results[model][matrix][c][0:1]
    
    def update(self, result, model_name, matrix, c):
        self.results[model_name][matrix][c].append(result)
    
    def update_overall(self):
        '''
        update the overall from different structures
        results = {'model1':{
                        'matrix1':{
                                    -1:['overall', 0.5, 0.25],
                                    1:['pancreas', 1, 0.5],
                                    2:['liver', 0.5, 0],
        '''
        for model in self.results:
            for matrix in self.results[model]:
                for s in range(1,len(self.results[model][matrix][1])):       #skip 0 because it is class name
                    overall = []
                    for c in self.results[model][matrix].keys():
                        if c==-1:
                            continue
                        else:
                            overall.append(self.results[model][matrix][c][s])
                    try:
                        self.results[model][matrix][-1][s] = statistics.mean(overall)
                    except:
                        self.results[model][matrix][-1].append(statistics.mean(overall))

    def save(self, save_dir, save_name='results'):
        with open(os.path.join(save_dir, save_name+'.pkl'), 'wb') as f:
            pickle.dump(self.results, f)

    def load(self, load_path):
        with open(load_path, 'rb') as f:
            self.results = pickle.load(f)

    def print(self, matrix_exclude=[]):
        'Create dictionary'
        dataframes_dict = {}
        for model in self.results:
            matrix_dict = {}
            for matrix in self.results[model]:
                if matrix in matrix_exclude:
                    continue
                r_list = []
                for r in self.results[model][matrix].values():
                    r_list.append(round(statistics.mean(r[1:]),4))
                matrix_dict.update({matrix:r_list})
                matrix_ref = matrix
            dataframes_dict.update({model:matrix_dict})

        'Create list of class names for indexing'
        class_names = []
        for r in self.results[model][matrix_ref].values():
            class_names.append(r[0])

        'Create dataframe'
        dataframes_list = []
        models_list = []
        for model, df in dataframes_dict.items():
            dataframes_list.append(pd.DataFrame(df, index=class_names))
            models_list.append(model)

        'Concatenate the dataframes'
        df_final = pd.concat(dataframes_list, axis=1, keys=models_list)#.swaplevel(0,1,axis=1).sort_index(axis=1)

        'Print the dataframe'
        print(df_final)

        return df_final

    def return_validation(self, model_name):
        '''
        return dict store each matrix results
        matrix_validation = {'matrix1':[0.5], 'matrix2':[0.7]}
        '''
        matrix_validation = {}
        for matrix in self.results[model_name]:
            matrix_validation.update({matrix:[statistics.mean(self.results[model_name][matrix][-1][1:])]})

        return matrix_validation



    
            

class Test_runner:
    def __init__(self, input_group, hp, results:Results):
        self.hp = hp
        self.input_group = input_group
        self.input_group['mask_original'] = self.input_group['mask_original'].squeeze(0).numpy()    # (1, Cm, H, W, D) -> (Cm, H, W, D)
        self.results = results
        self.prepare()

    def prepare(self):
        'Check 2D or 3D'
        if self.input_group['img'].shape[-1]==1:
            self.mode = '2d'
            self.batch_size = self.hp.batch_size*self.hp.slice_num
        else:
            self.mode = '3d'
            self.batch_size = self.hp.batch_size

        'Get number of classes'
        self.num_classes = self.hp.num_classes+1

        'Create tensor to store count'
        corner = self.input_group['corner']  # (N, 3)
        h, w, d = self.input_group['img'].shape[-3], self.input_group['img'].shape[-2], self.input_group['img'].shape[-1]
        self.pad_size = (1, 
                        self.num_classes, 
                        self.input_group['pad_size'][0].item(),
                        self.input_group['pad_size'][1].item(),
                        self.input_group['pad_size'][2].item()
                        )  # (1, Cm, Hp, Wp, Dp)
        
        count_size = (
                    self.input_group['pad_size'][0].item(),
                    self.input_group['pad_size'][1].item(),
                    self.input_group['pad_size'][2].item()
                    )  # (Hp, Wp, Dp)
        
        self.count_final = torch.zeros(count_size)

        crop_count = torch.ones((h,
                                 w,
                                 d)).to(dtype=torch.float)  # (96, 96, 96)

        'Loop through the batch'
        for i in range(0, self.input_group['img'].size(0)):
            self.count_final[corner[i, 0]:corner[i, 0]+h, corner[i, 1]:corner[i, 1]+w, corner[i, 2]:corner[i, 2]+d] += crop_count

    def test(self, model, model_name, return_pred=False):
        model = model.eval()

        'Loop through the batch'
        pred_final = torch.zeros(self.pad_size)  # (1, Cm, Hp, Wp, Dp)
        h, w, d = self.input_group['img'].shape[-3], self.input_group['img'].shape[-2], self.input_group['img'].shape[-1]
        for b in range(0, self.input_group['img'].size(0), self.batch_size):
            # print(b, 'out of', self.input_group['img'].size(0))
            try:
                img = self.input_group['img'][b:b+self.batch_size] # (b, Ci, 96, 96, 96)
                corner = self.input_group['corner'][b:b+self.batch_size] # (b, 3)
            except:
                img = self.input_group['img'][b:]
                corner = self.input_group['corner'][b:]

            'predict'
            pred = model(img.to(device=self.hp.device)) if self.mode=='3d' else model(img.squeeze(-1).to(device=self.hp.device))
            pred = pred.detach().cpu() if self.mode=='3d' else pred.detach().cpu().unsqueeze(-1)

            'Add the results to storing tensors'
            for i in range(img.shape[0]):
                pred_final[0, :, corner[i, 0]:corner[i, 0]+h, corner[i, 1]:corner[i, 1]+w, corner[i, 2]:corner[i, 2]+d] += pred[i]
                
        'Get whole volume prediction'
        for i in range(self.num_classes):
            pred_final[0, i] = pred_final[0, i] / self.count_final

        'Crop the padding'
        pred_final = pred_final[:, 
                                :, 
                                math.ceil(self.input_group['pad_size'][0].item()/2-self.input_group['original_size'][0].item()/2):math.ceil(self.input_group['pad_size'][0].item()/2+self.input_group['original_size'][0].item()/2),
                                math.ceil(self.input_group['pad_size'][1].item()/2-self.input_group['original_size'][1].item()/2):math.ceil(self.input_group['pad_size'][1].item()/2+self.input_group['original_size'][1].item()/2),
                                math.ceil(self.input_group['pad_size'][2].item()/2-self.input_group['original_size'][2].item()/2):math.ceil(self.input_group['pad_size'][2].item()/2+self.input_group['original_size'][2].item()/2)
                                ]
        
        
        'Resample to original scale and hardmax the prediction'
        h, w, d = self.input_group['original_size'][0].item(), self.input_group['original_size'][1].item(), self.input_group['original_size'][2].item()
        h_f, w_f, d_f = self.input_group['mask_original'].shape[-3], self.input_group['mask_original'].shape[-2], self.input_group['mask_original'].shape[-1]
        pred_arg = np.argmax(pred_final.numpy(), axis=1, keepdims=True)
        pred_arg = ndimage.zoom(pred_arg, (1, 1, h_f/h, w_f/w, d_f/d), order=0)
        pred_hard = torch.zeros((pred_arg.shape[0], self.num_classes, h_f, w_f, d_f))
        pred_hard.scatter_(1, torch.tensor(pred_arg), 1)


        'Calculate metrics'
        pred_hard = pred_hard.squeeze(0).numpy()    # (1, Cm, H, W, D) -> (Cm, H, W, D)
        for c in range(1, pred_hard.shape[0]):
            for matrix in self.results.matrix_list:
                if matrix=='dice' or matrix=='dc':
                    try:
                        result = metric.binary.dc(pred_hard[c], self.input_group['mask_original'][c])
                    except:
                        result = None
                    self.results.update(result, model_name, matrix, c)
                elif matrix=='hd95' or matrix=='95hd' or matrix=='hd':
                    try:
                        result = metric.binary.hd95(pred_hard[c], self.input_group['mask_original'][c])
                    except:
                        result = None
                    self.results.update(result, model_name, matrix, c)
                elif matrix=='asd':
                    try:
                        result = metric.binary.asd(pred_hard[c], self.input_group['mask_original'][c])
                    except:
                        result = None
                    self.results.update(result, model_name, matrix, c)
                elif matrix=='jaccard' or matrix=='jc':
                    try:
                        result = metric.binary.jc(pred_hard[c], self.input_group['mask_original'][c])
                    except:
                        result = None
                    self.results.update(result, model_name, matrix, c)
                else:
                    raise ValueError('Unknown matrix')
        self.results.update_overall()
        
        if return_pred:
            pred_final = ndimage.zoom(pred_final, (1, 1, h_f/h, w_f/w, d_f/d), order=1)
            pred_final = pred_final.squeeze(0)    # (1, Cm, H, W, D) -> (Cm, H, W, D)
            return self.results, pred_final
        else:
            return self.results



class Convert_2D_3D:
    def __init__(self):
        pass

    def convert_3D_to_2D(self, img):
        self.batch_size = img.size(0)
        self.dimension = img.size(-1)

        'Convert 3D to 2D'
        img_2d = torch.moveaxis(img, -1, 1)    # (N, C, H, W, D) -> (N, D, C, H, W)

        img_2d = img_2d.contiguous().view(-1, img_2d.size(-3), img_2d.size(-2), img_2d.size(-1))    # (N, D, C, H, W) -> (N*D, C, H, W)
                    
        return img_2d

    def convert_2D_to_3D(self, img):
        'Convert 2D to 3D'
        img_3d = img.view(self.batch_size, img.size(0)//self.batch_size, img.size(-3), img.size(-2), img.size(-1))    # (N*D, C, H, W) -> (N, D, C, H, W)
        img_3d = torch.moveaxis(img_3d, 1, -1)    # (N, D, C, H, W) -> (N, C, H, W, D)
        
                
        return img_3d


