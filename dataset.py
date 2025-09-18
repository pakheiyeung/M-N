import torch
from torch.utils import data

import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import math
from pydicom import dcmread
import glob
import os
import SimpleITK as sitk


from utilities import one_hot_encoding


def normalize_ct(img, low=-175, up=250):
    '''
    Cap between low and up, then normalize to 0~1
    '''
    image = img.copy()
    image[image<low]=low
    image[image>up] = up
    
    image-=low
    image=image/(up-low)
    
    return image

def normalize_mri(img):
    '''
    Normalize to 0~1
    '''
    image = img.copy()

    #  Normalize image to [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Compute mean and standard deviation
    mean_intensity = np.mean(image)
    std_dev_intensity = np.std(image)

    # Normalize image
    image = (img - mean_intensity) / std_dev_intensity

    return image



def preprocess_ct(data_dict):
    '''
    return data_dict:
        *img: (1,H,W,D)
        mask: ((Cm),H,W,D)
        img_path: str
        vox_space: numpy arrary (resH,resW,resD)
    
    * indicates changed in this function
    '''

    'Normalize'
    data_dict['img'] = normalize_ct(data_dict['img'])


    'Get dimension right'
    data_dict['img'] = np.expand_dims(data_dict['img'], 0)    # (1,H,W,D)
    
    return data_dict

def preprocess_mri(data_dict):
    '''
    return data_dict:
        *img: (1,H,W,D)
        mask: ((Cm),H,W,D)
        img_path: str
        vox_space: numpy arrary (resH,resW,resD)
    
    * indicates changed in this function
    '''
    'Get dimension right'
    if len(data_dict['img'].shape)==3:
        data_dict['img'] = np.expand_dims(data_dict['img'], 0)    # (1,H,W,D)
    elif len(data_dict['img'].shape)==4:
        pass
    else:
        raise ValueError('Image dimension not correct')

    'Normalize'
    temp = []
    for i in range(data_dict['img'].shape[0]):
        temp.append(normalize_mri(data_dict['img'][i:i+1]))
    data_dict['img'] = np.concatenate(temp, axis=0)

    
    return data_dict


def pick_dataset(data_list):
    '''
    data_list: list of file paths
    '''
    if 'NIH_Pancreas' in data_list[0]:
        ds = dataset_nih_pancreas(data_list)
    elif 'LA' in data_list[0]:
        ds = dataset_la(data_list)
    else:
        raise ValueError('Dataset not found')
    
    return ds

class Read_Data:
    '''
    (It's only tested on the LA and nih_pancreas dataset so far, please modify it if you want to use it on other datasets)
    return data_dict:
        img: (H,W,D)
        mask: ((Cm),H,W,D)
        img_path: str
        vox_space: numpy arrary (resH,resW,resD)
    '''

    def __init__(self, img_format='nib', mask_format='nib'):
        self.img_format = img_format
        self.mask_format = mask_format
        self.data_dict = {}


    def read_nib(self, img_path, data_type='img'):
        'Import Image'
        img_data = nib.load(img_path)
        img = img_data.get_fdata()    # (H,W,D)   

        'Import Meta data'
        if data_type=='img':
            vox_space = img_data.header['pixdim'][1:4]  # numpy arrary (resH,resW,resD)
            self.data_dict.update({'vox_space': vox_space, 'img_path': img_path})

        'Update data_dict'
        self.data_dict.update({data_type: img})

    def read_nrrd(self, img_path, data_type='img'):
        'Import Image'
        img_data = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img_data)    # (D,H,W)   

        'Import Meta data'
        if data_type=='img':
            voxel_spacing = np.array((1,1,1))
            self.data_dict.update({'vox_space': voxel_spacing, 'img_path': img_path})

        'Update data_dict'
        self.data_dict.update({data_type: img})

    def read_dcm(self, img_path, data_type='img', key_location='InstanceNumber'):
        # load the DICOM files
        files = []
        for fname in glob.glob(os.path.join(img_path, "*.dcm")):
            files.append(dcmread(fname))


        # skip files with no slice location (eg scout views)
        slices = []
        skipcount = 0
        for f in files:
            if hasattr(f, key_location):
                slices.append(f)
            else:
                skipcount = skipcount + 1
                print('skipped, no SliceLocation')


        # ensure they are in the correct order
        slices = sorted(slices, key=lambda s: getattr(s, key_location))

        # pixel aspects, assuming all slices are the same
        if data_type=='img':
            ps = slices[0].PixelSpacing
            ss = abs(slices[1].ImagePositionPatient[-1])
            voxel_spacing = np.array((ps[0], ps[1], ss))
            self.data_dict.update({'vox_space': voxel_spacing, 'img_path': img_path})
        

        # create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(slices):
            try:
                img2d = s.pixel_array
            except:
                a=1
            img3d[:, :, i] = img2d

        'Update data_dict'
        self.data_dict.update({data_type: img3d})

    def read_npy(self, img_path, data_type='img'):
        'Import Image'
        img = np.load(img_path)

        'Import Meta data'
        if data_type=='img':
            raise ValueError('Only support mask for npy')
        
        'Update data_dict'
        self.data_dict.update({data_type: img})

        
    
    def return_data_dict(self, img_path, mask_path):
        self.data_dict = {}

        for data_type, data_path, file_type in zip(['img', 'mask'], [img_path, mask_path], [self.img_format, self.mask_format]):
            if file_type=='nib':
                self.read_nib(data_path, data_type=data_type)
            elif file_type=='dcm':
                self.read_dcm(data_path, data_type=data_type)
            elif file_type=='npy':
                self.read_npy(data_path, data_type=data_type)
            elif file_type=='nrrd':
                self.read_nrrd(data_path, data_type=data_type)
            else:
                raise ValueError('File type not found')
            
        # 'Check if mask and image shape match'
        # if self.data_dict['img'].shape[-3:] != self.data_dict['mask'].shape[-3:]:
        #     raise ValueError('Mask and image shape not match')
       
        return self.data_dict
    
    

class Transform_Base:
    def __init__(self, prob=0.5, class_channel=False):
        self.class_channel = class_channel  # if True, the first dimension of the mask is class channel
        self.prob = prob
        
    
    def get_aug_zoom_factor(self, zoom_factor=[1,1,1], aug_range=0):
        '''
        zoom_factor: original voxel space/target voxel spacing, list of zoom factor, e.g. [1, 1, 1]
        aug_range: range of zoom factor, e.g. 0.1 means zoom_factor*0.9~zoom_factor*1.1
                    single value or list of n values
        '''
        
        'Get dim'
        dim = len(zoom_factor)

        'Check if aug_range is larger or equal to zero and smaller than 1'
        if isinstance(aug_range, (int, float)):
            if aug_range<0 or aug_range>=1:
                raise ValueError('Augmentation range must be larger or equal to zero and smaller than 1')
        elif isinstance(aug_range, list):
            if np.any(np.array(aug_range)<0) or np.any(np.array(aug_range)>=1):
                raise ValueError('Augmentation range must be larger or equal to zero and smaller than 1')
            if len(aug_range)!=len(zoom_factor):
                raise ValueError('Augmentation range list must have the same length as zoom factor list')
        else:
            raise ValueError('Augmentation range must be single value or list of n values')
        
        'Randomize zoom factor'
        if isinstance(aug_range, (int, float)):
            aug_range = list(1+np.random.rand(1)*2*aug_range-np.array(aug_range))*dim
        elif isinstance(aug_range, list):
            aug_range = list((1+np.random.rand(dim)*2*aug_range-np.array(aug_range)))
        zoom_factor = [zoom_factor[i]*aug_range[i] for i in range(dim)]

        return zoom_factor
    
    def get_pad_from_crop(self, img, mask, crop_size=None, mode='edge'):
        '''
        img: (Ci,H,W) or (Ci,H,W,D)
        mask: ((Cm),H,W) or ((Cm),H,W,D)
        crop_size: list of crop size, e.g. [128, 128, 128]
        mode: padding mode of np.pad, e.g. 'edge' or 'constant'
        '''

        'Get dim'
        dim = len(img.shape)-1

        'Check if crop size dimension is the same as image dimension'
        if crop_size is None or len(crop_size)!=dim:
            raise ValueError('Crop size must have the same dimension as image')
        
        'Pad'
        padding = np.clip(np.array(crop_size)-np.array(img.shape[1:]), 0, None)
        if dim==2:
            img_pad = np.pad(img, ((0,0), 
                                   (math.ceil(padding[0]/2),math.floor(padding[0]/2)), 
                                   (math.ceil(padding[1]/2),math.floor(padding[1]/2))),
                                   mode=mode)
            if self.class_channel:
                mask_pad = np.pad(mask, ((0,0), 
                                         (math.ceil(padding[0]/2),math.floor(padding[0]/2)), 
                                         (math.ceil(padding[1]/2),math.floor(padding[1]/2))),
                                         mode=mode)
            else:
                mask_pad = np.pad(mask, ((math.ceil(padding[0]/2),math.floor(padding[0]/2)), 
                                        (math.ceil(padding[1]/2),math.floor(padding[1]/2))),
                                        mode=mode)
        elif dim==3:
            img_pad = np.pad(img, ((0,0), 
                                   (math.ceil(padding[0]/2),math.floor(padding[0]/2)), 
                                   (math.ceil(padding[1]/2),math.floor(padding[1]/2)),
                                   (math.ceil(padding[2]/2),math.floor(padding[2]/2))),
                                   mode=mode)
            if self.class_channel:
                mask_pad = np.pad(mask, ((0,0), 
                                         (math.ceil(padding[0]/2),math.floor(padding[0]/2)), 
                                         (math.ceil(padding[1]/2),math.floor(padding[1]/2)),
                                         (math.ceil(padding[2]/2),math.floor(padding[2]/2))),
                                         mode=mode)
            else:
                mask_pad = np.pad(mask, ((math.ceil(padding[0]/2),math.floor(padding[0]/2)), 
                                        (math.ceil(padding[1]/2),math.floor(padding[1]/2)),
                                        (math.ceil(padding[2]/2),math.floor(padding[2]/2))),
                                        mode=mode)
        else:
            raise ValueError('Image dimension must be 2 or 3')
        
        return img_pad, mask_pad
    
    def crop(self, img, mask, crop_size=None, top_left=None, prob=None):
        '''
        img: (Ci,H,W) or (Ci,H,W,D)
        mask: ((Cm),H,W) or ((Cm),H,W,D)
        crop_size: list of crop size, e.g. [128, 128, 128], if None, no crop
        top_left: list of top left corner, e.g. [0, 0, 0], if none, random crop
        prob: probability of the patch must contain foreground labels, overwrite self.prob
        '''
        if prob is None:
            prob = self.prob

        'Get dim'
        dim = len(img.shape)-1
        
        'Check if crop size dimension is the same as image dimension'
        if crop_size is None or len(crop_size)!=dim:
            raise ValueError('Crop size must have the same dimension as image')
        
        'Check if top left dimension is the same as image dimension'
        if top_left is not None and len(top_left)!=dim:
            raise ValueError('Top left corner must have the same dimension as image')
        
        if top_left is None:
            random_crop = True
        else:
            random_crop = False

        if crop_size is None:
            return img, mask

        'Check if crop size smaller than image size. If yes, pad'
        if np.any(np.array(crop_size)>np.array(img.shape[1:])):
            img_pad, mask_pad = self.get_pad_from_crop(img, mask, crop_size=crop_size, mode='edge')
        else:
            img_pad = img
            mask_pad = mask

        fore_necessary = np.random.rand()
        while True:
            'Initialize top left corner'
            if random_crop:
                top_left = [np.random.randint(0, img_pad.shape[i+1]-crop_size[i]+1) for i in range(dim)]

            'Crop'
            if dim==3:
                img_crop = img_pad[:, 
                                   top_left[0]:top_left[0]+crop_size[0], 
                                   top_left[1]:top_left[1]+crop_size[1], 
                                   top_left[2]:top_left[2]+crop_size[2]
                                   ]
                if self.class_channel:
                    mask_crop = mask_pad[:, 
                                         top_left[0]:top_left[0]+crop_size[0], 
                                         top_left[1]:top_left[1]+crop_size[1], 
                                         top_left[2]:top_left[2]+crop_size[2]
                                         ]
                else:
                    mask_crop = mask_pad[top_left[0]:top_left[0]+crop_size[0], 
                                        top_left[1]:top_left[1]+crop_size[1], 
                                        top_left[2]:top_left[2]+crop_size[2]
                                        ]
            elif dim==2:
                img_crop = img_pad[:, 
                                   top_left[0]:top_left[0]+crop_size[0], 
                                   top_left[1]:top_left[1]+crop_size[1]
                                   ]
                if self.class_channel:
                    mask_crop = mask_pad[:, 
                                         top_left[0]:top_left[0]+crop_size[0], 
                                         top_left[1]:top_left[1]+crop_size[1]
                                         ]
                else:
                    mask_crop = mask_pad[top_left[0]:top_left[0]+crop_size[0], 
                                        top_left[1]:top_left[1]+crop_size[1]
                                        ]
            
            'Check if satisfy condition'
            if not random_crop:
                break
            if self.class_channel:
                if fore_necessary>prob:  
                    break
                record = np.zeros(mask_crop.shape)
                for i in range(1, mask_crop.shape[0]):
                    record[i] = mask_crop[i]-mask_crop[0]
                if np.any(mask_crop[1:]>0): 
                    break
            else:
                if np.any(mask_crop>0) or fore_necessary>prob:
                    break

        return img_crop, mask_crop
    
    def resample(self, img, mask, zoom_factor=[1,1,1], aug_range=0):
        '''
        img: (Ci,H,W) or (Ci,H,W,D)
        mask: ((Cm),H,W) or ((Cm),H,W,D)
        zoom_factor: original voxel space/target voxel spacing, list of zoom factor, e.g. [1, 1, 1]
        aug_range: range of zoom factor, e.g. 0.1 means zoom_factor*0.9~zoom_factor*1.1
                    single value or list of n values
        '''
        
        'Get augmented zoom factor'
        zoom_factor = self.get_aug_zoom_factor(zoom_factor, aug_range)

        'Resample'
        img_resample = np.stack([ndimage.zoom(img[i], zoom_factor, order=1) for i in range(img.shape[0])], axis=0)
        if self.class_channel:
            mask_resample = np.stack([ndimage.zoom(mask[i], zoom_factor, order=1) for i in range(mask.shape[0])], axis=0)
        else:
            mask_resample = ndimage.zoom(mask, zoom_factor, order=0)

        'Clip image'
        # img = np.clip(img, 0, 1)
        

        'Change float to int for mask'
        if not self.class_channel:
            mask_resample = mask_resample.astype(int)
        
        return img_resample, mask_resample
    
    def patch(self, img, mask,
              crop_size=None, top_left=None, prob=None, source_vox_space=None, target_vox_space=None, aug_range=0):
        '''
        Do crop and then resample. 
        A faster way than calling resample first and crop separately

        img: (Ci,H,W) or (Ci,H,W,D)
        mask: ((Cm),H,W) or ((Cm),H,W,D)
        crop_size: list of patch size, e.g. [128, 128, 128], if None, no patch
        top_left: list of top left corner, e.g. [0, 0, 0], if none, random patch
        prob: probability of the patch must contain foreground labels, overwrite self.prob
        source_vox_space: list of source voxel spacing, e.g. [1, 1, 1], if None, no resample
        target_vox_space: list of target voxel spacing, e.g. [1, 1, 1], if None, no resample
        aug_range: range of zoom factor, e.g. 0.1 means zoom_factor*0.9~zoom_factor*1.1
                    single value or list of n values
        '''
        if prob is None:
            prob = self.prob

        if crop_size is None:
            return img, mask
        
        'Get dim'
        dim = len(img.shape)-1

        'Get resample scale and reverse'
        if source_vox_space is not None and target_vox_space is not None:
            zoom_factor = list(np.array(source_vox_space)/np.array(target_vox_space))
        else:
            zoom_factor = [1]*dim
        zoom_factor = self.get_aug_zoom_factor(zoom_factor, aug_range)
        zoom_factor_reverse = list(1/np.array(zoom_factor))
        crop_size_new = [math.ceil(crop_size[i]*zoom_factor_reverse[i]) for i in range(dim)]
        aug_range = 0   # equal zero because the zoom factor is already augmented

        'Crop'
        img, mask = self.crop(img, 
                              mask, 
                              crop_size=crop_size_new, 
                              top_left=top_left, 
                              prob=prob
                              )
        
        'Resample to target voxel spacing'
        img, mask = self.resample(img, 
                                  mask, 
                                  zoom_factor=zoom_factor,
                                  aug_range=aug_range
                                  )
        
        'Crop again if the patch size is not the same as crop size'
        if img.shape[1:] != tuple(crop_size):
            old_shape = img.shape[1:]
            img, mask = self.crop(img, 
                                  mask, 
                                  crop_size=crop_size, 
                                  top_left=[0]*dim,
                                  prob=0
                                  )
            # print('Patch size from', old_shape, 'to', mask.shape)
        
        return img, mask
    
    def rotate_90(self):
        pass

    def flip(self):
        pass

    def gamma_contrast(self, img, mask, gamma=[0.5,2], prob=None):
        '''
        img: (Ci,H,W) or (Ci,H,W,D), in the range of 0~1
        mask: ((Cm),H,W) or ((Cm),H,W,D)
        gamma: list of upper and lower bound of gamma value, e.g. [0.5, 2]
        prob: probability of augmentation, overwrite self.prob
        '''
        if prob is None:
            prob = self.prob

        'Gamma contrast'
        if np.random.rand()<prob:
            img = img**gamma
        
        return img
    
    




class Transform_3D(Transform_Base):
    def __init__(self, prob=0.5, class_channel=False):
        super().__init__()
        self.class_channel = class_channel  # if True, the first dimension of the mask is class channel
        self.prob = prob

    
    def rotate_90(self, img, mask, exclude_axis=None, prob=None):
        '''
        img: (Ci,H,W,D)
        mask: ((Cm),H,W,D)
        exclude_axis: list of axis to exclude rotation, e.g. [(0,1), (1,2)] means rotate along axis 0 only
        prob: probability of augmentation, overwrite self.prob
        '''
        if prob is None:
            prob = self.prob

        'Rotate 90 degree'
        for a in [(0,1), (1,2), (0,2)]:
            if (exclude_axis is None or a not in exclude_axis) and np.random.rand()<prob:
                k = np.random.randint(0, 4)
                for i in range(img.shape[0]):
                    img[i] = np.rot90(img[i], k, axes=a)
                if self.class_channel:
                    for i in range(mask.shape[0]):
                        mask[i] = np.rot90(mask[i], k, axes=a)
                else:
                    mask = np.rot90(mask, k, axes=a)
        
        return img, mask
    
    def flip(self, img, mask, exclude_axis=None, prob=None):
        '''
        img: (Ci,H,W,D)
        mask: ((Cm),H,W,D)
        exclude_axis: list of axis to exclude flipping, e.g. [0,2] means flip along axis 1 only
        prob: probability of augmentation, overwrite self.prob 
        '''
        if prob is None:
            prob = self.prob

        'Flip'
        for a in [0,1,2]:
            if (exclude_axis is None or a not in exclude_axis) and np.random.rand()<prob:
                for i in range(img.shape[0]):
                    img[i] = np.flip(img[i], a)
                if self.class_channel:
                    for i in range(mask.shape[0]):
                        mask[i] = np.flip(mask[i], a)
                else:
                    mask = np.flip(mask, a)
        
        return img, mask
    
    
    



class Dataset_3D(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ds, transform=None, class_channel=False):
        '''
        ds: (Base_dataset class) dataset class of different 3D datasets
        transform: dictionary of transformations and augmentations
        {
            'rotate_90': {
                'exclude_axis': [(0,1), (1,2)] or None, 
                'prob': 0.5 or None
                },
            'flip': {
                'exclude_axis': [0,2] or None, 
                'prob': 0.5 or None
                },
            'patch': {
                'crop_size': [128, 128, 128],
                'top_left': [0, 0, 0] or None,
                'prob': 0.5 or None,
                (need to add in __getitem)'source_vox_space': [1, 1, 1] or None,
                'target_vox_space': [1, 1, 1] or None, 
                'aug_range': 0.1 or [0,1, 0.2, 0.3]
                },
            'gamma_contrast': {
                'gamma': [0.5, 2], 
                'prob': 0.5
                }
        }
        '''
        self.ds = ds
        self.transform = transform
        self.class_channel = class_channel
        self.Transform = Transform_3D(class_channel=class_channel)
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.ds._len()
    
    
    def __getitem__(self, index):
        data_dict = self.ds.read_and_preprocess_data(index)    # (Ci,H,W,D), ((Cm),H,W,D)

        'Resume data from dict'
        img = data_dict['img']
        mask = data_dict['mask']
        vox_space = data_dict['vox_space']

       
        'Patch'
        if 'patch' in self.transform.keys():
            dict_patch = self.transform['patch'].copy()
            dict_patch.update({'source_vox_space': vox_space})

            img, mask = self.Transform.patch(img, 
                                             mask, 
                                             **dict_patch
                                             )

       
        'Rotate 90'
        if 'rotate_90' in self.transform.keys():
            img, mask = self.Transform.rotate_90(img, 
                                                 mask, 
                                                 **self.transform['rotate_90'],
                                                 )

        'Flip'
        if 'flip' in self.transform.keys():
            img, mask = self.Transform.flip(img, 
                                            mask, 
                                            **self.transform['flip'],
                                            )
            
        'Gamma contrast'
        if 'gamma_contrast' in self.transform.keys():
            img = self.Transform.gamma_contrast(img, 
                                                mask = mask, 
                                                **self.transform['gamma_contrast']
                                                )

        'One-hot encoding'
        if not self.class_channel:
            mask = one_hot_encoding(mask, self.ds.num_classes).astype(np.float32)  #(H,W,D)-->(Cm,H,W,D)
            # print(mask.dtype, np.unique(mask))
        else:
            pass
            # print(mask.dtype, np.unique(mask))

        # print(mask.shape, mask.dtype)
        return img, mask
    
class Dataset_test(data.Dataset):
    '''
    Dataset for creating overlap patches for the whole image during testing
    Only for binary segmentation (self.class_channel=False), i.e. mask shape (H,W,D), please modify __getitem if you want to use it on (Cm,H,W,D)
    '''
    'Characterizes a dataset for PyTorch'
    def __init__(self, ds, transform=None, class_channel=False):
        self.ds = ds
        self.transform = transform
        # TODO Only for binary segmentation (self.class_channel=False), i.e. mask shape (H,W,D), please modify __getitem if you want to use it on (Cm,H,W,D)
        self.class_channel = False
        self.Transform = Transform_3D(class_channel=class_channel)
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.ds._len()
    
    # TODO Only for binary segmentation, i.e. mask shape (H,W,D), please modify it if you want to use it on (Cm,H,W,D)
    def __getitem__(self, index):
        data_dict = self.ds.read_and_preprocess_data(index)    # (Ci,H,W,D), (H,W,D)

        'Resume data from dict'
        img = data_dict['img']
        mask = data_dict['mask']
        image_original = img.copy()
        mask_original = mask.copy()
        vox_space = data_dict['vox_space']
        img_path = data_dict['img_path']

        'If 2D'
        if len(self.transform['resample']['target_vox_space'])==2:
            self.transform['resample']['target_vox_space']=self.transform['resample']['target_vox_space']+[vox_space[2]]
        if len(self.transform['crop']['crop_size'])==2:
            self.transform['crop']['crop_size']=self.transform['crop']['crop_size']+[1]

        'Resample to target voxel spacing'
        if 'resample' in self.transform.keys():
            if self.transform['resample']['target_vox_space'] is not None:
                zoom_factor = np.array(vox_space)/np.array(self.transform['resample']['target_vox_space'])
            else:
                zoom_factor = [1, 1, 1]

            dict_resample = self.transform['resample'].copy()
            dict_resample.pop('target_vox_space')
            dict_resample.update({'zoom_factor': zoom_factor})

            zoom_factor = list(zoom_factor)
            img, mask = self.Transform.resample(img, 
                                                mask, 
                                                **dict_resample
                                                )
            
        'Crop'
        img_patches = []    # (Ci,H,W,D)
        mask_patches = []   # (H,W,D)
        corner_patches = []
        if 'crop' in self.transform.keys():
            crop_size = self.transform['crop']['crop_size']

            img_pad, mask_pad = self.Transform.get_pad_from_crop(img, mask, crop_size=crop_size, mode='edge')

            # Get sliding window corners
            corners=[]
            for ax in [-3,-2,-1]:   #[x,y,z]
                corner = [i for i in range(0, img.shape[ax]-crop_size[ax], max(crop_size[ax]//2, 1))] + [img.shape[ax]-crop_size[ax]]
                
                if len(corner)>1 and corner[-1]<0:
                    raise ValueError('Sliding window incorrect')
                if corner[-1]<0:
                    corner[-1]=0

                corners.append(corner)

            # Sliding windows
            dict_crop = self.transform['crop'].copy()
            dict_crop.update({'crop_size': crop_size})
            for x in corners[0]:
                for y in corners[1]:
                    for z in corners[2]:
                        dict_crop.update({'top_left': [x,y,z]})
                        img_patch, mask_patch = self.Transform.crop(img_pad,
                                                                    mask_pad,
                                                                    **dict_crop
                                                                    )
                        
                        'Histogram matching'
                        if 'histogram_match' in self.transform.keys():
                            img_patch = self.Transform.histogram_match(img_patch, 
                                                                        mask = mask_patch, 
                                                                        **self.transform['histogram_match']
                                                                        )
                        
                        'One-hot encoding'
                        # mask_patch = one_hot_encoding(mask_patch, self.ds.num_classes)  #(H,W,D)-->(Cm,H,W,D)
                        mask_patch = mask_patch[None, ...]

                        img_patches.append(img_patch)
                        mask_patches.append(mask_patch)
                        corner_patches.append(np.array((x,y,z)))

            # To numpy array
            img_patches = np.stack(img_patches, axis=0)   # (N,Ci,H,W,D)
            mask_patches = np.stack(mask_patches, axis=0)   # (N,Cm,H,W,D)
            corner_patches = np.stack(corner_patches, axis=0)   # (N,3)

        'One-hot encoding'
        mask_original = one_hot_encoding(mask_original, self.ds.num_classes)  #(H,W,D)-->(Cm,H,W,D)


        return {'img': img_patches, 
                'mask': mask_patches,
                'image_original': image_original,
                'mask_original': mask_original,
                'corner': corner_patches,
                'original_size': mask.shape,    # (Hs,Ws,Ds)
                'pad_size': mask_pad.shape, # (Hp,Wp,Dp)
                'img_path': img_path,
                }
    

    

class Base_dataset:
    '''
    Base dataset class, please inherit it to create your own dataset class
    e.g. dataset_nih_pancreas, dataset_la
    '''

    def __init__(self):
        self.name = 'base'
        self.img_list = []
        self.mask_list = []
        self.img_format = None    # e.g. 'nib', 'dcm'
        self.mask_format = None
        self.in_channels = None
        self.num_classes = None # not including background
        self.mask_dict = None # e.g. {0: 'background', 1: 'liver', 2: 'kidney', 3: 'spleen', 4: 'pancreas'}
        self.read = Read_Data(img_format=self.img_format, mask_format=self.mask_format)

    def _len(self):
        return len(self.img_list)
    
    def verify_data_dict(self, data_dict):
        if 'img' not in data_dict.keys():
            raise ValueError('Image not found')
        if 'mask' not in data_dict.keys():
            raise ValueError('Mask not found')
        if 'img_path' not in data_dict.keys():
            raise ValueError('Image path not found')
        if 'vox_space' not in data_dict.keys():
            raise ValueError('Voxel spacing not found')
        
        'Check if mask and image shape match'
        if data_dict['img'].shape[-3:] != data_dict['mask'].shape[-3:]:
            raise ValueError('Mask and image shape not match')

    def read_data(self, index):
        data_dict = self.read.return_data_dict(self.img_list[index], self.mask_list[index])

        return data_dict
    
    def preprocess_data(self, data_dict):
        
        return data_dict
    
    def save_pseudo_label(self):
        pass

    def load_pseudo_label_list(self):
        pass
    
    def read_and_preprocess_data(self, index):
        data_dict = self.read_data(index)
        data_dict = self.preprocess_data(data_dict)
        self.verify_data_dict(data_dict)

        return data_dict

    def inspect_dataset(self, preprocess=False):
        img_shape = []
        mask_shape = []
        vox_space = []
        img_min = []
        img_max = []
        mask_min = []
        mask_max = []
        # img_mean = []
        # img_std = []

        img_shape_process = []
        mask_shape_process = []
        vox_space_process = []
        img_min_process = []
        img_max_process = []
        mask_min_process = []
        mask_max_process = []


        'Get statistics'
        for i in tqdm(range(self._len())):
            'Raw data'
            data_dict = self.read_data(i)
            img, mask = data_dict['img'], data_dict['mask']

            img_shape.append(img.shape)
            mask_shape.append(mask.shape)
            vox_space.append(tuple(data_dict['vox_space']))
            img_min.append(img.min())
            img_max.append(img.max())
            mask_min.append(mask.min())
            mask_max.append(mask.max())
           

            'After preprocess'
            if preprocess:
                data_dict = self.preprocess_data(data_dict)
                img, mask = data_dict['img'], data_dict['mask']

                img_shape_process.append(img.shape)
                mask_shape_process.append(mask.shape)
                vox_space_process.append(tuple(data_dict['vox_space']))
                img_min_process.append(img.min())
                img_max_process.append(img.max())
                mask_min_process.append(mask.min())
                mask_max_process.append(mask.max())


        print('Dataset:', self.name, 'Total:', self._len())
        print('Image shape:', np.mean(img_shape, axis=0), np.std(img_shape, axis=0), np.min(img_shape, axis=0), np.max(img_shape, axis=0))
        print('Mask shape:', np.mean(mask_shape, axis=0), np.std(mask_shape, axis=0), np.min(mask_shape, axis=0), np.max(mask_shape, axis=0))
        print('Voxel spacing:', np.mean(vox_space, axis=0), np.std(vox_space, axis=0), np.min(vox_space, axis=0), np.max(vox_space, axis=0))
        print('Image min:', np.mean(img_min), np.std(img_min), np.min(img_min), np.max(img_min))
        print('Image max:', np.mean(img_max), np.std(img_max), np.min(img_max), np.max(img_max))
        print('Mask min:', np.mean(mask_min), np.std(mask_min), np.min(mask_min), np.max(mask_min))
        print('Mask max:', np.mean(mask_max), np.std(mask_max), np.min(mask_max), np.max(mask_max))
        

        if preprocess:
            print('After preprocess:')
            print('Image shape:', np.mean(img_shape_process, axis=0), np.std(img_shape_process, axis=0), np.min(img_shape_process, axis=0), np.max(img_shape_process, axis=0))
            print('Mask shape:', np.mean(mask_shape_process, axis=0), np.std(mask_shape_process, axis=0), np.min(mask_shape_process, axis=0), np.max(mask_shape_process, axis=0))
            print('Voxel spacing:', np.mean(vox_space_process, axis=0), np.std(vox_space_process, axis=0), np.min(vox_space_process, axis=0), np.max(vox_space_process, axis=0))
            print('Image min:', np.mean(img_min_process), np.std(img_min_process), np.min(img_min_process), np.max(img_min_process))
            print('Image max:', np.mean(img_max_process), np.std(img_max_process), np.min(img_max_process), np.max(img_max_process))
            print('Mask min:', np.mean(mask_min_process), np.std(mask_min_process), np.min(mask_min_process), np.max(mask_min_process))
            print('Mask max:', np.mean(mask_max_process), np.std(mask_max_process), np.min(mask_max_process), np.max(mask_max_process))

        

    
class dataset_nih_pancreas(Base_dataset):
    '''
    https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT

    The National Institutes of Health Clinical Center performed 
    82 abdominal contrast enhanced 3D CT scans 
    (~70 seconds after intravenous contrast injection in portal-venous) 
    from 53 male and 27 female subjects.  
    17 of the subjects are healthy kidney donors scanned prior to nephrectomy.  
    The remaining 65 patients were selected by a radiologist from patients 
    who neither had major abdominal pathologies nor pancreatic cancer lesions.  
    Subjects' ages range from 18 to 76 years with a mean age of 46.8 ± 16.7. 
    The CT scans have resolutions of 512x512 pixels with varying pixel sizes 
    and slice thickness between 1.5 - 2.5 mm, acquired on Philips and Siemens MDCT scanners (120 kVp tube voltage).
    A medical student manually performed slice-by-slice segmentations of the pancreas as ground-truth 
    and these were verified/modified by an experienced radiologist.

    Image shape: [512.    512.    236.775] [ 0.    0.    49.1] [512 512 181] [512 512 466]
    Mask shape: [512.    512.    236.775] [ 0.     0.   49.10065555] [512 512 181] [512 512 466]
    Voxel spacing: [0.851   0.851   0.99] [0.092    0.092   0.064] [0.664   0.664   0.5] [0.976     0.976   1.]
    Image min: -1139.2 323.5641512899722 -2048.0 -1024.0
    Image max: 1802.925 677.5661734288393 1155.0 3071.0
    Mask min: 0.0 0.0 0.0 0.0
    Mask max: 1.0 0.0 1.0 1.0
    
    ── Pancreas-CT          
        ├── PANCREAS_0001
            ├── 11-24-2015-PANCREAS0001-Pancreas-18957
                ├── Pancreas-99667
                    ├── 1-001.dcm
                    ├── 1-002.dcm
                    ├── 1-003.dcm 
        ├── PANCREAS_0002
            ├── 11-24-2015-PANCREAS0002-Pancreas-23046
                ├── Pancreas-63502
                    ├── 1-001.dcm
                    ├── 1-002.dcm
                    ├── 1-003.dcm           
    └── Labels           
        ├── label0001.nii.gz        
        ├── label0002.nii.gz        
        ├── label0003.nii.gz         
    
        
    '''
    def __init__(self, data_list):
        self.name = 'nih_pancreas'
        self.img_list = data_list
        self.in_channels = 1
        self.num_classes = 1
        self.mask_list = [os.path.dirname(os.path.dirname(f)) for f in self.img_list]
        self.mask_list = [f.replace('Pancreas-CT', 'Labels') for f in self.mask_list]
        self.mask_list = [f.replace('PANCREAS_', 'label') for f in self.mask_list]
        self.mask_list = [f+'.nii.gz' for f in self.mask_list]
        self.mask_dict = {
            0: 'background', 
            1: 'Pancreas', 
        }
        self.img_format = 'dcm'
        self.mask_format = 'nib'
        self.read = Read_Data(img_format=self.img_format, mask_format=self.mask_format)

    
    def preprocess_data(self, data_dict):
        'Reorient the mask'
        if self.mask_format=='nib':
            data_dict['mask'] = np.moveaxis(data_dict['mask'], 0,1)
            
        data_dict = preprocess_ct(data_dict)
        
        return data_dict
    
    def save_pseudo_label(self, pseudo_label, image_path):
        '''
        Save pseudo label as nii.gz file
        pseudo_label: (H,W,D) or (Cm,H,W,D), numpy or torch tensor
        image_path: path of the original image, used to create the save path'''

        'Convert torch tensor to numpy'
        if isinstance(pseudo_label, torch.Tensor):
            pseudo_label = pseudo_label.detach().cpu().numpy()

        'Squeeze single dimensions'
        pseudo_label = np.squeeze(pseudo_label)

        'Convert one-hot to argmax if needed'
        if len(pseudo_label.shape) == 4:
            pseudo_label = np.argmax(pseudo_label, axis=0)

        'Reorient the mask (revert the preprocess)'
        pseudo_label = np.moveaxis(pseudo_label, 0,1)

        # Ensure integer dtype for mask
        pseudo_label = pseudo_label.astype(np.int16)

        'Save as nii.gz'
        save_path = image_path.replace('Pancreas-CT', 'Pseudo_label')   # .../NIH_Pancreas/Pseudo_label/PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667
        save_path = os.path.dirname(os.path.dirname(save_path)) # .../NIH_Pancreas/Pseudo_label/PANCREAS_0001
        if not os.path.exists(os.path.dirname(save_path)):  # .../NIH_Pancreas/Pseudo_label
            os.makedirs(os.path.dirname(save_path))
        nib_img = nib.Nifti1Image(pseudo_label, affine=np.eye(4))
        nib.save(nib_img, save_path+'.nii.gz')

    def load_pseudo_label_list(self):
        self.mask_list = [f.replace('Pancreas-CT', 'Pseudo_label') for f in self.img_list]
        self.mask_list = [os.path.dirname(os.path.dirname(f)) for f in self.mask_list]
        self.mask_list = [f+'.nii.gz' for f in self.mask_list]
        
    

class dataset_la(Base_dataset):
    '''
    https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/

    A total of 154 3D MRIs from patients with AF are used in for the purpose of this challenge. 
    The original resolution of the data is 0.625 x 0.625 x 0.625 mm³. 
    A large proportion of data were kindly provided by The University of Utah 
    (NIH/NIGMS Center for Integrative Biomedical Computing (CIBC)), 
    while the rest were from multiple other institutes. 
    All clinical data have obtained institutional ethics approval. 
    Each 3D MRI patient data was acquired using a clinical whole-body MRI scanner 
    and contained raw the MRI scan and the corresponding ground truth labels for the left atrial (LA) cavity. 
    The ground truths were manually segmented by experts in the field. 
    The raw MRIs are in grayscale and the segmentation labels are in binary (255 = positive, 0 = negative). 
    The dimensions of the MRIs may vary depending on each patient, 
    however, all MRIs contain exactly 88 slices in the Z axis.

    The dataset is split such that 100 patient data are used for training, 
    and 54 patient data will be used for testing and evaluation. 
    The participants will have access to all the MRIs and their respective labels (LA cavity mask) in the training set, 
    and only the MRIs in the testing set.

    The files are arranged such that each individual file contains one patient data. 
    For each data, the raw MRI “lgemri.nrrd”, 
    the LA cavity segmentation “laendo.nrrd” are provided. 
    “.nrrd” is a medical imaging file format, 
    and can be read using various programming languages.

    Image shape: [ 88.   609.92     609.92] [ 0.  31.94234807   31.94234807] [ 88 576 576] [ 88 640 640]
    Mask shape: [ 88.   609.92      609.92] [ 0.  31.94234807    31.94234807] [ 88 576 576] [ 88 640 640]
    Voxel spacing: [1. 1. 1.] [0. 0. 0.] [1 1 1] [1 1 1]
    Image min: 0.0 0.0 0 0
    Image max: 254.82   0.384   254     255
    Mask min: 0.0 0.0 0 0
    Mask max: 255.0 0.0 255 255
    Image mean: 0.05204402160629756 0.025352935185364325 0.018070529524634244 0.11853453170109153
    Image std: 0.0718448234675487 0.026400997812203505 0.026273936208207855 0.131777796861679
    
    After preprocess:
    Image shape: [ 1.   609.92 609.92  88.  ] [ 0.  31.94234807 31.94234807  0.] [1 576 576  88] [1 640 640  88]
    Mask shape: [609.92 609.92  88.  ] [31.94234807 31.94234807  0.] [576 576  88] [640 640  88]
    Voxel spacing: [0.625 0.625 0.625] [0. 0. 0.] [0.625 0.625 0.625] [0.625 0.625 0.625]
    Image min: 0.0 0.0 0.0 0.0
    Image max: 1.0 0.0 1.0 1.0
    Mask min: 0.0 0.0 0.0 0.0
    Mask max: 1.0 0.0 1.0 1.0

    ── Training Set          
        ├── 0RZDK210BSMWAA6467LU
            ├── lgemri.nrrd
            ├── laendo.nrrd
            ├── lawall.nrrd
        ├── 1D7CUD1955YZPGK8XHJX
            ├── lgemri.nrrd
            ├── laendo.nrrd
            ├── lawall.nrrd

    '''
    def __init__(self, data_list):
        self.name = 'la'
        self.img_list = data_list
        self.in_channels = 1
        self.num_classes = 1
        self.mask_list = [f.replace('lgemri', 'laendo') for f in self.img_list]
        self.mask_dict = {
            0: 'background', 
            1: 'LA cavity', 
        }
        self.img_format = 'nrrd'
        self.mask_format = 'nrrd'
        self.read = Read_Data(img_format=self.img_format, mask_format=self.mask_format)


    def preprocess_data(self, data_dict):
        'Reorient the image'
        data_dict['img'] = np.moveaxis(data_dict['img'], -3, -1)    # (D,H,W)-->(H,W,D)

        'Reorient the mask'
        if self.mask_format=='nrrd':
            data_dict['mask'] = np.moveaxis(data_dict['mask'], -3, -1)
            data_dict['mask'] = data_dict['mask']/255
        
        data_dict = preprocess_mri(data_dict)

        'Correct voxel spacing'
        data_dict['vox_space'] = [0.625, 0.625, 0.625]

        
        return data_dict
    
    def save_pseudo_label(self, pseudo_label, image_path):
        '''
        Save pseudo label as nii.gz file
        pseudo_label: (H,W,D) or (Cm,H,W,D), numpy or torch tensor
        image_path: path of the original image, used to create the save path'''

        'Convert torch tensor to numpy'
        if isinstance(pseudo_label, torch.Tensor):
            pseudo_label = pseudo_label.detach().cpu().numpy()

        'Squeeze single dimensions'
        pseudo_label = np.squeeze(pseudo_label)

        'Convert one-hot to argmax if needed'
        if len(pseudo_label.shape) == 4:
            pseudo_label = np.argmax(pseudo_label, axis=0)

        # Ensure integer dtype for mask
        pseudo_label = pseudo_label.astype(np.int16)

        'Save as nii.gz'
        save_path = image_path.replace('Training Set', 'Pseudo_label')   # .../LA/Pseudo_label/06SR5RBREL16DQ6M8LWS/lgemri.nrrd
        save_path = save_path.replace('Testing Set', 'Pseudo_label')   # .../LA/Pseudo_label/23G5K6IOGZGUOK7SJEUM/lgemri.nrrd
        save_path = os.path.dirname(save_path) # .../LA/Pseudo_label/23G5K6IOGZGUOK7SJEUM
        if not os.path.exists(os.path.dirname(save_path)):  # .../LA/Pseudo_label
            os.makedirs(os.path.dirname(save_path))
        nib_img = nib.Nifti1Image(pseudo_label, affine=np.eye(4))
        nib.save(nib_img, save_path+'.nii.gz')

    def load_pseudo_label_list(self):
        self.mask_format = 'nib'
        self.read = Read_Data(img_format=self.img_format, mask_format=self.mask_format)

        self.mask_list = [f.replace('Training Set', 'Pseudo_label') for f in self.img_list]
        self.mask_list = [f.replace('Testing Set', 'Pseudo_label') for f in self.mask_list]
        self.mask_list = [os.path.dirname(f) for f in self.mask_list]
        self.mask_list = [f+'.nii.gz' for f in self.mask_list]