import torch
from torch.utils import data
import torch.nn.functional as F

import os
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random

current_dir = os.path.dirname(os.path.realpath(__file__))

from dataset import pick_dataset, Dataset_test
from model import get_model
from utilities import Split_dataset, Test_runner, Results
from hyperparameters import Hyperparameters


torch.manual_seed(101)
torch.cuda.manual_seed(101)
np.random.seed(101)
random.seed(101)

parser = argparse.ArgumentParser(description="testing pipeline")
parser.add_argument("--save_path", default='la_4_pseudo', type=str, help="Path (or folder) to save the model")
parser.add_argument("--hp_name", default='hyperparameters.json', type=str, help="Name of the hyperparameter file")
parser.add_argument("--device_num", default=0, type=int, help="CUDA device number")
parser.add_argument("--num_workers", default=0, type=int, help="number of workers")
parser.add_argument("--resume_name", default='current_model', type=str, help="which saved model to load")



def main():
    'Initialize from training parameters'
    args = parser.parse_args()
    hp = Hyperparameters(**vars(args))
    hp.load_json()
    hp.set(**vars(args))

    'Import files names'
    split_dataset = Split_dataset(hp)
    files_split = split_dataset.split()
    filename = hp.resume_name
    files_testing = files_split['extra'] if len(files_split['extra'])>0 else files_split['testing']

    'Dataset and dataloader'
    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': hp.num_workers,
        'drop_last':False,
    }     # for data generator
    ds_testing = pick_dataset(files_testing)
    testing_set = Dataset_test(ds_testing, transform=hp.tform_testing)
    testing_generator = data.DataLoader(testing_set, **params)

    
    'Initialize the 3D model'
    model_name = 'model_3d'
    models_dict = {}
    
    model = get_model(model_name, hp)
    model = model.to(hp.device)
    saved_model = os.path.join(hp.save_path, model_name, hp.resume_name+'.pth')
    model.load_state_dict(torch.load(saved_model))
    model.eval()
    models_dict.update({model_name:model})


    'Create dict to save results'
    matrix_names = ['dice',
                    'jaccard',
                    'hd95',
                    'asd',
                    ]
    results = Results(models_list=list(models_dict.keys()), 
                      matrix_list=matrix_names, 
                      mask_dict=ds_testing.mask_dict
                      )

    'Print hyperparameters'
    hp.print()
    
    
    'Testing (whole volume)'
    with torch.set_grad_enabled(False):
        for input_group in tqdm(testing_generator):
            input_group['img'] = input_group['img'].to(dtype=torch.float).squeeze(0)
            input_group['corner'] = input_group['corner'].squeeze(0)
            test_runner = Test_runner(input_group, hp, results)

            for model_name, model in models_dict.items():
                results, mask_pred = test_runner.test(model, model_name, return_pred=True)
                # You can save the predicted masks (i.e. mask_pred) here if needed

    'Print results'
    results.print()

    'Save results'
    results.save(hp.save_path, save_name=filename)


if __name__ ==  '__main__':
    main()