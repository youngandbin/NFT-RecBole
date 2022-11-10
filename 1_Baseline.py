import torch
import pandas as pd
import numpy as np
import scipy.sparse
import yaml
from tqdm import tqdm
import shutil
import os
import glob
import wandb
import argparse

from logging import getLogger
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils import get_model, get_trainer
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

# random seed
SEED = 2022
np.random.seed(SEED)
torch.manual_seed(SEED)

"""
arg parser
"""
parser = argparse.ArgumentParser(description='recbole baseline')

# for loop
parser.add_argument('--model', type=str, default='BPR')
parser.add_argument('--dataset', type=str, default='bayc')
#
parser.add_argument('--item_cut', type=int, default=3)
parser.add_argument('--config', type=str, default='baseline')
# 
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

"""
arg parser -> variables
"""
MODEL = args.model
DATASET = args.dataset
ITEM_CUT = args.item_cut
CONFIG = f'config/fixed_config_{args.config}.yaml'

"""
main functions
"""
def main():
    
    config = Config(model=MODEL, dataset=DATASET, config_file_list=[CONFIG])
    config['user_inter_num_interval'] = f'[{ITEM_CUT},inf)'
    
    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # dataset creating and filtering # convert atomic files -> Dataset
    dataset = create_dataset(config)

    # dataset splitting # convert Dataset -> Dataloader
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    """ (1) training """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    """ (2) testing """
    trainer.eval_collector.data_collect(train_data)
    test_result = trainer.evaluate(test_data)
    # save result
    result_df = pd.DataFrame.from_dict(test_result, orient='index', columns=[f'{DATASET}'])
    result_df.to_csv(result_path + f'{MODEL}-{DATASET}.csv', index=True)


"""
main
"""
if __name__ == '__main__':
    
    # wandb
    wandb.init(project="nft-recommender", name=f'{MODEL}_{DATASET}', entity="nft-recommender")
    wandb.config.update(args)
    
    # result path
    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    main()
    
