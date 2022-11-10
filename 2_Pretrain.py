import pandas as pd
import numpy as np
import scipy.sparse
import scipy.sparse as sp
import yaml
import os
import glob
import pickle
from tqdm import tqdm
import wandb
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from logging import getLogger
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import BiGNNLayer, SparseDropout
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils import get_model, get_trainer
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
from recbole.data import create_dataset
from newmodel import NGCFpretrain, LightGCNpretrain, NGCFconcat, LightGCNconcat, NGCFpretrainMLP, LightGCNpretrainMLP, NGCFpretrain_all64, NGCFpretrain_txt64

# random seed
SEED = 2022
np.random.seed(SEED)
torch.manual_seed(SEED)


"""
arg parser
"""
parser = argparse.ArgumentParser(description='recbole pretrain')

# for loop
parser.add_argument('--model', type=str, default='NGCFpretrain')
parser.add_argument('--feature', type=str, default='price')
parser.add_argument('--dataset', type=str, default='bayc')
#
parser.add_argument('--item_cut', type=int, default=3)
parser.add_argument('--config', type=str, default='pretrain')
#
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

"""
arg parser -> variables
"""
if args.model == 'NGCFpretrain':
    MODEL = NGCFpretrain
elif args.model == 'LightGCNpretrain':
    MODEL = LightGCNpretrain
elif args.model == 'NGCFpretrain_all64':
    MODEL = NGCFpretrain_all64
elif args.model == 'NGCFpretrain_txt64':
    MODEL = NGCFpretrain_txt64
FEATURE = args.feature
DATASET = args.dataset
ITEM_CUT = args.item_cut
CONFIG = f'config/fixed_config_{args.config}.yaml'

"""
main functions
"""


def objective_function(config_dict=None, config_file_list=None):

    config = Config(model=MODEL, dataset=DATASET,
                    config_dict=config_dict, config_file_list=config_file_list)

    if FEATURE == 'img' or FEATURE == 'price':
        config['embedding_size'] = 64
    elif FEATURE == 'txt':
        if DATASET in ['azuki', 'bayc', 'meebits']:
            config['embedding_size'] = 1800
        elif DATASET in ['coolcats', 'doodles']:
            config['embedding_size'] = 1500

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # dataset creating and filtering # convert atomic files -> Dataset
    dataset = create_dataset(config)

    # dataset splitting # convert Dataset -> Dataloader
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = MODEL(config, train_data.dataset).to(config['device'])

    # trainer loading and initialization
    trainer = Trainer(config, model)

    """ (1) training """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    """ (2) testing """
    trainer.eval_collector.data_collect(train_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    return {
        'model': MODEL.__name__,
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def main_HPO():

    # read file
    with open(r'config/fixed_config_pretrain.yaml', 'r') as file:
        parameter_dict = yaml.load(file, Loader=yaml.FullLoader)
        parameter_dict[FEATURE] = True  # 이거해줌
        parameter_dict['user_inter_num_interval'] = f'[{ITEM_CUT},inf)'  # 이거해줌
        # save file
        with open(f'config/fixed_config_pretrain_{MODEL.__name__}_{FEATURE}_{DATASET}_{ITEM_CUT}.yaml', 'w') as file:
            documents = yaml.dump(parameter_dict, file)

    hp = HyperTuning(objective_function=objective_function, algo="exhaustive",
                     max_evals=50, params_file=f'hyper/{MODEL.__name__}.hyper', fixed_config_file_list=[f'config/fixed_config_pretrain_{MODEL.__name__}_{FEATURE}_{DATASET}_{ITEM_CUT}.yaml'])

    # run
    hp.run()
    # export result to the file
    hp.export_result(
        output_file=f'hyper_result/{MODEL.__name__}_{FEATURE}_{DATASET}_{ITEM_CUT}.result')
    # print best parameters
    print('best params: ', hp.best_params)
    # save best parameters
    with open(f'hyper_result/{MODEL.__name__}_{FEATURE}_{DATASET}_{ITEM_CUT}.best_params', 'w') as file:
        documents = yaml.dump(hp.best_params, file)
    # print best result
    best_result = hp.params2result[hp.params2str(hp.best_params)]

    best_result_df = pd.DataFrame.from_dict(
        best_result['test_result'], orient='index', columns=[f'{DATASET}'])
    best_result_df.to_csv(
        result_path + f'{MODEL.__name__}-{FEATURE}-{DATASET}.csv', index=True)


def main():

    # DATASET
    with open(f'hyper/NGCF_{DATASET}_3.best_params', 'r') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)
    # FEATURE, ITEM_CUT
    with open(r'config/fixed_config_pretrain.yaml', 'r') as file:
        parameter_dict = yaml.load(file, Loader=yaml.FullLoader)
        parameter_dict[FEATURE] = True  # 이거해줌
        parameter_dict['user_inter_num_interval'] = f'[{ITEM_CUT},inf)'  # 이거해줌
        with open(r'config/fixed_config_pretrain_temp.yaml', 'w') as file:
            documents = yaml.dump(parameter_dict, file)

    # config
    config = Config(model=MODEL, dataset=DATASET, config_file_list=[
                    'config/fixed_config_pretrain_temp.yaml'])
    if FEATURE == 'img':
        config['embedding_size'] = 64  # 초기 임베딩 사이즈를 말함
    elif FEATURE == 'txt':
        if DATASET in ['azuki', 'bayc', 'meebits']:
            config['embedding_size'] = 1800
        elif DATASET in ['coolcats', 'doodles']:
            config['embedding_size'] = 1500
    elif FEATURE == 'price':
        config['embedding_size'] = 64

    # set best hyperparameter setting
    config['hidden_size_list'] = [int(i) for i in best_params['hidden_size_list'].replace(
        '[', '').replace(']', '').split(',')]  # string to list
    config['node_dropout'] = best_params['node_dropout']
    config['message_dropout'] = best_params['message_dropout']
    config['reg_weight'] = best_params['reg_weight']
    config['learning_rate'] = best_params['learning_rate']

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # dataset creating and filtering # convert atomic files -> Dataset
    dataset = create_dataset(config)

    # dataset splitting # convert Dataset -> Dataloader
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = MODEL(config, train_data.dataset).to(config['device'])

    # trainer loading and initialization
    trainer = Trainer(config, model)

    """ (1) training """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    """ (2) testing """
    trainer.eval_collector.data_collect(train_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)  # , model_file=checkpoint_file)
    print('FINAL TEST RESULT')
    print(test_result)


""" 
main 
"""
if __name__ == '__main__':

    # wandb
    wandb.init(project="64-family",
               name=f'{MODEL.__name__}_{FEATURE}_{DATASET}', entity="nft-recommender")
    wandb.config.update(args)

    # result path
    result_path = './result_64family/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    main_HPO()
