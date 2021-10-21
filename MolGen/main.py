import copy
from datetime import datetime
import os
import random

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

import torch
from torch.utils.data import ConcatDataset

from src.datasets.dataset import get_dataset
from src.models.model_builder import get_model, ModelOpt
from src.tokenizers.CharTokenizer import CharTokenizer
from src.train.train import Trainer
from src.train.evaluate import generate_smiles, get_stats, gen_till_train
from src.train.reinforcement import policy_gradients
from src.utils.reward_fn import qed_reward
from src.utils.utils import get_max_smiles_len

RDLogger.DisableLog('rdApp.*')
torch.autograd.set_detect_anomaly(True)

# set seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def main():

    config = {
        'data_path': './data/gdb/gdb13/gdb13.rand1M.smi',
        'tokenizer_path': './data/tokenizers/gdb13CharTokenizer.json',
        'to_load': True,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }
    
    config['max_len'] = get_max_smiles_len(config['data_path'])
    print(config['max_len'])
    
    tokenizer = CharTokenizer(config['tokenizer_path'])

    dataset = get_dataset(config['data_path'],
                          tokenizer=tokenizer,
                          max_len=config['max_len'],
                          to_load=config['to_load'])

    model_config = {
        'n_embd': 512,
        'd_model': 1024,
        'n_layers': 4,
        'num_heads': 8,
        'vocab_size': tokenizer.vocab_size,
        'block_size': 512,
        'proj_size': 512,
        'attn_dropout_rate': 0.1,
        'proj_dropout_rate': 0.1,
        'resid_dropout_rate': 0.1,
        'padding_idx': tokenizer.pad_token_id,

    }

    train_config = {
        'batch_size': 1024,
        'epochs': 3,
        'optimizer': torch.optim.Adam,
        'criterion': torch.nn.CrossEntropyLoss,
    }


    rl_config = {
        'batch_size': 500,
        'epochs': 150,
        'discount_factor': 0.99,
        'reward_fn': qed_reward,
        'optimizer': torch.optim.Adam,
        'max_len': 100,
        # 'fn': lambda x: x * 10, 
    }

    eval_config = {
        'save_path': './data/results/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
        'size': 10000,
        'temprature': 1,
        'max_len': 100,
        
        
    }


    model = get_model(ModelOpt.GPT, **model_config).to(config['device'])
    print(sum(p.numel() for p in model.parameters()))
   
    optim = train_config['optimizer'](model.parameters())
    criterion = train_config['criterion']()

    trainer = Trainer(dataset, model, optim, criterion)
    trainer.train(train_config['epochs'], train_config['batch_size'], config['device'])

    old_model = copy.deepcopy(model)
    generated_molecules = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=eval_config['temprature'],
                                          size=eval_config['size'],
                                          max_len=eval_config['max_len'],
                                          device=config['device'])
    
    train_set = dataset.molecules if config['to_load'] else config['data_path']
    
    get_stats(train_set=train_set,
              generated_smiles=generated_molecules,
              save_path=eval_config['save_path'],
              folder_name='pre_RL')

    policy_gradients(model=model,
                     tokenizer=tokenizer,
                     **rl_config)
    
    generated_molecules = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=eval_config['temprature'],
                                          size=eval_config['size'],
                                          max_len=eval_config['max_len'],
                                          device=config['device'])

    get_stats(train_set=train_set,
              generated_smiles=generated_molecules,
              save_path=eval_config['save_path'],
              folder_name='post_RL')

    mean, std = gen_till_train(old_model,
                           dataset,
                           device=config['device'])
    print(f'Took on average {mean}+- {std} Generations for generate a mol from the test set before PG.')

    mean, std = gen_till_train(model,
                           dataset,
                           device=config['device'])
    print(f'Took on average {mean}+- {std} Generations for generate a mol from the test set after PG.')
    
if __name__ == "__main__":
    main()
