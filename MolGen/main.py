import copy
from datetime import datetime
import os
import random

import numpy as np
import rdkit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# lg = RDLogger.logger()
# lg.setLevel(RDLogger.CRITICAL)


import torch

from src.datasets.dataset import get_dataset
from src.models.model_builder import get_model, ModelOpt
from src.tokenizers.CharTokenizer import CharTokenizer
from src.train.train import Trainer
from src.train.evaluate import generate_smiles, get_stats, gen_till_train
from src.train.reinforcement import policy_gradients
from src.utils.reward_fn import QEDReward
from src.utils.utils import get_max_smiles_len

torch.autograd.set_detect_anomaly(True)

# set seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def main():

    datasets = {
            '1': ['gdb/gdb13/gdb13', 'gdb13'],
            '2': ['moses', 'moses'],
            '3': ['zinc/zinc250k', 'zinc']
    }
    
    dataset, tokenizer = datasets[os.environ['SLURM_ARRAY_TASK_ID']]

    config = {
        'data_path': f'./data/{dataset}.smi',
        'tokenizer_path': f'./data/tokenizers/{tokenizer}CharTokenizer.json',
        'to_load': True,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }
    
    config['max_len'] = get_max_smiles_len(config['data_path'])
    print(config['max_len'])
    
    tokenizer = CharTokenizer(config['tokenizer_path'], config['data_path'])

    dataset = get_dataset(config['data_path'],
                          tokenizer=tokenizer,
                          max_len=config['max_len'],
                          to_load=config['to_load'])

    model_config = {
        'n_embd': 512,
        'd_model': 512,
        'n_layers': 2,
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
        'discount_factor': 0.97,
        'reward_fn': QEDReward(),
        'optimizer': torch.optim.Adam,
        'max_len': 150,
        'size': 25000,
    }

    eval_config = {
        'save_path': './data/results/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
        'size': 100000,
        'temprature': 1,
        'max_len': 150,        
    }

    model = get_model(ModelOpt.RECURRENT, **model_config).to(config['device'])
    print(str(model))
    print(sum(p.numel() for p in model.parameters()))

    dataset_name = config['data_path'][config['data_path'].rfind('/')+1:config['data_path'].rfind('.')]

    eval_config['save_path'] = eval_config['save_path'] + f'_{str(model)}' + f'_{dataset_name}' + f'_{str(rl_config["reward_fn"])}'
    print(eval_config['save_path'])
    
    optim = train_config['optimizer'](model.parameters())
    criterion = train_config['criterion']()

    # trainer = Trainer(dataset, model, optim, criterion)
    # trainer.train(train_config['epochs'], train_config['batch_size'], config['device'])

    if not os.path.exists(f"{eval_config['save_path']}"):
        os.makedirs(f"{eval_config['save_path']}", exist_ok=True)

    torch.save(model.state_dict(), f"{eval_config['save_path']}/pre_rl.pt")
    train_set = dataset.molecules if config['to_load'] else config['data_path']

    old_model = copy.deepcopy(model)
    generated_smiles = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=eval_config['temprature'],
                                          size=eval_config['size'],
                                          max_len=eval_config['max_len'],
                                          device=config['device'])
    
    
    get_stats(train_set=train_set,
              generated_smiles=generated_smiles,
              save_path=f"{eval_config['save_path']}",
              folder_name='pre_RL')

    
    policy_gradients(model=model,
                     tokenizer=tokenizer,
                     **rl_config,
                     device=config['device'],
                     do_eval=True,
                     eval_steps=10,
                     save_path=eval_config['save_path'],
                     temprature=eval_config['temprature'],
                     train_set=train_set)

    torch.save(model.state_dict(), f"{eval_config['save_path']}/rl.pt")
    
    generated_smiles = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=eval_config['temprature'],
                                          size=eval_config['size'],
                                          max_len=eval_config['max_len'],
                                          device=config['device'])
                                          

    get_stats(train_set=train_set,
              generated_smiles=generated_smiles,
              save_path=f"{eval_config['save_path']}",
              folder_name='post_RL',
              run_moses=True)

    # mean, std = gen_till_train(old_model,
    #                        dataset,
    #                        device=config['device'])
    # print(f'Took on average {mean}+- {std} Generations for generate a mol from the test set before PG.')

    # mean, std = gen_till_train(model,
    #                        dataset,
    #                        device=config['device'])
    # print(f'Took on average {mean}+- {std} Generations for generate a mol from the test set after PG.')
    
if __name__ == "__main__":
    main()
