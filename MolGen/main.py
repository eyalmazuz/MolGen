import copy
from datetime import datetime
import os
import random

import numpy as np
import pandas as pd
import rdkit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# lg = RDLogger.logger()
# lg.setLevel(RDLogger.CRITICAL)

from sklearn.model_selection import train_test_split
import torch

from src.datasets.get_dataset import get_dataset
from src.datasets.bs1_dataset import BS1Dataset
from src.models.model_builder import get_model, ModelOpt 
from src.models.gpt import GPTValue
from src.models.property_predictor import Predictor, PredictorConfig
from src.tokenizers.CharTokenizer import CharTokenizer
from src.train.train import Trainer, PredictorTrainer
from src.train.evaluate import generate_smiles, generate_smiles_scaffolds, get_stats, gen_till_train
from src.train.reinforcement import policy_gradients
from src.utils.reward_fn import QEDReward, IC50Reward, get_reward_fn
from src.utils.utils import TaskOpt, get_max_smiles_len

torch.autograd.set_detect_anomaly(True)

# set seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def main():

    
    train_predictor = True

    if train_predictor:
        bs1_data = pd.read_csv('./data/csvs/bs1.csv')
        train, test = train_test_split(bs1_data, test_size=0.2, random_state=42, shuffle=True,)

        train.reset_index(inplace=True)
        test.reset_index(inplace=True)

        predictor_tokenizer = CharTokenizer('./data/tokenizers/predictor_tokenizer.json',
                                            data_path='./data/ic50_smiles.smi', build_scaffolds=False)
        max_len = max(map(len, train.Smiles)) + 2

        print(max_len)
        train_dataset = BS1Dataset(train, predictor_tokenizer, max_len=max_len)
        test_dataset = BS1Dataset(test, predictor_tokenizer, max_len=max_len)

        model_config = {
            'n_embd': 256,
            'd_model': 256,
            'n_layers': 4,
            'num_heads': 8,
            'vocab_size': predictor_tokenizer.vocab_size,
            'block_size': 256,
            'proj_size': 256,
            'attn_dropout_rate': 0.1,
            'proj_dropout_rate': 0.1,
            'resid_dropout_rate': 0.1,
            'padding_idx': predictor_tokenizer.pad_token_id,
        }
        predictor_config = PredictorConfig(**model_config)
        predictor_model = Predictor(predictor_config) 
        predictor_model = predictor_model.to('cuda')

        predictor_trainer = PredictorTrainer(train_dataset,
                                            test_dataset,
                                            predictor_model,
                                            torch.optim.Adam(predictor_model.parameters()),
                                            torch.nn.MSELoss(),)

        predictor_trainer.train(3, 512, 'cuda')

        torch.save(predictor_model, './data/models/predictor_model.pt')

    datasets = {
            '1': ['gdb/gdb13/gdb13', 'gdb13'],
            '2': ['moses', 'moses'],
            '3': ['zinc/zinc250k', 'zinc']
    }
    
    dataset, tokenizer = datasets[os.environ['SLURM_ARRAY_TASK_ID']]

    task = {'regular': TaskOpt.REGULAR, 'constrained': TaskOpt.CONSTRAINED}[os.environ['TASK']] 
    print(task)
    config = {
        'data_path': f'./data/{dataset}.smi',
        'tokenizer_path': f'./data/tokenizers/{tokenizer}ScaffoldCharTokenizer.json',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'model': ModelOpt.GPT
    }

    print(config['device'])
    
    config['max_len'] = get_max_smiles_len(config['data_path']) + 50
    print(config['max_len'])
    
    tokenizer = CharTokenizer(config['tokenizer_path'], config['data_path'], build_scaffolds=config['model'] == ModelOpt.TRANSFORMER)

    dataset = get_dataset(config['model'],
                          task,
                          data_path=config['data_path'],
                          tokenizer=tokenizer,
                          max_len=config['max_len'])

    model_config = {
        'n_embd': 256,
        'd_model': 256,
        'n_layers': 4,
        'num_heads': 8,
        'vocab_size': tokenizer.vocab_size,
        'block_size': 256,
        'proj_size': 256,
        'attn_dropout_rate': 0.1,
        'proj_dropout_rate': 0.1,
        'resid_dropout_rate': 0.1,
        'padding_idx': tokenizer.pad_token_id,

    }

    train_config = {
        'batch_size': 512,
        'epochs': 1,
        'optimizer': torch.optim.Adam,
        'criterion': torch.nn.CrossEntropyLoss,
    }


    rl_config = {
        'batch_size': 5,
        'epochs': 10,
        'discount_factor': 0.99,
        # 'reward_fn': QEDReward(),
        'predictor_path': None,
        'optimizer': torch.optim.Adam,
        'max_len': 150,
        'size': 250,
    }

    eval_config = {
        'save_path': './data/results/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
        'size': 250,
        'temprature': 1,
        'max_len': 150,        
    }

    model = get_model(config['model'], **model_config).to(config['device'])
    print(str(model))
    print(sum(p.numel() for p in model.parameters()))

    dataset_name = config['data_path'][config['data_path'].rfind('/')+1:config['data_path'].rfind('.')]

    reward_fn = get_reward_fn(reward_name='QED', **rl_config)    

    eval_config['save_path'] = eval_config['save_path'] + \
                                f'_{str(model)}' + \
                                f'_{dataset_name}' + \
                                f'_RlBatch_{str(rl_config["batch_size"])}' + \
                                f'_RlEpochs_{str(rl_config["epochs"])}' + \
                                f'_Reward_{str(reward_fn)}' + \
                                f'_discount_{str(rl_config["discount_factor"])}'

    print(eval_config['save_path'])
    
    optim = train_config['optimizer'](model.parameters())
    criterion = train_config['criterion']()

    trainer = Trainer(dataset, model, optim, criterion)
    trainer.train(train_config['epochs'], train_config['batch_size'], config['device'])

    if not os.path.exists(f"{eval_config['save_path']}"):
        os.makedirs(f"{eval_config['save_path']}", exist_ok=True)

    torch.save(model.state_dict(), f"{eval_config['save_path']}/pre_rl.pt")
    
    old_model = copy.deepcopy(model)
    if config['model'] == ModelOpt.TRANSFORMER:
        generated_smiles = generate_smiles_scaffolds(model=model,
                                            tokenizer=tokenizer,
                                            scaffolds=dataset.scaffolds,
                                            temprature=eval_config['temprature'],
                                            size=eval_config['size'],
                                            max_len=eval_config['max_len'],
                                            device=config['device'])
    else:
        generated_smiles = generate_smiles(model=model,
                                            tokenizer=tokenizer,
                                            temprature=eval_config['temprature'],
                                            size=eval_config['size'],
                                            max_len=eval_config['max_len'],
                                            device=config['device'])
    
    
    get_stats(train_set=dataset,
              generated_smiles=generated_smiles,
              save_path=f"{eval_config['save_path']}",
              folder_name='pre_RL')

    policy_gradients(model=model,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    **rl_config,
                    device=config['device'],
                    do_eval=True,
                    eval_steps=10,
                    save_path=eval_config['save_path'],
                    temprature=eval_config['temprature'],
                    train_set=dataset)

    torch.save(model.state_dict(), f"{eval_config['save_path']}/rl.pt")
    
    generated_smiles = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=eval_config['temprature'],
                                          size=eval_config['size'],
                                          max_len=eval_config['max_len'],
                                          device=config['device'])
                                          

    get_stats(train_set=dataset,
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
