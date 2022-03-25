import copy
from datetime import datetime
import math
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
from src.models.model_builder import get_model 
from src.models.gpt import GPTValue
from src.models.bert import Bert, BertConfig
from src.tokenizers.CharTokenizer import CharTokenizer
from src.train.train import Trainer, PredictorTrainer
from src.train.evaluate import generate_smiles, generate_smiles_scaffolds, get_stats, gen_till_train
from src.train.reinforcement import policy_gradients
from src.utils.reward_fn import QEDReward, IC50Reward, get_reward_fn
from src.utils.utils import TaskOpt, get_max_smiles_len
from src.utils.utils import eval_arguments, model_arguments, general_arguments, rl_train_arguments, train_arguments, predictor_arguments

torch.autograd.set_detect_anomaly(True)

# set seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def main():

    general_parser = general_arguments() 
    train_parser = train_arguments()
    rl_parser = rl_train_arguments()
    model_parser = model_arguments()
    eval_parser = eval_arguments()
    predictor_parser = predictor_arguments()
    
    load_pretrained = True

    if predictor_parser.train_predictor:
        bs1_data = pd.read_csv(predictor_parser.predictor_dataset_path)
        train, test = train_test_split(bs1_data, test_size=0.2, random_state=42, shuffle=True,)

        print(train.shape)
        train.reset_index(inplace=True)
        test.reset_index(inplace=True)

        predictor_tokenizer = CharTokenizer(predictor_parser.predictor_toeknizer_path,
                                            data_path='./data/ic50_smiles.smi')

        train_dataset = BS1Dataset(train, predictor_tokenizer)
        test_dataset = BS1Dataset(test, predictor_tokenizer)

        predictor_config = BertConfig(n_embd=predictor_parser.n_embd,
                                      d_model=predictor_parser.d_model,
                                      n_layers=predictor_parser.n_layers,
                                      num_heads=predictor_parser.num_heads,
                                      vocab_size=tokenizer.vocab_size,
                                      block_size=predictor_parser.block_size,
                                      proj_size=predictor_parser.proj_size,
                                      attn_dropout_rate=predictor_parser.attn_dropout_rate,
                                      proj_dropout_rate=predictor_parser.proj_dropout_rate,
                                      resid_dropout_rate=predictor_parser.resid_dropout_rate,
                                      padding_idx=tokenizer.pad_token_id)
        predictor_model = Bert(predictor_config) 
        predictor_model = predictor_model.to('cuda')

        predictor_trainer = PredictorTrainer(train_dataset,
                                            test_dataset,
                                            predictor_model,
                                            torch.optim.Adam(predictor_model.parameters(), lr=5e-3),
                                            torch.nn.MSELoss(),)

        predictor_trainer.train(predictor_parser.predictor_epochs, predictor_parser.predictor_batch_size, general_parser.device)

        torch.save(predictor_model, predictor_parser.predictor_save_path)

    else:
        predictor_tokenizer = CharTokenizer('./data/tokenizers/predictor_tokenizer.json',
                                            data_path='./data/ic50_smiles.smi')

        predictor_model = torch.load('./data/models/predictor_model.pt') 
 
    print(general_parser.device)
    
    max_smiles_len = get_max_smiles_len(general_parser.dataset_path) + 50
    
    tokenizer = CharTokenizer(general_parser.tokenizer_path, general_parser.dataset_path)

    dataset = get_dataset(data_path=general_parser.dataset_path,
                          tokenizer=tokenizer,
                          max_len=max_smiles_len)

    model = get_model(general_parser.model,
                      n_embd=model_parser.n_embd,
                      d_model=model_parser.d_model,
                      n_layers=model_parser.n_layers,
                      num_heads=model_parser.num_heads,
                      vocab_size=tokenizer.vocab_size,
                      block_size=model_parser.block_size,
                      proj_size=model_parser.proj_size,
                      attn_dropout_rate=model_parser.attn_dropout_rate,
                      proj_dropout_rate=model_parser.proj_dropout_rate,
                      resid_dropout_rate=model_parser.resid_dropout_rate,
                      padding_idx=tokenizer.pad_token_id).to(general_parser.device)

    # if load_pretrained:
    #     print(f'./data/models/gpt_pre_rl_{tokenizer_name}.pt')
    #     model.load_state_dict(torch.load(f'./data/models/gpt_pre_rl_{tokenizer_name}.pt'))

    print(str(model))
    print(sum(p.numel() for p in model.parameters()))

    dataset_name = general_parser.dataset_path[general_parser.dataset_path.rfind('/')+1:general_parser.dataset_path.rfind('.')]

    reward_fn = get_reward_fn(reward_name=rl_parser.reward_fn,
                            predictor_path=rl_parser.predictor_path,
                            predictor=predictor_model,
                            tokenizer=predictor_tokenizer,)

    eval_save_path = eval_parser.save_path + \
                                f'_{str(model)}' + \
                                f'_{dataset_name}' + \
                                f'_RlBatch_{str(rl_parser.rl_batch_size)}' + \
                                f'_RlEpochs_{str(rl_parser.rl_epochs)}' + \
                                f'_Reward_{str(reward_fn)}' + \
                                f'_discount_{str(rl_parser.discount_factor)}'

    print(eval_save_path)
    
    if not load_pretrained:
        optim = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        trainer = Trainer(dataset, model, optim, criterion)
        trainer.train(train_parser.epochs, train_parser.batch_size)

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path, exist_ok=True)

    torch.save(model.state_dict(), f"{eval_save_path}/pre_rl.pt")
    
    generated_smiles = generate_smiles(model=model,
                                        tokenizer=tokenizer,
                                        temprature=eval_parser.temprature,
                                        size=eval_parser.eval_size,
                                        max_len=eval_parser.eval_max_len,
                                        device=general_parser.device)
    
    
    reward_fn.multiplier = lambda x: x
    get_stats(train_set=dataset,
              generated_smiles=generated_smiles,
              save_path=eval_save_path,
              folder_name='pre_RL',
              reward_fn=reward_fn if str(reward_fn) == 'IC50' else None)

    reward_fn.multiplier = lambda x: math.exp(x / 3)
    policy_gradients(model=model,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    optimizer=torch.optim.Adam(),
                    batch_size=rl_parser.rl_batch_size,
                    epochs=rl_parser.rl_epochs,
                    discount_factor=rl_parser.discount_factor,
                    max_len=rl_parser.rl_max_len,
                    device=general_parser.device,
                    do_eval=rl_parser.do_eval,
                    eval_steps=rl_parser.eval_steps,
                    save_path=eval_save_path,
                    temprature=rl_parser.rl_temprature,
                    train_set=dataset)

    torch.save(model.state_dict(), "{eval_save_path}/rl.pt")
    
    generated_smiles = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=eval_parser.temprature,
                                          size=eval_parser.eval_size,
                                          max_len=eval_parser.eval_max_len,
                                          device=general_parser.device)
                                          
    reward_fn.multiplier = lambda x: x

    get_stats(train_set=dataset,
              generated_smiles=generated_smiles,
              save_path=eval_save_path,
              folder_name='post_RL',
              run_moses=True,
              reward_fn=reward_fn if str(reward_fn) == 'IC50' else None)

if __name__ == "__main__":
    main()
