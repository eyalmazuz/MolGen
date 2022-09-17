import copy
from datetime import datetime
import math
import os
import random

import numpy as np
import pandas as pd
from parso import parse
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
from src.utils.utils import parse_arguments

torch.autograd.set_detect_anomaly(True)

# set seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def main():

    parser = parse_arguments() 

    device = torch.device(parser.device) 

    if parser.train_predictor:
        bs1_data = pd.read_csv(parser.predictor_dataset_path)
        train, test = train_test_split(bs1_data, test_size=0.2, random_state=42, shuffle=True,)

        print(train.shape)
        train.reset_index(inplace=True)
        test.reset_index(inplace=True)

        predictor_tokenizer = CharTokenizer(parser.predictor_toeknizer_path,
                                            data_path='./data/ic50_smiles.smi')

        train_dataset = BS1Dataset(train, predictor_tokenizer)
        test_dataset = BS1Dataset(test, predictor_tokenizer)

        predictor_config = BertConfig(n_embd=parser.predictor_n_embd,
                                      d_model=parser.predictor_d_model,
                                      n_layers=parser.predictor_n_layers,
                                      num_heads=parser.predictor_num_heads,
                                      vocab_size=tokenizer.vocab_size,
                                      block_size=parser.predictor_block_size,
                                      proj_size=parser.predictor_proj_size,
                                      attn_dropout_rate=parser.predictor_attn_dropout_rate,
                                      proj_dropout_rate=parser.predictor_proj_dropout_rate,
                                      resid_dropout_rate=parser.predictor_resid_dropout_rate,
                                      padding_idx=tokenizer.pad_token_id)
        predictor_model = Bert(predictor_config) 
        predictor_model = predictor_model.to('cuda')

        predictor_trainer = PredictorTrainer(train_dataset,
                                            test_dataset,
                                            predictor_model,
                                            torch.optim.Adam(predictor_model.parameters(), lr=5e-3),
                                            torch.nn.MSELoss(),)

        predictor_trainer.train(parser.predictor_epochs, parser.predictor_batch_size, device)

        torch.save(predictor_model, parser.predictor_save_path)

    print(parser.device)
    
    max_smiles_len = get_max_smiles_len(parser.dataset_path) + 50
    #max_smiles_len = 256
    tokenizer = CharTokenizer(parser.tokenizer_path, parser.dataset_path)

    dataset = get_dataset(data_path=parser.dataset_path,
                          tokenizer=tokenizer,
                          use_scaffold=parser.use_scaffold,
                          max_len=max_smiles_len)

    model = get_model(parser.model,
                      n_embd=parser.n_embd,
                      d_model=parser.d_model,
                      n_layers=parser.n_layers,
                      num_heads=parser.num_heads,
                      vocab_size=tokenizer.vocab_size,
                      block_size=parser.block_size,
                      proj_size=parser.proj_size,
                      attn_dropout_rate=parser.attn_dropout_rate,
                      proj_dropout_rate=parser.proj_dropout_rate,
                      resid_dropout_rate=parser.resid_dropout_rate,
                      padding_idx=tokenizer.pad_token_id).to(device)

    if parser.load_pretrained:
        print(parser.pretrained_path)
        model.load_state_dict(torch.load(parser.pretrained_path))

    print(str(model))
    print(sum(p.numel() for p in model.parameters()))

    dataset_name = parser.dataset_path[parser.dataset_path.rfind('/')+1:parser.dataset_path.rfind('.')]

    reward_fn = get_reward_fn(reward_names=parser.reward_fns,
                            paths=parser.predictor_paths,
                            multipliers=parser.multipliers,)

    print(str(reward_fn))
    if hasattr(reward_fn, 'reward_fns'):
        print([str(fn) for fn in reward_fn.reward_fns])

    eval_save_path = parser.save_path + \
                                f'_{str(model)}' + \
                                f'_{dataset_name}' + \
                                f'_RlBatch_{str(parser.rl_batch_size)}' + \
                                f'_RlEpochs_{str(parser.rl_epochs)}' + \
                                f'_Reward_{str(reward_fn)}' + \
                                f'_Scaffold_{str(parser.use_scaffold)}' + \
                                f'_discount_{str(parser.discount_factor)}'

    print(eval_save_path)
    
    if not parser.load_pretrained and parser.do_train:
        optim = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        trainer = Trainer(dataset, model, optim, criterion)
        trainer.train(parser.epochs, parser.batch_size, device)

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path, exist_ok=True)

    torch.save(model.state_dict(), f"{eval_save_path}/pre_rl.pt")
    
    if parser.use_scaffold:
        generated_smiles = generate_smiles_scaffolds(model=model,
                                                    tokenizer=tokenizer,
                                                    scaffolds=dataset.scaffolds,
                                                    temprature=parser.temprature,
                                                    size=parser.eval_size,
                                                    max_len=parser.eval_max_len,
                                                    device=device)
    else:
        generated_smiles = generate_smiles(model=model,
                                           tokenizer=tokenizer,
                                           temprature=parser.temprature,
                                           size=parser.eval_size,
                                           max_len=parser.eval_max_len,
                                           device=device)
    
    
    if hasattr(reward_fn, 'eval'):
        reward_fn.eval = True

    get_stats(train_set=dataset,
              generated_smiles=generated_smiles,
              save_path=eval_save_path,
              folder_name='pre_RL',
              reward_fn=reward_fn)

    policy_gradients(model=model,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    optimizer=torch.optim.Adam,
                    batch_size=parser.rl_batch_size,
                    epochs=parser.rl_epochs,
                    discount_factor=parser.discount_factor,
                    max_len=parser.rl_max_len,
                    do_eval=parser.do_eval,
                    use_scaffold=parser.use_scaffold,
                    train_set=dataset,
                    scaffolds=dataset.scaffolds if parser.use_scaffold else [],
                    eval_steps=parser.eval_steps,
                    save_path=eval_save_path,
                    temprature=parser.rl_temprature,
                    size=parser.rl_size,
                    device=device,)

    torch.save(model.state_dict(), f"{eval_save_path}/rl.pt")
    
    generated_smiles = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=parser.temprature,
                                          size=parser.eval_size,
                                          max_len=parser.eval_max_len,
                                          device=parser.device)
                                          
    reward_fn.multiplier = lambda x: x

    get_stats(train_set=dataset,
              generated_smiles=generated_smiles,
              save_path=eval_save_path,
              folder_name='post_RL',
              run_moses=True,
              reward_fn=reward_fn)

if __name__ == "__main__":
    main()
