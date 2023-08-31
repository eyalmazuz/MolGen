import copy
from datetime import datetime
import math
import os
import random
import sys

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
from src.tokenizers.BPETokenizer import BPETokenizer
from src.train.train import Trainer, PredictorTrainer
from src.train.evaluate import generate_smiles, generate_smiles_scaffolds, get_stats, gen_till_train
from src.train.reinforcement import policy_gradients
from src.utils.reward_fn import QEDReward, get_reward_fn
from src.utils.utils import TaskOpt, get_max_smiles_len
from src.utils.utils import parse_arguments

torch.autograd.set_detect_anomaly(True)

# set seeds
# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

def main():

    args = parse_arguments()

    device = torch.device(args.device)

    if args.train_predictor:
        bs1_data = pd.read_csv(args.predictor_dataset_path)
        train, test = train_test_split(bs1_data, test_size=0.2, random_state=42, shuffle=True,)

        print(train.shape)
        train.reset_index(inplace=True)
        test.reset_index(inplace=True)

        predictor_tokenizer = CharTokenizer(args.predictor_toeknizer_path,
                                            data_path='./data/ic50_smiles.smi')

        train_dataset = BS1Dataset(train, predictor_tokenizer)
        test_dataset = BS1Dataset(test, predictor_tokenizer)

        predictor_config = BertConfig(n_embd=args.predictor_n_embd,
                                      d_model=args.predictor_d_model,
                                      n_layers=args.predictor_n_layers,
                                      num_heads=args.predictor_num_heads,
                                      vocab_size=tokenizer.vocab_size,
                                      block_size=args.predictor_block_size,
                                      proj_size=args.predictor_proj_size,
                                      attn_dropout_rate=args.predictor_attn_dropout_rate,
                                      proj_dropout_rate=args.predictor_proj_dropout_rate,
                                      resid_dropout_rate=args.predictor_resid_dropout_rate,
                                      padding_idx=tokenizer.pad_token_id)
        predictor_model = Bert(predictor_config) 
        predictor_model = predictor_model.to('cuda')

        predictor_trainer = PredictorTrainer(train_dataset,
                                            test_dataset,
                                            predictor_model,
                                            torch.optim.Adam(predictor_model.parameters(), lr=5e-3),
                                            torch.nn.MSELoss(),)

        predictor_trainer.train(args.predictor_epochs, args.predictor_batch_size, device)

        torch.save(predictor_model, args.predictor_save_path)

    print(args.device)
    
    max_smiles_len = get_max_smiles_len(args.dataset_path) + 50
    #max_smiles_len = 256
    print(f'{max_smiles_len=}')

    if args.tokenizer == "Char":
        tokenizer = CharTokenizer(args.tokenizer_path, args.dataset_path)

    elif args.tokenizer == "BPE":
        tokenizer = BPETokenizer(args.tokenizer_path, args.dataset_path, vocab_size=500)

    else:
        raise ValueError("Tokenizer type not supported")

    dataset = get_dataset(data_path=args.dataset_path,
                          tokenizer=tokenizer,
                          use_scaffold=args.use_scaffold,
                          max_len=max_smiles_len)

    model = get_model(args.model,
                      n_embd=args.n_embd,
                      d_model=args.d_model,
                      n_layers=args.n_layers,
                      num_heads=args.num_heads,
                      vocab_size=tokenizer.vocab_size,
                      block_size=args.block_size,
                      proj_size=args.proj_size,
                      attn_dropout_rate=args.attn_dropout_rate,
                      proj_dropout_rate=args.proj_dropout_rate,
                      resid_dropout_rate=args.resid_dropout_rate,
                      padding_idx=tokenizer.pad_token_id).to(device)

    if args.load_pretrained:
        print(args.pretrained_path)
        model.load_state_dict(torch.load(args.pretrained_path))

    print(str(model))
    print(sum(p.numel() for p in model.parameters()))

    dataset_name = args.dataset_path[args.dataset_path.rfind('/')+1:args.dataset_path.rfind('.')]

    reward_fn = get_reward_fn(reward_names=args.reward_fns,
                            paths=args.predictor_paths,
                            multipliers=args.multipliers,)

    print(str(reward_fn))
    if hasattr(reward_fn, 'reward_fns'):
        print([str(fn) for fn in reward_fn.reward_fns])

    eval_save_path = args.save_path + \
                                f'_{str(model)}' + \
                                f'_{dataset_name}' + \
                                f'_RlBatch_{str(args.rl_batch_size)}' + \
                                f'_RlEpochs_{str(args.rl_epochs)}' + \
                                f'_Reward_{str(reward_fn)}' + \
                                f'_Scaffold_{str(args.use_scaffold)}' + \
                                f'_discount_{str(args.discount_factor)}'

    print(eval_save_path)
    
    if not args.load_pretrained and args.do_train:
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        trainer = Trainer(dataset, model, optim, criterion)
        trainer.train(args.epochs, args.batch_size, device)

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path, exist_ok=True)

    with open(f'{eval_save_path}/command.txt', 'w') as f:
        f.write(' '.join(sys.argv))

    torch.save(model.state_dict(), f"{eval_save_path}/pre_rl.pt")
    
    if args.use_scaffold:
        print("Using scaffolds")
        generated_smiles = generate_smiles_scaffolds(model=model,
                                                    tokenizer=tokenizer,
                                                    scaffolds=dataset.scaffolds,
                                                    temprature=args.temprature,
                                                    size=args.eval_size,
                                                    max_len=args.eval_max_len,
                                                    device=device)
    else:
        generated_smiles = generate_smiles(model=model,
                                           tokenizer=tokenizer,
                                           temprature=args.temprature,
                                           size=args.eval_size,
                                           max_len=args.eval_max_len,
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
                    batch_size=args.rl_batch_size,
                    epochs=args.rl_epochs,
                    discount_factor=args.discount_factor,
                    max_len=args.rl_max_len,
                    do_eval=args.do_eval,
                    use_scaffold=args.use_scaffold,
                    train_set=dataset,
                    scaffolds=dataset.scaffolds if args.use_scaffold else [],
                    eval_steps=args.eval_steps,
                    save_path=eval_save_path,
                    temprature=args.rl_temprature,
                    size=args.rl_size,
                    no_batched_rl=args.no_batched_rl,
                    device=device,)

    torch.save(model.state_dict(), f"{eval_save_path}/rl.pt")
    
    generated_smiles = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=args.temprature,
                                          size=args.eval_size,
                                          max_len=args.eval_max_len,
                                          device=args.device)
                                          
    reward_fn.multiplier = lambda x: x

    get_stats(train_set=dataset,
              generated_smiles=generated_smiles,
              save_path=eval_save_path,
              folder_name='post_RL',
              run_moses=True,
              reward_fn=reward_fn)

if __name__ == "__main__":
    main()
