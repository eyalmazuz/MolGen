import argparse
from datetime import datetime
from email.policy import default
from enum import Enum
import os
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from tqdm import tqdm, trange

class ModelOpt(Enum):
	RECURRENT = 1
	GPT = 2
	TRANSFORMER = 3

class TaskOpt(Enum):
	REGULAR = 1
	CONSTRAINED = 2
 
def train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for the language modeling task')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of training epochs for the language modeling task')

    return parser.parse_args()

def rl_train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_batch_size', type=int, default=500,
                        help='number of episode to compute batch for the policy gradient')
    parser.add_argument('--rl_epochs', type=int, default=300,
                        help='number of epochs to run for the policy graidnet stage')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='discount factor to use')
    parser.add_argument('--max_len', type=int, default=150,
                        help='the maximum size of molecule the model can generate during the RL stage')
    parser.add_argument('--rl_size', type=int, default=25000,
                        help='number of molecules to generate on each eval step during the RL stage')
    parser.add_argument('--reward_fn', type=str, default='QED', choices=['QED', 'IC50'],
                        help='reward function to use during the rl stage')
    parser.add_argument('--do_eval', type=bool, default=True,
                        help='eval the model during the RL stage')
    parser.add_argument('--eval_steps', type=int, default=10,
                        help='every how many steps do eval during the RL stage')
    parser.add_argument('--rl_temprature', type=float, default=1,
                        help='temprature during the RL stage')
    parser.add_argument('--predictor_path', type=str, default=None,
                        help='predictor path for the IC50 reward function')
    return parser.parse_args()


def eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./data/results/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                        help='path to where to save the results')
    parser.add_argument('--eval_size', type=int, default=25000,
                        help='number of molecules to generate during final evaluation')
    parser.add_argument('--eval_max_len', type=int, default=150,
                        help='the maximum size of molecule the model can generate during the final evalutation stage')
    parser.add_argument('--temprature', type=float, default=1,
                        help='softmax temprature during the final evaluation')

    return parser.parse_args()


def model_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_embd', type=int, default=512,
                        help='model embedding size')
    parser.add_argument('--d_model', type=int, default=1024,
                        help='model ffn size')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='number of ltsm/decoder layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of attention heads') 
    parser.add_argument('--block_size', type=int, default=512,
                        help='the maximum length of token for the model')
    parser.add_argument('--proj_size', type=int, default=256,
                        help='projection size for the attnetion')
    parser.add_argument('--attn_dropout_rate', type=float, default=0.1,
                        help='attention dropout rate')
    parser.add_argument('--proj_dropout_rate', type=float, default=0.1,
                        help='projection dropout rate')
    parser.add_argument('--resid_dropout_rate', type=float, default=0.1,
                        help='residual layers dropout rate')

    return parser.parse_args()

def predictor_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_dataset_path', type=str, default='./data/csvs/bs1.csv')
    parser.add_argument('--predictor_tokenizer_path', type=str, default='./data/tokenizers/predictor_tokenizer.json')
    parser.add_argument('--predictor_save_path,', type=str, default='./data/models/predictor_model.pt')
    parser.add_argument('--train_predictor', type=bool, default=False)
    parser.add_argument('--predictor_batch_size', type=int, default=32)
    parser.add_argument('--predictor_epochs', type=int, default=10)
    parser.add_argument('--n_embd', type=int, default=512,
                        help='model embedding size')
    parser.add_argument('--d_model', type=int, default=1024,
                        help='model ffn size')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='number of ltsm/decoder layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of attention heads') 
    parser.add_argument('--block_size', type=int, default=512,
                        help='the maximum length of token for the model')
    parser.add_argument('--proj_size', type=int, default=256,
                        help='projection size for the attnetion')
    parser.add_argument('--attn_dropout_rate', type=float, default=0.1,
                        help='attention dropout rate')
    parser.add_argument('--proj_dropout_rate', type=float, default=0.1,
                        help='projection dropout rate')
    parser.add_argument('--resid_dropout_rate', type=float, default=0.1,
                        help='residual layers dropout rate')

    return parser.parse_args()



def general_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/gdb/gdb13/gdb13.smi',
                        help='dataset to trian for language modeling')
    parser.add_argument('--tokenizer_path', type=str, default='./data/tokenizers/gdb13CharTokenizer.json',
                        help='path to tokenizer')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', default=ModelOpt.GPT, type=lambda opt: ModelOpt[opt], choices=list(ModelOpt))
    
    return parser.parse_args()

def generate_and_save_plot(values: List[float],
                           plot_func,
                           xlabel: str,
                           ylabel: str,
                           title: str,
                           save_path: str,
                           name: str,
                           **kwargs) -> None:
    """
    Generates a plot and saves it.
    """
    
    plot = plot_func(values, **kwargs)
    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plot.figure.savefig(f'{save_path}/{name}.png')
    plt.clf()


def get_max_smiles_len(data_path: str) -> int:
    """
    Returns the length of the molecule which has the longest SMILES string.
    this is used for padding in the tokenizer when traning the model.  
    """
    if os.path.isdir(data_path):
        max_len = 0
        for path in os.listdir(data_path):
            full_path = os.path.join(data_path, path)
            file_max_len = len(max(open(full_path, 'r'), key=len))
            max_len = file_max_len if file_max_len > max_len else max_len
    else:
        max_len = len(max(open(data_path, 'r'), key=len).strip())
    
    return max_len

def sample(model,
           start_token: int,
           size: int,
           max_len: int,
           temprature: int,
           device,):

    x = torch.tensor([start_token] * size, dtype=torch.long).to(device)
    for k in trange(max_len, leave=False):
        logits = model(x)

        if isinstance(logits, tuple):
                logits = logits[0]

        logits = logits[:, -1, :] / temprature
        probs = F.softmax(logits, dim=-1)
        idxs = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, idxs), dim=1)

    return x

def sample_scaffodls(model,
           start_token: int,
           enc_inp,
           enc_padding_mask,
           size: int,
           max_len: int,
           temprature: int,
           device,):

    x = torch.tensor([[start_token]] * size, dtype=torch.long).to(device)
    enc_inp = torch.tensor([enc_inp] * size, dtype=torch.long).to(device)
    enc_padding_mask = torch.tensor([enc_padding_mask] * size, dtype=torch.long).to(device)
    for k in trange(max_len, leave=False):
        logits = model(enc_inp=enc_inp, dec_inp=x, enc_padding_mask=enc_padding_mask)

        if isinstance(logits, tuple):
                logits = logits[0]

        logits = logits[:, -1, :] / temprature
        probs = F.softmax(logits, dim=-1)
        idxs = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, idxs), dim=1)

    return x
    
