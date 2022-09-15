import copy
import math
import random

import numpy as np
from numpy.lib.arraysetops import isin
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch
from tqdm import trange, tqdm

from .evaluate import generate_smiles, generate_smiles_scaffolds, get_stats

def policy_gradients(model,
                     tokenizer,
                     reward_fn,
                     optimizer=torch.optim.Adam,
                     batch_size: int=16,
                     epochs: int=100,
                     step_size: float=3e-5,
                     discount_factor: float=0.99,
                     max_len: int=100,
                     use_scaffold=False,
                     scaffolds=[],
                     eval_steps: int=50,
                     do_eval: bool=False,
                     device=torch.device('cuda'),
                     **kwargs):
    print(f'Reinfocement {device}')
    model.train()
    model.to(device)
    optimizer = optimizer(model.parameters(), step_size)

    for epoch in trange(epochs):
        #reward_fn.eval = False
        loss = 0
        batch_reward = 0
        if use_scaffold:
            scaffold = random.choice(scaffolds)
            scaffold_tokens = tokenizer('[BOS]' + scaffold + '[SEP]')['input_ids']
            batch_tokens = generate_smiles_scaffolds(model=model,
                                                     tokenizer=tokenizer,
                                                     scaffolds=[scaffold],
                                                     temprature=kwargs['temprature'],
                                                     size=batch_size,
                                                     num_samples=1,
                                                     batch_size=batch_size // 5,
                                                     max_len=max_len,
                                                     device=device,
                                                     return_smiles=False)
        else:
            batch_tokens = generate_smiles(model=model, tokenizer=tokenizer,
                                temprature=kwargs['temprature'], size=batch_size, batch_size=batch_size // 2, max_len=max_len, device=device, return_smiles=False)

        len_scaffold = len(scaffold_tokens) - 1 if use_scaffold else 0
        batch_smiles = [tokenizer.decode(tokens[len_scaffold+1:-1]) for tokens in batch_tokens]
        batch_rewards = reward_fn(batch_smiles)
        for tokens, reward in tqdm(zip(batch_tokens, batch_rewards), leave=False):
            # if 'Transformer' in str(model):
            #     scaffold = random.sample(kwargs['train_set'].scaffolds, 1)[0]
            #     encoding = tokenizer('[BOS]' + scaffold + '[EOS]')
            #     tokens = model.generate(tokenizer.bos_token_id, tokenizer.eos_token_id, encoding['input_ids'], 
            #                         encoding['padding_mask'], kwargs['temprature'], max_len, device)
            # else:
            #tokens = model.generate(tokenizer.bos_token_id, tokenizer.eos_token_id, kwargs['temprature'], max_len, device)

            # smiles = tokenizer.decode(tokens[1:-1])
            # idx = smiles.find('[PAD]')
            # if idx != -1:
            #     smiles = smiles[:idx]
            
            # # print(f'{smiles=}')
            # reward = reward_fn(smiles)[0]

            discounted_returns = (torch.pow(discount_factor, torch.arange(len(tokens[:-1]), 0, -1)) * reward).to(device)
            
            # if 'Transformer' in str(model):  
            #     y_hat = model(torch.tensor(encoding['input_ids'], [tokens[:-1]], encoding_padding_mask=encoding['padding_mask'], dtype=torch.long).to(device))
            # else:
            y_hat = model(torch.tensor([tokens[:-1]], dtype=torch.long).to(device))
            if isinstance(y_hat, tuple):
                    y_hat = y_hat[0]
            log_preds = torch.nn.functional.log_softmax(y_hat[0], dim=1)
            
            idxs = torch.tensor(tokens[1:], dtype=torch.long).to(device).view(-1, 1)
            action_values = log_preds.gather(dim=1, index=idxs).view(-1, 1)
            
            expected_reward = -torch.sum(action_values * discounted_returns.view(-1, 1))
            batch_reward = batch_reward + reward
            loss = loss + expected_reward

        loss /= batch_size
        batch_reward /= batch_size
        # print(f'Epoch: {epoch + 1} Loss: {loss.item()}, Reward: {batch_reward}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if do_eval and (epoch + 1) % eval_steps == 0:
            if use_scaffold:
                scaffold = random.choice(scaffolds)
                generated_smiles = generate_smiles_scaffolds(model=model,
                                            tokenizer=tokenizer,
                                            scaffolds=[scaffold],
                                            temprature=kwargs['temprature'],
                                            size=kwargs['size'],
                                            batch_size=100,
                                            max_len=max_len,
                                            device=device)
    
            else:
                generated_smiles = generate_smiles(model=model,
                                          tokenizer=tokenizer,
                                          temprature=kwargs['temprature'],
                                          size=kwargs['size'],
                                          max_len=max_len,
                                          device=device)
                                          

            #reward_fn.eval = True

            get_stats(train_set=kwargs['train_set'],
                    generated_smiles=generated_smiles,
                    save_path=f"{kwargs['save_path']}",
                    folder_name=f'mid_RL/step_{epoch +1}',
                    reward_fn=reward_fn,
                    scaffold=scaffold if use_scaffold else None)

            model.train()
