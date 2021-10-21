import numpy as np
from numpy.lib.arraysetops import isin
from rdkit import Chem
import torch
from tqdm import trange


def policy_gradients(model,
                     tokenizer,
                     reward_fn,
                     optimizer=torch.optim.Adam,
                     batch_size=16,
                     epochs=100,
                     step_size=3e-5,
                     discount_factor=0.99,
                     max_len=100,
                     device='cpu',
                     **kwargs):
    model.train()
    model.to(device)
    optimizer = optimizer(model.parameters(), step_size)
    for epoch in trange(epochs):
        loss = 0
        batch_reward = 0
        for batch in trange(batch_size, leave=False):
            tokens = model.generate(tokenizer.bos_token_id, tokenizer.eos_token_id, 1, max_len, device)

            smiles = tokenizer.decode(tokens[1:-1])
            
            reward = reward_fn(smiles, kwargs['fn'])

            discounted_returns = (torch.pow(discount_factor, torch.arange(len(tokens[:-1]), 0, -1)) * reward).to(device)
            # discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)
            
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
