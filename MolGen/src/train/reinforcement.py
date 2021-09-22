import numpy as np
from numpy.lib.arraysetops import isin
from rdkit import Chem
import torch
from tqdm import trange


def policy_gradients(model, tokenizer, reward_fn, batch_size=16, epochs=100, step_size=3e-5, discount_factor=0.99, max_len=100):
    model.train()
    model.to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), step_size)
    for epoch in trange(epochs):
        loss = 0
        batch_reward = 0
        for batch in trange(batch_size, leave=False):
            tokens = [tokenizer.bos_token_id]
            next_token = ''
            while next_token != tokenizer.eos_token_id and len(tokens) < max_len:
                x = torch.tensor([tokens])
                y_pred = model(x)

                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                # print(y_pred.size())
                last_word_logits = y_pred[0][-1]
                p = torch.nn.functional.softmax(last_word_logits, dim=0)
                next_token = np.random.choice(len(last_word_logits), p=p.detach().numpy())
                tokens.append(next_token)

            smiles = tokenizer.decode(tokens[1:-1])
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                reward = reward_fn(mol) 
            else:
                reward = 0

            discounted_returns = torch.pow(discount_factor, torch.arange(len(tokens[:-1]), 0, -1)) * reward
            # discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)
            
            y_hat = model(torch.tensor([tokens[:-1]], dtype=torch.long))
            if isinstance(y_hat, tuple):
                    y_hat = y_hat[0]
            log_preds = torch.nn.functional.log_softmax(y_hat[0], dim=1)

            action_values = log_preds.gather(dim=1, index=torch.tensor(tokens[1:], dtype=torch.long).view(-1, 1)).view(-1, 1)
            expected_reward = torch.sum(action_values * discounted_returns.view(-1, 1))
            batch_reward = batch_reward + reward
            loss = loss - expected_reward

        loss /= batch_size
        batch_reward /= batch_size
        # print(f'Epoch: {epoch + 1} Loss: {loss.item()}, Reward: {batch_reward}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
