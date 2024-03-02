"""
Code based on andrej karpathy minGPT code with a little bit of modifications
https://github.com/karpathy/minGPT/
"""

import numpy as np
import torch
from torch import nn
from torch.nn.modules import padding

from .layers import MultiheadAttention, DecoderOnlyBlock

class GPTConfig():
    def __init__(self,
                vocab_size=512,
                n_embd=512,
                block_size=512,
                proj_size=512,
                d_model=512,
                num_heads=8,
                n_layers=12,
                attn_dropout_rate=0.1,
                proj_dropout_rate=0.1,
                resid_dropout_rate=0.1,
                embd_dropout_rate=0.1,
                **kwargs
                ) -> None:
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.proj_size = proj_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.resid_dropout_rate = resid_dropout_rate
        self.embd_dropout_rate = embd_dropout_rate

class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super(GPT, self).__init__()

        self.token_embds = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_dropout_rate)

        self.blocks = nn.ModuleList([DecoderOnlyBlock(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.register_buffer('mask', 1 - torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.config = config

    def forward(self, input_ids, padding_mask=None, labels=None):
        
        output = {}
        B, T = input_ids.size()
       
        token_embds = self.token_embds(input_ids)
        pos_embs = self.pos_emb[:, :T, :] # each position maps to a (learnable) vector
        x = self.drop(token_embds + pos_embs)

        look_ahead_mask = self.mask[:, :, :T, :T]
        if padding_mask is not None:
            attention_mask = padding_mask.view(B, 1, 1, T)
            mask = torch.maximum(look_ahead_mask, attention_mask)
        else:
            mask = look_ahead_mask

        attn_weights = {}
        
        for i, block in enumerate(self.blocks):
            x, weights = block(x, mask=mask)
            attn_weights[f'block_{i}'] = weights

        x = self.ln(x)
        logits = self.logits(x)

        output['logits'] = logits
        output['attention weights'] = attn_weights

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            output['loss'] = loss
            return loss, logits, attn_weights
        
        else:
            return logits, attn_weights
        # return output

    def generate(self, initial_token, end_token, temprature: int=1, max_len: int=100, device=torch.device('cuda')):
        tokens = [initial_token]
        next_token = -1
        while next_token != end_token and len(tokens) < max_len:
            x = torch.tensor([tokens]).to(device)
            y_pred = self.forward(x)

            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits / temprature, dim=0)
            if p.device.type != 'cpu':
                p = p.cpu()
            next_token = np.random.choice(len(last_word_logits), p=p.detach().numpy())
            tokens.append(next_token)

        return tokens

    def __str__(self):
        return f"GPT_Layers_{self.config.n_layers}_Heads_{self.config.num_heads}_Emb_{self.config.n_embd}_Dmodel_{self.config.d_model}"

class GPTValue(nn.Module):
    def __init__(self, gpt):
      super(GPTValue, self).__init__()

      self.gpt = gpt
      self.value = nn.Linear(self.gpt.config.vocab_size, 1)

    def forward(self, input_ids, padding_mask=None, labels=None):
        logits, _ = self.gpt(input_ids, padding_mask, labels)
        state_values = self.value(logits)

        return state_values

def main():
    config = GPTConfig(num_heads=8, block_size=512, proj_dropout_rate=0, attn_dropout_rate=0, n_embd=512, n_layers=2)
    attn = MultiheadAttention(config)


    k = torch.tensor([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]]).float()

    v = torch.tensor([[1, 0],
                          [10, 0],
                          [100, 5],
                          [100, 6],]).float()

    q = torch.tensor([[0, 0, 10],
                      [0, 10, 0],
                      [10, 10, 0]]).float()
    # mask = torch.tril(torch.ones(4, 4)).view(1, 1, 4, 4)
    # print(mask)
    y, w = attn.attention(q, k , v)
    print(y)
    print(w)

    x = torch.rand((1, 60, 512))

    y, w = attn(x, x, x)
    print(y.size(), w.size())



    torch.autograd.set_detect_anomaly(True)
    block = DecoderOnlyBlock(config)
    # optimizer = torch.optim.SGD(block.parameters(), 3e-5)
    y, w = block(x)
    print(y.size(), w.size())

    # y_t = torch.randint(0, 50, (64, 50))

    # loss = F.cross_entropy(y.transpose(1, 2), y_t)
    # print(loss)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    gpt = GPT(config)

    x = torch.randint(0, 512, (64, 26))

    logits, att_weights = gpt(x)

    print(logits.size(), att_weights['block_1'].size())


if __name__ == "__main__":
    main()
