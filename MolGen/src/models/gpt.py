"""
Code based on andrej karpathy minGPT code with a little bit of modifications
https://github.com/karpathy/minGPT/
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()

        self.query = nn.Linear(config.n_embd, config.proj_size)
        self.key = nn.Linear(config.n_embd, config.proj_size)
        self.value = nn.Linear(config.n_embd, config.proj_size)
        self.proj = nn.Linear(config.proj_size, config.n_embd)

        self.attn_drop = nn.Dropout(config.attn_dropout_rate)
        self.proj_drop = nn.Dropout(config.proj_dropout_rate)

        self.num_heads = config.num_heads

        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, q, k, v, attn_mask=None):
       
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        

        B, T, C = q.size()
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        look_ahead_mask = self.mask[:, :, :T, :T]
        if attn_mask is not None:
            attention_mask = attn_mask.view(B, 1, 1, T)
            mask = torch.minimum(look_ahead_mask, attention_mask)
        else:
            mask = look_ahead_mask

        y, att_weights = self.attention(q, k ,v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))

        return y, att_weights

    def attention(self, q, k, v, attn_mask=None):
        att = (q @ k.transpose(-2, -1))
        att = att  * (1.0 / k.size()[-1] ** 0.5)

        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float('-inf'))

        att_weights = F.softmax(att, dim=-1)
        att = self.attn_drop(att_weights)

        y = att @ v
        return y, att_weights

class DecoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super(DecoderBlock, self).__init__()

        self.attn = MultiheadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.n_embd),
            nn.Dropout(config.resid_dropout_rate),
        )

    def forward(self, x, attention_mask=None):
        x_norm = self.ln1(x)
        attn_logits, attn_weights = self.attn(x_norm, x_norm, x_norm, attn_mask=attention_mask)
        x = x + attn_logits
        x = x + self.mlp(self.ln2(x))

        return x, attn_weights


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super(GPT, self).__init__()

        self.token_embds = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_dropout_rate)

        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.config = config

    def forward(self, idx, attention_mask=None, labels=None):
        B, T = idx.size()
       
        token_embds = self.token_embds(idx)
        pos_embs = self.pos_emb[:, :T, :] # each position maps to a (learnable) vector
        x = self.drop(token_embds + pos_embs)

        attn_weights = {}
        
        for i, block in enumerate(self.blocks):
            x, weights = block(x, attention_mask=attention_mask)
            attn_weights[f'block_{i}'] = weights

        x = self.ln(x)
        logits = self.logits(x)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            return loss, logits, attn_weights
        
        else:
            return logits, attn_weights
    
    def generate(self, initial_token, end_token, max_len: int=100, device: str='cpu'):
        tokens = [initial_token]
        next_token = ''
        while next_token != end_token and len(tokens) < max_len:
            x = torch.tensor([tokens]).to(device)
            y_pred = self.forward(x)

            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            # print(y_pred.size())
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0)
            if p.device.type != 'cpu':
                p = p.cpu()
            next_token = np.random.choice(len(last_word_logits), p=p.detach().numpy())
            tokens.append(next_token)

        return tokens

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
    block = DecoderBlock(config)
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