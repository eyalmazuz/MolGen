from torch import nn
import torch
from torch.nn import functional as F

class GPTConfig():
    def __init__(self,
                vocab_size=512,
                n_positions=512,
                n_emb=512,
                proj_size=512,
                d_model=512,
                num_heads=8,
                n_layers=12,
                attn_dropout_rate=0.1,
                proj_dropout_rate=0.1,
                resid_dropout_rate=0.1,
                **kwargs
                ) -> None:
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_emb = n_emb
        self.proj_size = proj_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.resid_dropout_rate = resid_dropout_rate

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()

        self.query = nn.Linear(config.n_emb, config.proj_size)
        self.key = nn.Linear(config.n_emb, config.proj_size)
        self.value = nn.Linear(config.n_emb, config.proj_size)
        self.proj = nn.Linear(config.proj_size, config.n_emb)

        self.attn_drop = nn.Dropout(config.attn_dropout_rate)
        self.proj_drop = nn.Dropout(config.proj_dropout_rate)

        self.num_heads = config.num_heads

    def forward(self, x, attn_mask=None):
       
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        

        B, T, C = q.size()
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        y, att_weights = self.self_attention(q, k ,v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))

        return y, att_weights

    def self_attention(self, q, k, v, attn_mask=None):
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
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.ln2 = nn.LayerNorm(config.n_emb)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_emb, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.n_emb),
            nn.Dropout(config.resid_dropout_rate),
        )

    def forward(self, x, attention_mask=None):
        attn_logits, attn_weights = self.attn(self.ln1(x), attn_mask=attention_mask)
        x = x + attn_logits
        x = x + self.mlp(self.ln2(x))

        return x, attn_weights


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super(GPT, self).__init__()

        self.token_embs = nn.Embedding(config.vocab_size, config.n_emb)
        self.pos_embs = nn.Embedding(config.n_positions, config.n_emb)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.n_emb)
        self.logits = nn.Linear(config.n_emb, config.vocab_size)

        self.config = config

    def forward(self, idx, attention_mask=None, labels=None):
        B, T = idx.size()
        look_ahead_mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(idx.device)

        if attention_mask is not None:
            attention_mask = attention_mask.view(B, 1, 1, T)
            mask = torch.minimum(look_ahead_mask, attention_mask)
        
        else:
            mask = attention_mask

        token_embs = self.token_embs(idx)
        pos_embs = self.pos_embs(idx)
        x = token_embs + pos_embs
        attn_weights = []
        
        for i, block in enumerate(self.blocks):
            x, weights = block(x, attention_mask=mask)
            attn_weights.append(weights)

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

def main():
    config = GPTConfig(num_heads=8, proj_size=512, proj_dropout_rate=0, attn_dropout_rate=0, n_emb=512, n_layers=2)
    attn = MultiheadAttention(config)


    # k = torch.tensor([[10, 0, 0],
    #                   [0, 10, 0],
    #                   [0, 0, 10],
    #                   [0, 0, 10]]).float()

    # v = torch.tensor([[1, 0],
    #                       [10, 0],
    #                       [100, 5],
    #                       [100, 6],]).float()

    # q = torch.tensor([[0, 10, 0]]).float()
    # mask = torch.tril(torch.ones(4, 4)).view(1, 1, 4, 4)
    # print(mask)
    # y, w = attn(q, k , v, torch.tensor([[0, 1, 1, 1]]))

    x = torch.rand((64, 50, 512))

    # y, w = attn(x, x, x)
    # print(y.size(), w.size())



    torch.autograd.set_detect_anomaly(True)
    block = DecoderBlock(config)
    optimizer = torch.optim.SGD(block.parameters(), 3e-5)
    y, w = block(x)
    print(y.size(), w.size())

    y_t = torch.randint(0, 50, (64, 50))

    loss = F.cross_entropy(y.transpose(1, 2), y_t)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # gpt = GPT(config)

    # x = torch.randint(0, 512, (64, 26))

    # logits, att_weights = gpt(x)

    # print(logits.size(), att_weights[1].size())


if __name__ == "__main__":
    main()