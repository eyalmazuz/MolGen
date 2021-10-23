from torch import nn
from torch.nn import functional as F

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

    def forward(self, q, k, v, attention_mask=None):
       
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        

        B, T, C = q.size()
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)


        y, att_weights = self.attention(q, k ,v, attention_mask=attention_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))

        return y, att_weights

    def attention(self, q, k, v, attention_mask=None):
        att = (q @ k.transpose(-2, -1))
        att = att  * (1.0 / k.size()[-1] ** 0.5)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))

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