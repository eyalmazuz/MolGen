import torch
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

    def forward(self, q, k, v, mask=None):
       
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        

        B, T, C = q.size()
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)


        y, att_weights = self.attention(q, k ,v, mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))

        return y, att_weights

    def attention(self, q, k, v, mask=None):
        att = (q @ k.transpose(-2, -1))
        att = att  * (1.0 / k.size(-1) ** 0.5)

        if mask is not None:
            # att = att + (mask * -1e9)
            att = att.masked_fill(mask == 1, float('-inf'))

        att_weights = F.softmax(att, dim=-1)
        att = self.attn_drop(att_weights)

        y = att @ v
        return y, att_weights

class DecoderOnlyBlock(nn.Module):
    def __init__(self, config) -> None:
        super(DecoderOnlyBlock, self).__init__()

        self.attn = MultiheadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.n_embd),
            nn.Dropout(config.resid_dropout_rate),
        )

    def forward(self, x, mask=None):
        x_norm = self.ln1(x)
        attn_logits, attn_weights = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_logits
        x = x + self.mlp(self.ln2(x))

        return x, attn_weights


class EncoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super(EncoderBlock, self).__init__()

        self.mha = MultiheadAttention(config)
        self.dropout = nn.Dropout(config.resid_dropout_rate)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.n_embd),
            nn.Dropout(config.resid_dropout_rate),
        )

    def forward(self, x, padding_mask=None):
        attn_logits, attn_weights = self.mha(x, x, x, mask=padding_mask)
        attn_logits = self.dropout(attn_logits)
        x = self.ln1(x + attn_logits)
        x = x + self.ln2(x + self.mlp(x))

        return x, attn_weights

class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super(Encoder, self).__init__()


        self.token_embds = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_dropout_rate)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])

    def forward(self, idx, padding_mask=None):
        B, T = idx.size()

        token_embds = self.token_embds(idx)
        pos_embs = self.pos_emb[:, :T, :] # each position maps to a (learnable) vector
        x = self.drop(token_embds + pos_embs)

        attn_weights = {}

        for i, block in enumerate(self.blocks):
            x, weights = block(x, padding_mask=padding_mask)
            attn_weights[f'encoder_block_{i}'] = weights

        return x, attn_weights

class DecoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiheadAttention(config)
        self.mha2 = MultiheadAttention(config)

        self.dropout1 = nn.Dropout(config.resid_dropout_rate)
        self.dropout2 = nn.Dropout(config.resid_dropout_rate)
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.n_embd),
            nn.Dropout(config.resid_dropout_rate),
        )

    def forward(self, x, enc_out, look_ahead_mask=None, padding_mask=None):
        dec_attn_logits, dec_attn_weights = self.mha1(x, x, x, mask=look_ahead_mask)
        dec_attn_logits = self.dropout1(dec_attn_logits)
        
        x = self.ln1(x + dec_attn_logits)
        
        enc_dec_attn_logits, enc_dec_attn_weights = self.mha2(x, enc_out, enc_out, mask=padding_mask)
        enc_dec_attn_logits = self.dropout2(enc_dec_attn_logits)
        
        x = self.ln2(x + enc_dec_attn_logits)
        
        x = x + self.ln3(x + self.mlp(x))

        return x, dec_attn_weights, enc_dec_attn_weights

class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super(Decoder, self).__init__()

        self.token_embds = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_dropout_rate)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])

    def forward(self, idx, enc_out, look_ahead_mask=None, dec_padding_mask=None):
        B, T = idx.size()
       
        token_embds = self.token_embds(idx)
        pos_embs = self.pos_emb[:, :T, :] # each position maps to a (learnable) vector
        x = self.drop(token_embds + pos_embs)

        dec_attn_weights = {}
        for i, block in enumerate(self.blocks):
            x, dec_weights, enc_dec_weights = block(x, enc_out,
                                                    look_ahead_mask=look_ahead_mask,
                                                    padding_mask=dec_padding_mask)

            dec_attn_weights[f'decoder_block_{i}'] = dec_weights
            dec_attn_weights[f'encoder_decoder_block_{i}'] = enc_dec_weights

        return x, dec_attn_weights
