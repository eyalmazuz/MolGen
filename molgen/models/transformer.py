"""
Code based on andrej karpathy minGPT code with a little bit of modifications
https://github.com/karpathy/minGPT/
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from molgen.models.layers import EncoderBlock, DecoderBlock


@dataclass(init=True)
class TransformerConfig():
    vocab_size: int = 32768
    block_size: int = 512
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super(Transformer, self).__init__()

        self.block_size = config.block_size
        
        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
 
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self, enc_idx, dec_idx, enc_mask=None, targets=None):
        enc_h = self.enc(enc_idx, enc_mask)
        logits = self.dec(dec_idx, enc_h, enc_mask)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


    def enc(self, idx, mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward encoder sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the Enocder model itself
        tok_emb = self.encoder.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.encoder.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.encoder.drop(tok_emb + pos_emb)
        for block in self.encoder.h:
            x = block(x, mask)
        x = self.encoder.ln_f(x)

        return x


    def dec(self, idx, enc_h, enc_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward decoder sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

       # forward the Decoder model itself
        tok_emb = self.decoder.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.decoder.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.decoder.drop(tok_emb + pos_emb)
        for block in self.decoder.h:
            x = block(x, enc_h, mask=enc_mask)

        x = self.decoder.ln_f(x)
        logits = self.lm_head(x)

        return logits


    @torch.no_grad()
    def generate(self, enc_idx, dec_idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        enc_h = self.enc(enc_idx, enc_mask)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.dec(idx_cond, enc_h, enc_mask)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
