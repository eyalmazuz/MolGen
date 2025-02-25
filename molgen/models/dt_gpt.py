"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
import random
import numpy as np
import inspect

from molgen.models.layers import DecoderOnlyBlock
torch.set_float32_matmul_precision('high')


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


@dataclass(init=True)
class DTGPTConfig:
    vocab_size: int = 32768
    block_size: int = 90
    max_seq_len: int = block_size // 3
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    model_type: str = "reward_conditioned"
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    ignore_index: int = -100
    n_goals: int = 1
    # max_timestep = 25


# class CausalSelfAttention(nn.Module):
#     """
#     A vanilla multi-head masked self-attention layer with a projection at the end.
#     It is possible to use torch.nn.MultiheadAttention here but I am including an
#     explicit implementation here to show that there is nothing too scary here.
#     """
#
#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
#         # key, query, value projections for all heads
#         self.key = nn.Linear(config.n_embd, config.n_embd)
#         self.query = nn.Linear(config.n_embd, config.n_embd)
#         self.value = nn.Linear(config.n_embd, config.n_embd)
#         # regularization
#         self.attn_drop = nn.Dropout(config.attn_pdrop)
#         self.resid_drop = nn.Dropout(config.resid_pdrop)
#         # output projection
#         self.proj = nn.Linear(config.n_embd, config.n_embd)
#         # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
#         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
#         if not self.flash:
#             print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
#         # causal mask to ensure that attention is only applied to the left in the input sequence
#         # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
#         #                              .view(1, 1, config.block_size, config.block_size))
#         self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
#                              .view(1, 1, config.block_size + 1, config.block_size + 1))
#         self.n_head = config.n_head
#         self.n_embd = config.n_embd
#
#     def forward(self, x, layer_past=None):
#         B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
#
#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#
#         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#         if self.flash:
#             # efficient attention using Flash Attention CUDA kernels
#             y = torch.nn.functional.scaled_dot_product_attention(
#                 q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
#             )
#         else:
#             # manual implementation of attention
#             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#             att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
#             att = F.softmax(att, dim=-1)
#             att = self.attn_dropout(att)
#             y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
#
#         # output projection
#         y = self.resid_drop(self.proj(y))
#         return y
#
#
# class Block(nn.Module):
#     """ an unassuming Transformer block """
#
#     def __init__(self, config):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(config.n_embd)
#         self.ln2 = nn.LayerNorm(config.n_embd)
#         self.attn = CausalSelfAttention(config)
#         self.mlp = nn.Sequential(
#             nn.Linear(config.n_embd, 4 * config.n_embd),
#             GELU(),
#             nn.Linear(4 * config.n_embd, config.n_embd),
#             nn.Dropout(config.resid_pdrop),
#         )
#
#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x


class DtGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_seq_len is not None
        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, dtype=torch.float32)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd, dtype=torch.float32)
        # self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        if config.n_goals > 0:
            self.goal_emb = nn.Embedding(config.n_goals, config.n_embd, dtype=torch.float32)
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[DecoderOnlyBlock(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd, dtype=torch.float32)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=torch.float32)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.state_embedding = self.tok_emb
        # self.state_encoder = nn.Linear(config.block_size // 3 * config.n_embd, config.n_embd)
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd, dtype=torch.float32), nn.Tanh())

        self.action_embeddings = self.tok_emb  # Actions are simply SMILES tokens to add to the state
        nn.init.normal_(self.action_embeddings.weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')
        # no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size())
        return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # state, action, and return
    def forward(self, input_ids, labels, targets=None, rtgs=None, attention_mask=None, goal=None):
        # input_ids: (batch, block_size, state_size)
        # labels: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # goals: optional - (batch, block_size, 1)

        batch_size = input_ids.shape[0]
        block_size = input_ids.shape[1]
        assert block_size <= self.block_size, \
            f"Cannot forward sequence of length {block_size}, block size is only {self.block_size}"
        state_embeddings = self.state_embedding(input_ids)  # (batch_size, block_size, state_size, n_embd)
        # TODO: replace mean_pooling with a mini-transformer model
        if attention_mask is not None:
            state_embeddings = self.mean_pooling(state_embeddings, attention_mask)  # (batch_size, block_size, n_embd)
        else:
            state_embeddings = state_embeddings.squeeze(-2)  # (1, 1, n_embd)

        if labels is not None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.ret_emb(rtgs.unsqueeze(-1))  # (batch, block_size, n_embd)
            # Modify RTG embedding
            # gs with goal embeddings (add or concat)
            if goal is not None:
                goal_embeddings = self.goal_emb(goal)  # (batch, n_embd)
                rtg_embeddings = rtg_embeddings + goal_embeddings
            action_embeddings = self.action_embeddings(labels)  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (batch_size, block_size * 3 - int(targets is None), self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -input_ids.shape[1] + int(targets is None):, :]
        elif labels is None and self.model_type == 'reward_conditioned':  # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            # Modify RTG embeddings with goal embeddings (add or concat)
            if goal is not None:
                goal_embeddings = self.goal_emb(goal)  # (batch, n_embd)
                rtg_embeddings = rtg_embeddings + goal_embeddings
            token_embeddings = torch.zeros((batch_size, input_ids.shape[1] * 2, self.config.n_embd),
                                           dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif labels is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(
                labels.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (batch_size, input_ids.shape[1] * 2 - int(targets is None), self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -input_ids.shape[1] + int(targets is None):, :]
        elif labels is None and self.model_type == 'naive':  # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        n_blocks = 2 if labels is None else 3  # only happens at very first timestep of evaluation
        pos = torch.arange(
            0, block_size, dtype=torch.long, device=input_ids.device
        ).repeat_interleave(n_blocks).unsqueeze(0)
        pos_emb = self.pos_emb(pos)

        x = self.drop(token_embeddings + pos_emb[:, :token_embeddings.shape[1], :])
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if labels is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
        elif labels is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif labels is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif labels is None and self.model_type == 'naive':
            logits = logits  # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=self.config.ignore_index
            )

        return logits, loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(
        model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, attention=None, goal=None
):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    max_seq_len = model.config.max_seq_len
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= max_seq_len else x[:, -max_seq_len:]  # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= max_seq_len else actions[:, -max_seq_len:]  # crop context if needed

        rtgs = rtgs if rtgs.size(1) <= max_seq_len else rtgs[:, -max_seq_len:]  # crop context if needed
        logits, _ = model(
            input_ids=x_cond, labels=actions, targets=None, rtgs=rtgs, attention_mask=attention, goal=goal
        )
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x
