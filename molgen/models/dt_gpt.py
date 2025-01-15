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
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
import random
import numpy as np
import inspect
import selfies as sf

torch.set_float32_matmul_precision('high')
CUDA_LAUNCH_BLOCKING = 1


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
    dropout: float = 0.1
    model_type: str = "reward_conditioned"
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    ignore_index: int = -100
    n_goals: int = 1
    # max_timestep = 25


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(normalized_shape=config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(normalized_shape=config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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
        self.goal_emb = nn.Embedding(config.n_goals, config.n_embd, dtype=torch.float32)
        self.drop = nn.Dropout(config.dropout)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
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
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, attention=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    try:
        block_size = model.module.get_block_size()
    except AttributeError:
        block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size // 3 else x[:, -block_size // 3:]  # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size // 3 else actions[:,
                                                                         -block_size // 3:]  # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size // 3 else rtgs[:, -block_size // 3:]  # crop context if needed
        logits, _ = model(input_ids=x_cond, labels=actions, targets=None, rtgs=rtgs, attention_mask=attention)
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


def get_returns(ret, model, train_dataset, reward_func: Callable, device, k: int = 10, temperature: float = 1.0):
    model.train(False)
    bos_token_id = train_dataset.dataset.tokenizer.bos_token_id
    eos_token_id = train_dataset.dataset.tokenizer.eos_token_id
    pad_token_id = train_dataset.dataset.tokenizer.pad_token_id

    T_rewards, T_Qs = [], []
    done = True
    for i in range(k):
        terminated = False
        init_state = torch.tensor([bos_token_id], dtype=torch.int64)
        init_state = init_state.to(device).unsqueeze(0).unsqueeze(0)
        rtgs = [ret]
        # first state is from env, first rtg is target return, and first timestep is 0
        sampled_action = sample(
            model=model,
            x=init_state,
            steps=1,
            temperature=temperature,
            sample=True,
            actions=None,
            rtgs=torch.tensor(rtgs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(-1),
            # timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device)
        )

        j = 0
        all_states = init_state
        actions = []
        while True:
            if done:
                state, reward_sum, done = ([bos_token_id], 0, False)
            action = sampled_action.cpu().numpy()[0, -1]
            actions += [action]
            state.append(action)
            sequence = train_dataset.dataset.tokenizer.decode(state, skip_special_tokens=True)[0]
            if train_dataset.dataset.string_type == "SELFIES":
                sequence = sf.decoder(sequence)
            reward = reward_func(sequence)
            done = action == eos_token_id  # mol is complete when [EOS] token is generated
            reward_sum = reward
            j += 1

            # if molecule length exceeds block_size and [EOS] token wasn't generated terminate generation
            if len(state) >= model.block_size // 3 and not done:
                terminated = True

            if done or terminated:
                T_rewards.append(reward_sum)
                break

            tensor_state = torch.tensor(state, device=device).unsqueeze(0).unsqueeze(0)
            pad_size = tensor_state.shape[-1] - all_states.shape[-1]
            all_states = torch.nn.functional.pad(all_states, (0, pad_size), value=pad_token_id)
            all_states = torch.cat([all_states, tensor_state], dim=1)

            rtgs += [rtgs[-1] - reward]
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            sampled_action = sample(
                model=model,
                x=all_states,
                steps=1,
                temperature=temperature,
                sample=True,
                actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(0),
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(device).unsqueeze(0),
                attention=torch.tensor(np.tril(np.ones(all_states.shape[1:])), dtype=torch.long).to(device).unsqueeze(0)
                # timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
            )

    eval_return = sum(T_rewards) / 10.
    print("target return: %d, eval return: %d" % (ret, eval_return))
    model.train(True)
    return eval_return
