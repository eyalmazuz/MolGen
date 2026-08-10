import os
import gc
import re
import math
import time
import numpy as np
from tqdm import tqdm
from typing import Any

import wandb
import torch
import selfies as sf
import random

from molgen.models.dt_gpt import sample
# from molgen.utils.famo import FAMO
from molgen.utils.plot_utils import save_plot


def group_batch_by_goal_count(x, y, attention_mask, rtgs, goals, goal_masks):
    """Yield dense sub-batches whose samples have the same active-goal count.

    Dynamic goals make the goal dimension ragged. Grouping samples by that
    dimension keeps every tensor dense without padding inactive goals and
    avoids running one model forward per sample.
    """
    if rtgs is None:
        yield x, y, attention_mask, None, goals, goal_masks
        return

    groups = {}
    for sample_idx, sample_rtgs in enumerate(rtgs):
        groups.setdefault(sample_rtgs.shape[0], []).append(sample_idx)

    for indices in groups.values():
        index = torch.tensor(indices, dtype=torch.long, device=x.device)
        group_attention = attention_mask.index_select(0, index) if attention_mask is not None else None
        group_rtgs = torch.stack([rtgs[i] for i in indices])
        group_goals = torch.stack([goals[i] for i in indices]) if goals is not None else None
        group_masks = torch.stack([goal_masks[i] for i in indices]) if goal_masks is not None else None
        yield (
            x.index_select(0, index),
            y.index_select(0, index),
            group_attention,
            group_rtgs,
            group_goals,
            group_masks,
        )


class Trainer:

    def __init__(
            self,
            model,
            train_dataset,
            test_dataset,
            reward_func,
            config,
            save_path=None,
            device: str = "cuda",
            wandb_log=False
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.reward_func = reward_func
        self.save_path = save_path
        self.config = config
        self.bos_token_id = train_dataset.dataset.tokenizer.bos_token_id
        self.eos_token_id = train_dataset.dataset.tokenizer.eos_token_id
        self.pad_token_id = train_dataset.dataset.tokenizer.pad_token_id
        self.ignore_token_id = train_dataset.collate_fn.ignore_index
        self.wandb_log = wandb_log
        self.device = device
        self.optimizer = None
        # Initialize FAMO
        # self.famo = FAMO(
        #     num_tasks=model.config.n_goals,
        #     min_losses=torch.full((model.config.n_goals,), 1e-8, device=device),
        #     lr=config.get("famo_beta", 0.025),
        #     gamma=config.get("famo_gamma", 0.001),
        # )

        # take over whatever gpus are on the system
        # if torch.cuda.is_available():
        #     self.device = torch.cuda.current_device()
        #     self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, ckpt_name: str = None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        if ckpt_name is None:
            ckpt_name = f"epoch_{epoch}.pth" if self.test_dataset is None else f"best.pth"

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'token_counter': self.tokens
        }
        torch.save(checkpoint, os.path.join(self.save_path, ckpt_name))

    def load_checkpoint(self, ckpt_name: str = None):
        checkpoint_dir = self.save_path
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint folder {checkpoint_dir} not found, starting training from scratch")
            os.makedirs(checkpoint_dir, exist_ok=True)
            return 0, 0

        if ckpt_name is None:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")]
            if len(checkpoint_files) == 0:
                print(f"No checkpoints to load, starting training from scratch")
                return 0, 0

            epoch_numbers = [
                int(re.search(r"epoch_(\d+)", file).group(1))
                for file in checkpoint_files
                if re.search(r"epoch_(\d+)", file)
            ]
            ckpt_name = f"epoch_{max(epoch_numbers)}.pth"

        else:
            ckpt_name = ckpt_name

        path = os.path.join(checkpoint_dir, ckpt_name)
        checkpoint = torch.load(path)

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tokens = checkpoint['token_counter']

        epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}")

        return epoch, tokens

    def train(self, optimizer):
        model, config = self.model, self.config
        self.optimizer = optimizer
        if self.config.get("load_checkpoint", False):
            epoch_n, token_n = self.load_checkpoint(ckpt_name="latest.pth")
        else:
            epoch_n, token_n = 0, 0

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataset if is_train else self.test_dataset

            total_loss = 0
            accumulation_steps = config.get("gradient_accumulation_steps", 1)
            if is_train:
                self.optimizer.zero_grad()

            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            # accumulation_steps = config.get("gradient_accumulation_steps", 1)
            
            lr = config["learning_rate"]
            optimizer_steps = 0
            log_every = config.get("log_every", 10)
            # if is_train:
            #     self.optimizer.zero_grad(set_to_none=True)
            for it, batch in pbar:
                def to_device(val):
                    if isinstance(val, torch.Tensor):
                        if "cuda" in self.device:
                            return val.pin_memory().to(self.device, non_blocking=True)
                        return val.to(self.device)
                    elif isinstance(val, list):
                        return [to_device(item) for item in val]
                    return val

                batch = {k: to_device(v) for k, v in batch.items()}    # place data on the correct device
                x = batch["input_ids"]  # states
                y = batch["labels"]     # actions
                # r = batch["rtgs"]       # rtgs (reward-to-go) is a list 
                # a = batch["attention_mask"]
                # g = batch["goal"]
                a = batch.get("attention_mask", None)
                r_list = batch.get("rtgs", None)       
                g_list = batch.get("goal", batch.get("goal_idx", None))
                gm_list = batch.get("goal_mask", None)

                batch_size = x.size(0)
                weighted_losses = []

                # During the first two epochs every sample uses one goal. Slice
                # before grouping so the whole batch can share a single pass.
                if epoch_num < 2 and r_list is not None and g_list is not None:
                    r_list = [sample[:1] for sample in r_list]
                    g_list = [sample[:1] for sample in g_list]
                    if gm_list is not None:
                        gm_list = [sample[:1] for sample in gm_list]

                measure_forward = is_train and it % log_every == 0
                if measure_forward and "cuda" in self.device:
                    torch.cuda.synchronize()
                forward_started_at = time.perf_counter()
                forward_calls = 0
                forward_group_sizes = []

                with torch.set_grad_enabled(is_train):
                    for x_group, y_group, a_group, r_group, g_group, gm_group in group_batch_by_goal_count(
                        x, y, a, r_list, g_list, gm_list
                    ):
                        forward_calls += 1
                        forward_group_sizes.append(x_group.size(0))
                        _, group_loss = model(
                            input_ids=x_group,
                            labels=y_group,
                            targets=y_group,
                            rtgs=r_group,
                            attention_mask=a_group,
                            goal=g_group,
                            goal_mask=gm_group,
                        )
                        weighted_losses.append(group_loss * x_group.size(0))

                if measure_forward:
                    if "cuda" in self.device:
                        torch.cuda.synchronize()
                    forward_seconds = time.perf_counter() - forward_started_at
                    pbar.write(
                        f"[timing] epoch={epoch_num + 1} iter={it} "
                        f"forward={forward_seconds:.4f}s calls={forward_calls} "
                        f"group_sizes={forward_group_sizes} samples={batch_size}"
                    )

                batch_loss = torch.stack(weighted_losses).sum() / batch_size
                total_loss += batch_loss.detach()

                if is_train:
                    display_loss = batch_loss.item()
                    scaled_loss = batch_loss / accumulation_steps

                    with torch.set_grad_enabled(is_train):
                        scaled_loss.backward()
                    if (it + 1) % accumulation_steps == 0 or (it + 1) == len(loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                        self.optimizer.step()
                        optimizer_steps += 1
                        self.optimizer.zero_grad()

                        # decay the learning rate based on our progress
                        if config.get("decay_lr", True):
                            self.tokens += (y != self.ignore_token_id).sum()  # number of tokens processed this step (i.e. label is not -100)
                            warmup_tokens = config.get("warmup_steps", 0)
                            if self.tokens < warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - warmup_tokens) / float(
                                    max(1, config.get("lr_decay_steps", warmup_tokens * 300) - warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config["learning_rate"] * lr_mult
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config["learning_rate"]

                        # report progress
                        pbar.set_description(f"epoch {epoch_num + 1} of {epochs} | iter {it}: train loss {display_loss:.5f}. lr {lr:e}")
                    torch.cuda.empty_cache()
                    

            # if not is_train:
            if is_train:
                print(f"Optimizer steps this epoch: {optimizer_steps}")
                print(f"Total batches this epoch: {len(loader)}")
                print(f"Accumulation steps: {accumulation_steps}")
                
            episode_loss = total_loss.item() / len(loader)
            print(f"\nMean Epoch Loss: {episode_loss:.4f}")
            if self.wandb_log:
                wandb.log(
                    {
                        'training_loss': episode_loss,
                        'epoch': epoch
                    }
                )

            return episode_loss

        best_loss = float('inf')
        best_return = -float('inf')

        self.tokens = token_n  # counter used for learning rate decay
        epochs = config["max_steps"] // len(self.train_dataset)
        epoch_losses = []
        test_loss = best_loss
        for epoch in range(epoch_n, epochs):

            epoch_loss = run_epoch('train', epoch_num=epoch)
            epoch_losses.append(epoch_loss)
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # Save every epoch after warmup when no validation set is available;
            # otherwise keep the best validation checkpoint.
            good_model = (epoch > 2 and self.test_dataset is None and epoch % 1 == 0) or test_loss < best_loss
            if self.save_path is not None and good_model:
                if self.test_dataset is not None:
                    best_loss = test_loss
                self.save_checkpoint(epoch)

            # self.save_checkpoint(epoch, ckpt_name="latest.pth")

            # -- pass in target returns
            # model_type = self.model.module.model_type if hasattr(self.model, "module") else self.model.model_type
            # if model_type == 'naive':
            #     eval_return = self.get_returns(0)
            # elif model_type == 'reward_conditioned':
            #     # TODO: return should be based on the reward function, for now put 1 for a scaled reward
            #     eval_return = self.get_returns(1)

        if not self.wandb_log:
            [print(f"{ep_loss:.5f}") for ep_loss in epoch_losses]  # Debug print
            save_plot({"Loss_per_Epoch": epoch_losses})

    def get_returns(self, ret, k: int = 10, temperature: float = 1.0):
        self.model.train(False)

        T_rewards, T_Qs = [], []
        done = True
        for i in range(k):
            terminated = False
            init_state = torch.tensor([self.bos_token_id], dtype=torch.int64)
            init_state = init_state.to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            goal_idx = np.random.choice(self.model.module.config.n_goals)
            goal = [goal_idx]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(
                model=self.model,
                x=init_state,
                steps=1,
                temperature=temperature,
                sample=True,
                actions=None,
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1),
                goal=torch.tensor(goal, dtype=torch.int64).to(self.device).unsqueeze(0),
                # timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device)
            )

            j = 0
            all_states = init_state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = ([self.bos_token_id], 0, False)
                action = sampled_action.cpu().numpy()[0, -1]
                actions += [action]
                state.append(action)
                sequence = self.train_dataset.dataset.tokenizer.decode(state, skip_special_tokens=True)[0]
                if self.train_dataset.dataset.string_type == "SELFIES":
                    sequence = sf.decoder(sequence)
                reward = self.reward_func[goal_idx](sequence)
                done = action == self.eos_token_id  # mol is complete when [EOS] token is generated
                reward_sum = reward
                j += 1

                # if molecule length exceeds block_size and [EOS] token wasn't generated terminate generation
                if len(state) >= self.model.config.max_seq_len and not done:
                    terminated = True

                if done or terminated:
                    T_rewards.append(reward_sum)
                    break

                tensor_state = torch.tensor(state, device=self.device).unsqueeze(0).unsqueeze(0)
                pad_size = tensor_state.shape[-1] - all_states.shape[-1]
                all_states = torch.nn.functional.pad(all_states, (0, pad_size), value=self.pad_token_id)
                all_states = torch.cat([all_states, tensor_state], dim=1)

                rtgs += [rtgs[-1] - reward]
                goal.append(goal_idx)
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(
                    model=self.model,
                    x=all_states,
                    steps=1,
                    temperature=temperature,
                    sample=True,
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(0),
                    rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0),
                    attention=torch.tensor(np.tril(np.ones(all_states.shape[1:])), dtype=torch.long).to(self.device).unsqueeze(0),
                    goal=torch.tensor(goal, dtype=torch.int64).to(self.device).unsqueeze(0),
                    # timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
                )
        eval_return = sum(T_rewards) / 10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return


def run_dt_training(
        model,
        train_dataloader,
        optimizer,
        ctx,
        scaler,
        reward_func,
        save_path,
        train_config,
        test_dataloader=None,
        device: str = "cuda",
        wandb_log=False,
):
    trainer = Trainer(model, train_dataloader, test_dataloader, reward_func, train_config, save_path, device, wandb_log)
    trainer.train(optimizer)
