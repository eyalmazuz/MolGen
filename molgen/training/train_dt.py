import os
import gc
import re
import math
import numpy as np
from tqdm import tqdm
from typing import Any

import wandb
import torch
import selfies as sf

from molgen.models.dt_gpt import sample
from molgen.utils.plot_utils import save_plot


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, reward_func, config, device: str = "cuda", wandb_log=False):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.reward_func = reward_func
        self.config = config
        self.bos_token_id = train_dataset.dataset.tokenizer.bos_token_id
        self.eos_token_id = train_dataset.dataset.tokenizer.eos_token_id
        self.pad_token_id = train_dataset.dataset.tokenizer.pad_token_id
        self.ignore_token_id = train_dataset.collate_fn.ignore_index
        self.wandb_log = wandb_log
        self.device = device
        self.optimizer = None

        # take over whatever gpus are on the system
        # if torch.cuda.is_available():
        #     self.device = torch.cuda.current_device()
        #     self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch):
        ckpt_name = f"epoch_{epoch}.pth" if self.test_dataset is None else f"best.pth"
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'token_counter': self.tokens
        }
        torch.save(checkpoint, os.path.join(self.config.get("ckpt_path", "."), ckpt_name))

    def load_checkpoint(self, ckpt_name="latest"):
        checkpoint_dir = self.config.get("ckpt_path", ".")
        if ckpt_name == "latest":
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")]
            if len(checkpoint_files) == 0:
                print(f"No checkpoints to load, starting training from scratch")
                epoch, tokens = 0, 0
                return epoch, tokens

            epoch_numbers = [
                int(re.search(r"epoch_(\d+)", file).group(1))
                for file in checkpoint_files
                if re.search(r"epoch_(\d+)", file)
            ]
            ckpt_name =  f"epoch_{max(epoch_numbers)}.pth"

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
        epoch_n, token_n = self.load_checkpoint()

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataset if is_train else self.test_dataset

            total_loss = 0
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                if "cuda" in self.device:
                    batch = {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in batch.items()}
                else:
                    batch = {k: v.to(self.device) for k, v in batch.items()}    # place data on the correct device
                x = batch["input_ids"]  # states
                y = batch["labels"]     # actions
                r = batch["rtgs"]       # rtgs (reward-to-go)
                a = batch["attention_mask"]
                g = batch["goal"]

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(input_ids=x, labels=y, targets=y, rtgs=r, attention_mask=a, goal=g)
                    # logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    total_loss += loss

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()     # TODO: scaler.scale(loss).backward() - only for mix precision training
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                    self.optimizer.step()

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
                    pbar.set_description(f"epoch {epoch_num + 1} of {epochs} | iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

                    del batch
                    torch.cuda.empty_cache()
                    gc.collect()

            # if not is_train:
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

            # supports early stopping based on the test loss, or save every X epochs if no test set
            good_model = (self.test_dataset is None and (epoch % 5 == 0)) or test_loss < best_loss
            if self.config.get("ckpt_path") is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(epoch)

            # -- pass in target returns
            # model_type = self.model.module.model_type if hasattr(self.model, "module") else self.model.model_type
            # if model_type == 'naive':
            #     eval_return = self.get_returns(0)
            # elif model_type == 'reward_conditioned':
            #     # TODO: return should be based on the reward function, for now put 1 for a scaled reward
            #     eval_return = self.get_returns(1)

        if self.wandb_run is None:
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
        train_config,
        test_dataloader=None,
        device: str = "cuda",
        wandb_log=False,
):
    trainer = Trainer(model, train_dataloader, test_dataloader, reward_func, train_config, device, wandb_log)
    trainer.train(optimizer)
