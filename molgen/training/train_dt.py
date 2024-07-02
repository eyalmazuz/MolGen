import math
import numpy as np
from tqdm import tqdm

import torch

from molgen.models.dt_gpt import sample


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, reward_func, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.reward_func = reward_func
        self.config = config
        self.bos_token_id = train_dataset.dataset.tokenizer.bos_token_id
        self.eos_token_id = train_dataset.dataset.tokenizer.eos_token_id

        # take over whatever gpus are on the system
        self.device = config["device"]
        # if torch.cuda.is_available():
            # self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.get("ckpt_path", "."))

    def train(self, optimizer):
        model, config = self.model, self.config

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataset if is_train else self.test_dataset

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:

                # place data on the correct device
                x = batch["input_ids"].to(self.device)  # states
                y = batch["labels"].to(self.device)     # actions
                r = batch["rtg"].to(self.device)        # rtgs (reward-to-go)
                a = batch["attention_mask"].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(states=x, actions=y, targets=y, rtgs=r, attention_mask=a)
                    # logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())   # TODO: consider removing .item if aggregating

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()     # TODO: scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.get("decay_lr", True):
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
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
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config["learning_rate"]

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                return test_loss

        # best_loss = float('inf')

        best_return = -float('inf')

        self.tokens = 0  # counter used for learning rate decay
        epochs = config["max_steps"] // len(self.train_dataset)
        for epoch in range(epochs):

            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            # -- pass in target returns
            if self.model.model_type == 'naive':
                eval_return = self.get_returns(0)
            elif self.model.model_type == 'reward_conditioned':
                # TODO: return should be based on the reward function, for now put 1 for a scaled reward
                eval_return = self.get_returns(1)

    def get_returns(self, ret):
        self.model.train(False)

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            # TODO: need BOS token initial state
            state = torch.tensor(self.bos_token_id, dtype=torch.int64)
            state = state.to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None,
                                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(
                                        -1),
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    # TODO: need to replace env with tokens, i.e. done is action == EOS token
                    state, reward_sum, done = (
                        torch.tensor(self.bos_token_id, dtype=torch.int64), 0, False
                    )
                action = sampled_action.cpu().numpy()[0, -1]
                actions += [sampled_action]
                # TODO: get next state by appending the action and use EOS token for done
                state.appened(action)
                reward = self.reward_func(state)
                done = action == self.eos_token_id
                # state, reward, done = env.step(action)
                reward_sum = reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward] # TODO: Check this
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(
                                            1).unsqueeze(0),
                                        rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(
                                            0).unsqueeze(-1),
                                        timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1),
                                                                                                 dtype=torch.int64).to(
                                            self.device)))
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
        test_dataloader=None
):
    trainer = Trainer(model, train_dataloader, test_dataloader, reward_func, train_config)
    trainer.train(optimizer)
