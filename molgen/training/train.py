import os
import gc
import time
from itertools import cycle
from typing import Any, Optional, Callable
from tqdm import tqdm
import torch

import wandb
from molgen.utils.utils import is_distributed_run, is_master_process
from molgen.utils.train_utils import get_checkpoint
from molgen.models.dt_gpt import get_returns


def pretrain_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    ctx,
    scaler,
    reward_func: Optional[Callable] = None,
    checkpoint_dir: str = "./model/",
    load_checkpoint: bool = False,
    max_steps: int = 1000000,
    grad_clip: float = 1.0,
    gradient_accumulation_steps: int = 1,
    eval_interval: int = -1,
    log_interval: int = -1,
    wandb_log: bool = True,
    device: str = "cuda",
    globals_config: dict[str, Any] | None = None,
) -> None:
    ema_loss: float | None = None  # Initialize EMA loss
    alpha = 0.1  # Smoothing factor for EMA; adjust as needed
    best_val_loss = torch.tensor(1e9)
    # Use cycle to create an infinite iterator
    iter_loader = cycle(train_dataloader)
    t0 = time.time()
    model.train()

    init_step = 0
    if load_checkpoint:
        checkpoint = get_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            raw_model = model.module if is_distributed_run() else model
            raw_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            model.config = checkpoint['model_args']
            init_step = checkpoint['step']
            best_val_loss = checkpoint['best_val_loss']
            globals_config = checkpoint['globals']
            print(f"Checkpoint loaded. Resuming training from step {init_step}")
        else:
            print(f"No checkpoint found at {checkpoint_dir}, starting training from scratch!")

    epoch = init_step // len(train_dataloader)
    for step in tqdm(range(init_step, max_steps)):
        for micro_step in range(gradient_accumulation_steps):
            # Handle distributed training if applicable
            if is_distributed_run():
                model.require_backward_grad_sync = micro_step == (gradient_accumulation_steps - 1)

            batch = next(iter_loader)
            # Move batch to device
            if "cuda" in device:
                batch = {k: v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
            else:
                batch = {k: v.to(device) for k, v in batch.items()}

            with ctx:
                logits, loss = model(**batch)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Update EMA of loss
        lossf = loss.item() * gradient_accumulation_steps
        ema_loss = lossf if ema_loss is None else alpha * lossf + (1 - alpha) * ema_loss
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        # if step % log_interval == 0 and is_master_process():
        if (step + 1) % len(train_dataloader) == 0 and is_master_process():
            epoch += 1
            # Print EMA loss
            print(f"epoch {epoch}, step {step}: EMA loss = {ema_loss:.4f} time {dt*1000:.2f}ms (per step)")

        # if step % eval_interval == 0 and is_master_process():
            model.eval()
            losses = torch.zeros(len(val_dataloader))
            for val_step, batch in enumerate(val_dataloader):
                if "cuda" in device:
                    batch = {k: v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
                else:
                    batch = {k: v.to(device) for k, v in batch.items()}

                with ctx:
                    logits, loss = model(**batch)
                losses[val_step] = loss.item()

            val_loss = losses.mean()
            model.train()
            print(f"val loss = {val_loss:.4f}")

            if wandb_log:
                wandb.log(  # type: ignore
                    {
                        "epoch": epoch,
                        "step": step,
                        "training_loss": ema_loss,
                        "validation_loss": val_loss,
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=step,
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if step > 0:
                    raw_model = model.module if is_distributed_run() else model
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "model_args": model.config,
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "globals": globals_config,
                    }
                    print(f"saving checkpoint to {checkpoint_dir}")
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(checkpoint_dir, "ckpt.pt"))

        # if reward_func is not None and model.model_type == 'reward_conditioned' and (epoch + 1) % eval_interval == 0:
        #     eval_return = get_returns(1, model, train_dataloader, reward_func, device)

        torch.cuda.empty_cache()
        gc.collect()
