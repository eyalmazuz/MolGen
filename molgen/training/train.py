import os
import time
from itertools import cycle
from typing import Any

import torch

import wandb
from molgen.utils.utils import is_distributed_run, is_master_process


def pretrain_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    ctx,
    scaler,
    checkpoint_dir: str = "./model/",
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
    for step in range(max_steps):
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
        if step % log_interval == 0 and is_master_process():
            # Print EMA loss
            print(f"step {step}: EMA loss = {ema_loss:.4f} time {dt*1000:.2f}ms")

        if step % eval_interval == 0 and is_master_process():
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
                        "step": step,
                        "train/loss": ema_loss,
                        "val/loss": val_loss,
                        # "lr": optimizer.lr,
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
                        "model_args": model.config,
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "globals": globals_config,
                    }
                    print(f"saving checkpoint to {checkpoint_dir}")
                    torch.save(checkpoint, os.path.join(checkpoint_dir, "ckpt.pt"))
