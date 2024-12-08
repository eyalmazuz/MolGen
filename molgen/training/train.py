import os
from itertools import cycle

import torch

import wandb
from molgen.utils.utils import is_distributed_run, is_master_process


def pretrain_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    ctx,
    scaler,
    checkpoint_dir: str = "./model/",
    max_steps: int = 1000000,
    grad_clip: float = 1.0,
    gard_acc_steps: int = 1,
    eval_every: int = -1,
    log_every: int = -1,
    wandb_log: bool = True,
    device: str = "cuda",
) -> None:
    ema_loss = 0.0  # Initialize EMA loss
    alpha = 0.1  # Smoothing factor for EMA; adjust as needed
    best_val_loss = torch.tensor(1e9)
    # Use cycle to create an infinite iterator
    iter_loader = cycle(train_dataloader)
    for step in range(max_steps):
        for micro_step in range(gard_acc_steps):
            # Handle distributed training if applicable
            if is_distributed_run():
                model.require_backward_grad_sync = micro_step == (gard_acc_steps - 1)

            batch = next(iter_loader)
            # Move batch to device
            if "cuda" in device:
                batch = {k: v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
            else:
                batch = {k: v.to(device) for k, v in batch.items()}

            with ctx:
                logits, loss = model(**batch)
                loss = loss / gard_acc_steps

            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % log_every == 0 and is_master_process():
            lossf = loss.item() * gard_acc_steps
            # Update EMA of loss
            ema_loss = lossf if ema_loss is None else alpha * lossf + (1 - alpha) * ema_loss

            # Print EMA loss
            print(f"Step {step}: EMA loss = {ema_loss:.4f}")

        if step % eval_every == 0 and is_master_process():
            model.eval()
            losses = torch.zeros(len(val_dataloader))
            for step, batch in enumerate(val_dataloader):
                if "cuda" in device:
                    batch = {k: v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
                else:
                    batch = {k: v.to(device) for k, v in batch.items()}

                with ctx:
                    logits, loss = model(**batch)
                    losses[step] = loss.item()

            val_loss = losses.mean()
            print(f"Step {step}: val loss = {val_loss:.4f}")
            model.train()
            if wandb_log:
                wandb.log(  # type: ignore
                    {
                        "step": step,
                        "train/loss": ema_loss,
                        "val/loss": val_loss,
                        "lr": optimizer.lr,
                    }
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
                    }
                    print(f"saving checkpoint to {checkpoint_dir}")
                    torch.save(checkpoint, os.path.join(checkpoint_dir, "ckpt.pt"))
