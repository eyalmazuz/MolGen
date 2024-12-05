import os

import torch

from molgen.utils.utils import is_distributed_run


def pretrain_model(model, train_dataloader, val_dataloader, optimizer, ctx, scaler, training_args) -> None:
    epochs = training_args["max_steps"] // len(train_dataloader)
    ema_loss = 0.0  # Initialize EMA loss
    alpha = 0.1      # Smoothing factor for EMA; adjust as needed
    best_val_loss = torch.tensor(1e9)
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            if training_args["device"] == "cuda":
                batch = {k: v.pin_memory().to(training_args["device"], non_blocking=True) for k, v in batch.items()}
            else:
                batch = {k: v.to(training_args["device"]) for k, v in batch.items()}

            # Handle distributed training if applicable
            if is_distributed_run():
                model.require_backward_grad_sync = (step == training_args["gradient_accumulation_steps"])

            with ctx:
                logits, loss = model(batch["input_ids"], targets=batch["labels"])
                loss = loss / training_args["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % training_args["gradient_accumulation_steps"] == 0:
                if training_args["grad_clip"] != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args["grad_clip"])

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Update EMA of loss
                loss_item = loss.item()
                ema_loss = loss_item if ema_loss is None else alpha * loss_item + (1 - alpha) * ema_loss

                # Print EMA loss
                print(f"Epoch {epoch}, Step {step}: EMA Loss = {ema_loss:.4f}")

        print("Running validation loop")
        model.eval()
        losses = torch.zeros(len(val_dataloader))
        for step, batch in enumerate(val_dataloader):
            if training_args["device"] == "cuda":
                batch = {k: v.pin_memory().to(training_args["device"], non_blocking=True) for k, v in batch.items()}
            else:
                batch = {k: v.to(training_args["device"]) for k, v in batch.items()}

            with ctx:
                logits, loss = model(batch["input_ids"], targets=batch["labels"])
                losses[step] = loss.item()

            val_loss = losses.mean()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model.config,
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {training_args['checkpoint_dir']}")
                torch.save(checkpoint, os.path.join(training_args["checkpoint_dir"], 'ckpt.pt'))
