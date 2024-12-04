import torch

from molgen.utils.utils import is_distributed_run


def pretrain_model(model, dataloader, optimizer, ctx, scaler, training_args) -> None:
    epochs = training_args["max_steps"] // len(dataloader)
    for step in range(epochs):
        acc_steps = 0
        for batch in dataloader:
            acc_steps += 1
            if training_args["device"] == "cuda":
                batch = {k: v.pin_memory().to(training_args["device"], non_blocking=True) for k, v in batch.items()}
            else:
                batch = {k: v.to(training_args["device"]) for k, v in batch.items()}

            if is_distributed_run():
                model.require_backward_grad_sync = (acc_steps == training_args["gradient_accumulation_steps"] - 1)

            with ctx:
                logits, loss = model(batch["input_ids"], batch["labels"])

            scaler.scale(loss).backward()

            if acc_steps % training_args["gradient_accumulation_steps"] == 0:
                if training_args["grad_clip"] != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args["grad_clip"])

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                acc_steps = 0
