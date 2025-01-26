import os
import torch

from typing import Optional

from molgen.utils.utils import get_local_rank, get_rank, is_distributed_run


def setup_torch(seed: int = 0, device: str = "cuda") -> str:
    torch.manual_seed(seed + is_distributed_run() * get_rank())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device == "cuda" and is_distributed_run():
        ddp_local_rank = get_local_rank()
        device = f"cuda:{ddp_local_rank}"
        print(f"distributed_run & CUDA available using device: {device}")
        torch.cuda.set_device(device)
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"CUDA available using device: {device}")
        torch.cuda.set_device(device)
    else:
        print(f"CUDA not available")

    return device


def setup_mixed_precision(device: str, dtype: str):
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("bfloat16 is not supported on this GPU type, reverting to float16")
        dtype = "float16"

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == "float16"))
    print(f"Running using {device=} {ptdtype=}")

    return ctx, scaler


def get_checkpoint(checkpoint_dir: str = "./model/", ckpt_name: str = "ckpt.pt") -> Optional[dict]:
    path = os.path.join(checkpoint_dir, ckpt_name)
    if os.path.exists(path) and os.path.isfile(path):
        checkpoint = torch.load(path)
    else:
        return

    return checkpoint
