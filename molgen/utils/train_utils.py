import os
from contextlib import nullcontext

import torch

from molgen.utils.utils import is_distributed_run


def setup_torch(seed: int=0, device: str="cuda") -> str:
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device == "cuda" and is_distributed_run():
        ddp_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

    return device


def setup_mixed_precision(device: str, dtype: str):
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("bfloat16 is not supported on this GPU type, reverting to float16")
        dtype = "float16"

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == "float16"))
    print(f"Running using {device=} {ptdtype=}")

    return ctx, scaler
