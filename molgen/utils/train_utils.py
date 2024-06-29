from contextlib import nullcontext
import os

import torch

from molgen.utils.utils import is_distributed_run


def setup_torch(seed: int=0, device: str="cuda") -> None:
    torch.manual_seed(seed)
    torch.backends.cuda.matmul_allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device == "cuda" and is_distributed_run():
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)


def setup_mixed_precision(device: str, dtype: str):
    if dtype == "bfloat16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        print("bfloat16 is not supported on this GPU type, reverting to float16")
        dtype = "float16"

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype=="float16"))

    return ctx, scaler
