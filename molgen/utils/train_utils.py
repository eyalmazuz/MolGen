from contextlib import nullcontext

import torch


def setup_torch(seed: int=0, device: str="cuda", dtype: str="bfloat16"):
    torch.manual_seed(seed)

    if device.startswith("cuda"):
        torch.backends.cuda.matmul_allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(device)

        if dtype == "bfloat16" and not torch.cuda.is_bf16_support():
            print("bfloat16 is not supported on this GPU type, reverting to float16")
            dtype = "float16"

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    return ctx, scaler
