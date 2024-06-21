import os

def is_distributed_run():
    return int(os.environ.get("RANK", -1)) != -1
