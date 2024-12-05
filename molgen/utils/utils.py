import os


def is_distributed_run():
    return int(os.environ.get("RANK", -1)) != -1


def is_master_process():
    ddp_rank = int(os.environ.get("RANK", -1))
    return ddp_rank == -1 or ddp_rank == 0
