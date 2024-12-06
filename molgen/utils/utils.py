import os


def get_rank() -> int:
    return int(os.environ.get("RANK", -1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed_run():
    return get_rank() != -1


def is_master_process():
    ddp_rank = get_rank()
    return ddp_rank == -1 or ddp_rank == 0
