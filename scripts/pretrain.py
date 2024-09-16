import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from molgen.utils.args.pretrain_args import get_pretrain_args
from molgen.utils.utils import is_distributed_run
from molgen.train_single_gpu import single_gpu_training
from molgen.train_distributed import multi_gpu_training


if __name__ == "__main__":
    args = get_pretrain_args()
    if True:#is_distributed_run():
        single_gpu_training(args)
    else:
        multi_gpu_training(args)
