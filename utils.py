import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

# Convert string arguments to boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# Print the number of trainable parameters (M)
def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.2fM' % (total / 1e6))

# Functions for Distributed Data Parallel (DDP) training
class DDPutils(object):
    @staticmethod
    # Setup the DDP process group
    def setup(rank, world_size, port):
        # rank: index of the current process (GPU)
        # world_size: total number of processes (GPUs)
        # port: communication port for the master node

        # Environment setting
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)

        # Initializing with NCCL backend
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    @staticmethod
    # Spawn multiple processes for DDP
    def run_demo(demo_fn, world_size):
        # demo_fn: name of the main training function (e.g., DDP_main)
        # world_size: the number of GPUs
        mp.spawn(demo_fn,
                args=(world_size,),
                nprocs=world_size,
                join=True)

    @staticmethod
    # Clean up DDP after training
    def cleanup():
        dist.destroy_process_group()

