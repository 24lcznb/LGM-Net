import os
import torch
import argparse
from pathlib import Path
from engine import GDSRTrainer
from utils import str2bool, print_model_parm_nums, DDPutils
from model import LGMNet

# Enable cuDNN benchmark
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Specify visible CUDA device(s)
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

# Training configuration arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training configurations for LGM-Net",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rgbd_dir", action="store", type=lambda x: Path(x), default=r'/path/to/your/RGBD_datasets', 
                        help="Path to mixed RGB-D training datasets", required=False)
    parser.add_argument("--hole_dir", action="store", type=lambda x: Path(x), default=r'/path/to/your/Hole_datasets',
                        help="Path to hole/mask dataset", required=False)
    parser.add_argument("--save_dir", action="store", type=str, default=r'/path/to/your/save_dir',
                        help="Path to save model checkpoints", required=False)
    parser.add_argument("--epochs", action="store", type=int, default=300, nargs="+",
                        help="Number of training epochs", required=False)
    parser.add_argument("--batch_size", action="store", type=int, default=1, nargs="+",
                        help="Batch size per GPU", required=False)
    parser.add_argument("--resume_train", action="store", type=str2bool, default=False,
                        help="Resume from checkpoint", required=False)
    parser.add_argument("--model_dir", action="store", type=lambda x: Path(x), default=r'/path/to/your/checkpoint.pth', 
                        help="Path to pre-trained model or checkpoint for resuming", required=False)
    parser.add_argument("--port", action="store", type=int, default=6008,
                        help="Port for DDP communication")
    return parser.parse_args()

# DDP(Distributed Data Parallel) training worker
def DDP_main(rank, world_size):
    # Setup arguments and DDP
    args = parse_arguments()
    args.save_dir = Path(args.save_dir)
    DDPutils.setup(rank, world_size, args.port)
        
    # Initialize the network
    network = LGMNet()
    if rank == 0: print_model_parm_nums(network)
        
    # Load checkpoint if resuming training
    if args.resume_train:
        if rank == 0:
            print('Resuming training from checkpoint...')
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.model_dir, map_location=map_location)
    else:
        checkpoint = None

    # Initialize trainer (loss, optimizer, hyperparameters)
    semigan = GDSRTrainer(network, rank)
    semigan.train(
        args=args,
        rank=rank,
        learning_rate=0.0001,
        feedback_factor=300,
        checkpoint_factor=1,
        num_workers=4,
        checkpoint=checkpoint,
    )

    # Clean up DDP after training
    DDPutils.cleanup()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print('Start training...')
        n_gpus = 1  # Number of GPUs
        DDPutils.run_demo(DDP_main, n_gpus)
    else:
        print('Error: GPU is not available.')