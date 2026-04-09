import datetime
import time
import timeit
import argparse
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Tuple
from data_loader import DataGenerator, get_dataloader
from loss import TotalLoss
from model import LGMNet

# GDSRTrainer: Trainer of LGM-Net for Guided Depth Super-Resolution (GDSR)
class GDSRTrainer:
    def __init__(self, network: LGMNet, rank: torch.device) -> None:
        self.network = network.to(rank)

    def optimize_net(self, rgb: Tensor, gt: Tensor, point_map: Tuple, hole_tuple: Tuple, optimizer: Optimizer, scaler: GradScaler
                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        with autocast():
            loss_fun = TotalLoss()
            depth = self.network(rgb, point_map, hole_tuple[1])
            loss_abs, loss_std, loss_grd, loss_esc, loss = loss_fun(depth, rgb, gt, hole_tuple[1], hole_tuple[0])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss_abs.item(), loss_std.item(), loss_grd.item(), loss_esc.item(), loss.item(), depth

    def feedback_module(self, elapsed: str, i: int, iteration_num: int, loss_abs: Tensor, loss_std: Tensor, loss_grd: Tensor, loss_esc: Tensor, loss: Tensor, summary: SummaryWriter, global_step: int
                        ) -> None:
        print('Elapsed:[%s]|batch:[%d/%d]|L1:%.4f|L_std:%.4f|L_grd:%.4f|L_esc:%.4f|Total Loss:%.4f'
              % (elapsed, i, iteration_num, float(loss_abs), float(loss_std), float(loss_grd), float(loss_esc), float(loss)))

        summary.add_scalar("loss/loss_abs", loss_abs, global_step=global_step)
        summary.add_scalar("loss/loss_std", loss_std, global_step=global_step)
        summary.add_scalar("loss/loss_grad", loss_grd, global_step=global_step)
        summary.add_scalar("loss/loss_esc", loss_esc, global_step=global_step)
        summary.add_scalar("loss/loss", loss, global_step=global_step)

    def train(self, args: argparse, rank: torch.device, learning_rate: float = 0.0001, feedback_factor: int = 10, checkpoint_factor: int = 1, num_workers: int = 0, checkpoint: Dict = None):
        self.network.train()
        optimizer = torch.optim.Adam([{'params': self.network.parameters(), 'initial_lr': learning_rate,}], lr=learning_rate)
        scaler = GradScaler()
        start_epoch = 0

        if checkpoint is not None:
            start_epoch = checkpoint['epoch']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            checkpoint.clear()
            del checkpoint

        scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5, last_epoch=(start_epoch - 1))
        self.network = DDP(self.network, device_ids=[rank])

        if rank == 0:
            model_dir, log_dir = args.save_dir / "models", args.save_dir / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            summary = SummaryWriter(str(log_dir / "tensorboard"))
            global_time = time.time()

        rgbgph_data, hole_data, rgbgph_sampler, hole_sampler = get_dataloader(args.rgbd_dir, args.hole_dir, args.batch_size, rank, num_workers)
        iteration_num = len(rgbgph_data)
        global_step = start_epoch * iteration_num

        warm_up_cnt = 0.0
        warm_up_max_cnt = iteration_num + 1.0

        for epoch in range(start_epoch + 1, args.epochs + 1):
            rgbgph_sampler.set_epoch(epoch)
            hole_sampler.set_epoch(epoch + args.epochs)

            if rank == 0:
                start = timeit.default_timer()
                print(f"\nEpoch: [{epoch}/{args.epochs}]")

            for (i, all_data) in enumerate(zip(rgbgph_data, hole_data), start=1):
                if epoch == 1:
                    warm_up_cnt += 1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt

                train_data = DataGenerator(args.batch_size)
                rgb, gt, point_map, hole_tuple = train_data.generate_data(all_data)
                loss_abs, loss_std, loss_grd, loss_esc, loss, depth = self.optimize_net(rgb, gt, point_map, hole_tuple, optimizer, scaler)
                global_step += 1

                if rank == 0 and ((i % feedback_factor == 0) or (i == 1)):
                    with torch.no_grad():
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        self.feedback_module(elapsed, i, iteration_num, loss_abs, loss_std, loss_grd, loss_esc, loss, summary, global_step)

            scheduler.step()

            if rank == 0:
                stop = timeit.default_timer()
                print("Epoch training time: %.3f sec" % (stop - start))

                if (epoch == 1 or epoch % checkpoint_factor == 0 or epoch == args.epochs):
                    save_file = model_dir / f"epoch_{epoch}.pth"
                    torch.save({
                        'epoch': epoch,
                        'network_state_dict': self.network.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    }, save_file)

        if rank == 0:
            print("Training completed")