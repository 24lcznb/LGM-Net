import os
import math
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as trans
from torchvision.transforms import InterpolationMode

# Cropped patch size
target_size = 320

# Load an RGB image from file and return as normalized tensor (C, H, W)
def rgb_read(filename: Path) -> Tensor:
    data = Image.open(filename)
    rgb = (np.array(data) / 255.).astype(np.float32)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
    data.close()
    return rgb

# Load a depth image, apply min-max normalization, return as tensor (1, H, W)
def depth_read(filename: Path) -> Tensor:
    data = Image.open(filename)
    depth = np.array(data, dtype=np.float32)
    max_value = np.max(depth)
    min_value = np.min(depth)
    depth_normalized = (depth - min_value) / (max_value - min_value)
    depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0)
    data.close()
    return depth_tensor

# Load a hole mask image, normalize by 255, return as tensor (1, H, W)
def hole_read(filename: Path) -> Image:
    data = Image.open(filename)
    hole = (np.array(data) / 255.).astype(np.float32)
    hole = torch.from_numpy(hole).unsqueeze(0)
    data.close()
    return hole

# Apply random crop and horizontal flip to RGB and depth, then color jitter on RGB
def rgbd_transform(rgb: Tensor, gt: Tensor,) -> Tuple[Tensor, Tensor, Tensor]:
    together_transform = trans.Compose(                
        [
            trans.RandomCrop((target_size, target_size)),
            trans.RandomHorizontalFlip(0.5),
        ]
    )
    rgb_transform = trans.ColorJitter(0.2, 0.2, 0.2)  
    rgbgt = torch.cat((rgb, gt), dim=0)     
    rgbgt = together_transform(rgbgt)               
    rgb = rgbgt[:3, :, :]                         
    gt = rgbgt[3, :, :].unsqueeze(0)
    rgb = rgb_transform(rgb)                  
    hole_gt = torch.ones_like(gt)              
    hole_gt[gt == 0] = 0.
    return rgb, gt, hole_gt         

# Apply heavy augmentation (crop, affine, flips) to hole mask, then binarize
def hole_transform(hole: Tensor) -> Tensor:     
    transform = trans.Compose(                  
        [
            trans.RandomCrop((target_size, target_size)),  
            trans.RandomAffine(degrees=180, translate=(0.5, 0.5), scale=(0.5, 4.0), shear=60, fill=1.), 
            trans.RandomHorizontalFlip(0.5),  
            trans.RandomVerticalFlip(0.5),   
        ]
    )
    hole = transform(hole) 
    hole[hole > 0.] = 1.    
    return hole              

# Dataset for RGB-D pairs with augmentation and dynamic hole generation
class RGBDDataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
    ) -> None:
        super(RGBDDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = rgbd_transform
        if os.path.exists('rgb_ls.pkl'):        
            rgb_list = open('/path/to/your/data_list/rgb_ls.pkl', 'rb')    
            self.rgb_ls = pickle.load(rgb_list)         
            rgb_list.close()
            depth_list = open('/path/to/your/data_list/depth_ls.pkl', 'rb')
            self.depth_ls = pickle.load(depth_list)        
            depth_list.close()
        else:
            self.rgb_ls, self.depth_ls = self.__getrgbd__(self.data_dir)   
            rgb_list = open('/path/to/your/data_list/rgb_ls.pkl', 'wb')  
            pickle.dump(self.rgb_ls, rgb_list) 
            rgb_list.close()
            depth_list = open('/path/to/your/data_list/depth_ls.pkl', 'wb')
            pickle.dump(self.depth_ls, depth_list)  
            depth_list.close()

    @staticmethod
    def __sample_metric__(data_shape, zero_rate) -> Tensor:     
        if zero_rate == 0.0: 
            random_point = torch.ones(data_shape)
        elif zero_rate == 1.0:
            random_point = torch.zeros(data_shape)
        else:  
            random_point = torch.ones(data_shape).uniform_(0.0, 1.0)   
                                                                        
            random_point[random_point <= zero_rate] = 0.   
            random_point[random_point > zero_rate] = 1.  
        return random_point    

    # Training Data Augmentation Pipeline
    def __randomlyadddistortion__(      
            self,
            distorted_gt: Tensor,
            hole_gt: Tensor,
            p_noise: float = 0.5 ,
            p_blur: float = 0.5
    ) -> Tensor:
        distorted_depth_shape = distorted_gt.shape  

        # add noise
        if np.random.uniform(0.0, 1.0) < p_noise:   
            gaussian_noise = torch.ones(distorted_depth_shape).normal_(0, np.random.uniform(0.01, 0.1))
            random_point = self.__sample_metric__(distorted_depth_shape, np.random.uniform(0.0, 1.0)) * hole_gt 
            distorted_gt = distorted_gt + gaussian_noise * random_point 

        # add blur
        if np.random.uniform(0.0, 1.0) < p_blur:    
            sample_factor = 2 ** (np.random.randint(2, 5))  
            # sample_factor = 2 ** 2
            depth_trans = trans.Compose([ 
                trans.Resize(
                    (int(distorted_depth_shape[1] * 1.0 / sample_factor),
                     int(distorted_depth_shape[2] * 1.0 / sample_factor)),
                    interpolation=InterpolationMode.BICUBIC
                ),
                trans.Resize((distorted_depth_shape[1], distorted_depth_shape[2]),
                            interpolation=InterpolationMode.BICUBIC),
            ])
            distorted_gt = depth_trans(distorted_gt)
        distorted_gt = torch.clamp(distorted_gt, 0.0, 1.0) 
        return distorted_gt 

    @staticmethod
    def __getrgbd__(data_path: Path) -> Tuple[List[Path], List[Path]]:      
        rgb_ls = []
        depth_ls = []

        for file in data_path.rglob('*.png'):       
            str_file = str(file)                   
            if '/rgb/' in str_file:                 
                rgb_ls.append(file)                 
                depth_file = str_file.replace('/rgb/', '/depth/', 1)   
                depth_ls.append(Path(depth_file))
        return rgb_ls, depth_ls                   

    def __getpoint__(self, gt: Tensor, hole_gt: Tensor) -> Tensor:   
        distorted_gt = gt.clone()     
        zero_rate=0
        distorted_gt = self.__randomlyadddistortion__(distorted_gt, hole_gt, p_noise=0.0, p_blur=1.0)
        point_map = self.__sample_metric__(gt.shape, zero_rate) * distorted_gt

        return point_map  

    def __len__(self) -> int:
        assert (len(self.rgb_ls) == len(self.depth_ls)), f"The number of RGB and gen_depth is unpaired"
        return len(self.rgb_ls)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        rgb_path = self.rgb_ls[item]
        depth_path = self.depth_ls[item]
        rgb = rgb_read(rgb_path)
        gt = depth_read(depth_path)
        rgb, gt, hole_gt = self.transform(rgb, gt)
        point_map = self.__getpoint__(gt, hole_gt)
        return rgb, gt, point_map, hole_gt  

# Dataset for hole masks (augmented and repeated)
class HoleDataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
    ) -> None:
        super(HoleDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = hole_transform
        if os.path.exists('hole_ls.pkl'):
            hole_data_list = open('/path/to/your/data_list/hole_ls.pkl', 'rb') 
            self.hole_ls = pickle.load(hole_data_list)
            hole_data_list.close()
        else:
            self.hole_ls = self.__gethole__(self.data_dir)
            hole_data_list = open('/path/to/your/data_list/hole_ls.pkl', 'wb')
            pickle.dump(self.hole_ls, hole_data_list)
            hole_data_list.close()

    @staticmethod
    def __gethole__(path: Path) -> List[Path]:
        hole = []
        for file in path.rglob('*.png'):
            hole.append(file)
        return hole

    def __len__(self) -> int:
        return len(self.hole_ls)

    def __getitem__(self, item: int) -> Tensor:
        hole = torch.ones((1, target_size, target_size))
        return hole

# Create two dataloaders (RGB-D and hole) with distributed sampling and automatic balancing
def get_dataloader(
        rgbd_dirs: Path,    
        hole_dirs: Path,   
        batch_size: int,  
        rank: torch.device,
        num_workers: int = 0, 
        factor: int = 1,  
) -> Tuple[
    DataLoader,
    DataLoader,
    DistributedSampler,
    DistributedSampler,
]:                          
    rgbd_dataset = RGBDDataset(rgbd_dirs)      
    hole_dataset = HoleDataset(hole_dirs)     
    if rank == 0:
        print(
            f"Loaded the RGBD dataset with: {len(rgbd_dataset)} images...\n"
            f"Loaded the Hole dataset with: {len(hole_dataset)} images...\n"
        )

    ratio_factor = math.ceil(factor * len(rgbd_dataset) / len(hole_dataset))  
    hole_dataset.hole_ls *= ratio_factor      

    rgbgph_sampler = DistributedSampler(rgbd_dataset)
    hole_sampler = DistributedSampler(hole_dataset)

    rgbgph_data = DataLoader(          
        rgbd_dataset, batch_size=batch_size, drop_last=True, sampler=rgbgph_sampler, num_workers=num_workers,
        pin_memory=True, persistent_workers=False)
    hole_data = DataLoader(             
        hole_dataset, batch_size=factor * batch_size, drop_last=True, sampler=hole_sampler, num_workers=num_workers,
        pin_memory=True, persistent_workers=False)

    return rgbgph_data, hole_data, rgbgph_sampler, hole_sampler


# Generate batched data for training
class DataGenerator(object):
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def generate_data(self, all_data):
        rgb = all_data[0][0].cuda(non_blocking=True)
        gt = all_data[0][1].cuda(non_blocking=True)
        point_map = all_data[0][2].cuda(non_blocking=True)
        hole_gt = all_data[0][3].cuda(non_blocking=True)
        hole_point = all_data[1].cuda(non_blocking=True)

        hole_point[point_map == 0] = 0
        point_map *= hole_point

        return rgb, gt, point_map, (hole_gt, hole_point)