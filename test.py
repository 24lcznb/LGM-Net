import os
import torch
import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from model import LGMNet
from PIL import ImageFile

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Directory structure (example with Lu dataset)
# test_folder/              # Testing Folder 
# └── Lu/                   # Dataset Name
#     ├── rgb/              # High-Resolution RGB Images
#     │   ├── 001.png
#     │   └── ...
#     ├── raw/              # Low-Resolution Depth Maps
#     │   ├── raw_4/        # Downsampling LR Depths (×4 scale)
#     │   │   ├── 001.png
#     │   │   └── ...
#     │   ├── raw_8/
#     │   │   ├── 001.png
#     │   │   └── ...
#     │   └── raw_16/
#     │       ├── 001.png
#     │       └── ...
#     └── result/           # GDSR Results of Our LGM-Net
#         ├── result_4/     # Reconstructed HR Depths (×4 scale)
#         │   └── 001.png   
#         ├── result_8/
#         │   └── 001.png 
#         └── result_16/
#             └── 001.png  


# Run inference on test datasets
def inference():
    # List of datasets to evaluate
    dataset_names = ['Lu', 'Middlebury', 'NYUv2', 'RGBDD']

    # Path to the testing root folder
    test_folder = '/path/to/your/test_folder'

    # Initialize model and load weights
    model_dir = Path(f'/path/to/your/model.pth')
    network = LGMNet().cuda()
    network.load_state_dict(torch.load(model_dir)['network_state_dict'])

    # Process each scale factor: ×4, ×8, ×16
    for n in [4, 8, 16]:
        for dataset_name in dataset_names:
            # Path to HR RGB images
            rgbs_dir = Path(f'{test_folder}/{dataset_name}/rgb')

            # Process each RGB-D pair
            for rgb_dir in rgbs_dir.rglob('*.png'):
                # Path to each HR RGB image and corresponding LR depth map
                rgb_dir = str(rgb_dir)
                raw_dir = rgb_dir.replace('rgb', f'raw/raw_{n}')
                img_name = os.path.basename(rgb_dir)

                # Load RGB image and LR depth
                rgb_pil = Image.open(rgb_dir)
                raw_pil = Image.open(raw_dir)
                
                # Preprocess RGB image
                rgb_array = np.array(rgb_pil, dtype=np.float32)
                h1, w1 = rgb_array.shape[0], rgb_array.shape[1]
                rgb_array = rgb_array[:(h1 - h1 % n) ,:(w1 - w1 % n)]
                rgb_tensor = torch.from_numpy((rgb_array / 255.).transpose((2, 0, 1))).unsqueeze(0)
                
                # Preprocess LR depth 
                raw_array = np.array(raw_pil, dtype=np.float32)
                max_value, min_value = np.max(raw_array), np.min(raw_array)
                raw_nor = (raw_array - min_value) / (max_value - min_value)
                raw_tensor = torch.from_numpy(raw_nor).unsqueeze(0).unsqueeze(0)

                # Upsample LR depth to match the same resolution of RGB image (bicubic)
                raw_tensor = torch.nn.functional.interpolate(raw_tensor, (rgb_tensor.shape[2], rgb_tensor.shape[3]), mode='bicubic')
                hole_tensor = torch.ones((1,1,rgb_tensor.shape[2],rgb_tensor.shape[3]))

                # Convert tensor type to float32 and move to GPU
                rgb_tensor = rgb_tensor.to(torch.float32)
                raw_tensor = raw_tensor.to(torch.float32)
                hole_tensor = hole_tensor.to(torch.float32)

                # Network forward pass
                enhance = network(rgb_tensor.cuda(), raw_tensor.cuda(), hole_tensor.cuda())
                enhance = enhance.squeeze().to('cpu').detach().numpy()

                # Inverse normalization (restore tensor result to original depth range)
                if raw_pil.mode in ["L", "P"]:
                    # 8-bit depth
                    enhance = np.clip(enhance * (max_value - min_value) + min_value, 0, 255).astype(np.uint8)
                else:
                    # 16-bit depth
                    enhance = np.clip(enhance * (max_value - min_value) + min_value, 0, 65535).astype(np.int32) 
                
                # Save predicted HR depth result
                enhance_pil = Image.fromarray(enhance)
                base_dir = rgb_dir.split('/rgb')[0]
                save_dir = os.path.join(base_dir, 'result', f'result_{n}')
                os.makedirs(save_dir, exist_ok=True)
                enhance_pil.save(os.path.join(save_dir, img_name))

if __name__ == "__main__":
    print("Testing started...")
    start_time = datetime.datetime.now()
    with torch.no_grad():
        inference()
    end_time = datetime.datetime.now()
    print("Testing finished")
    print(f"Time: {end_time - start_time}")

