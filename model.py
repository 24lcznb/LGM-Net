import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Sequential, PixelShuffle
from modules import SAGSP, DCMA, MDAF, EADE

class LGMNet(Module):
    def __init__(self, 
                dim=12, 
                blk_nums = [1,2,3,3], 
                middle_blk_num = 3,
                refine_blk_num = 3):
        
        super(LGMNet, self).__init__()

        # Scale Factor
        self.padder_size = 2 ** len(blk_nums)

        # Overlapping Patch Embedding
        self.inrgb = Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.indp = Conv2d(in_channels=1, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=1, bias=True)

        # Encoder
        self.encoder1 = Sequential(*[SAGSP(dim) for _ in range(blk_nums[0])])
        self.encoder2 = Sequential(*[SAGSP(dim*2**1) for _ in range(blk_nums[1])])
        self.encoder3 = Sequential(*[SAGSP(dim*2**2) for _ in range(blk_nums[2])])
        self.encoder4 = Sequential(*[SAGSP(dim*2**3) for _ in range(blk_nums[3])])

        # Bidirectional Cross-Modal Interaction
        self.cross1 = DCMA(dim)
        self.cross2 = DCMA(dim*2**1)
        self.cross3 = DCMA(dim*2**2)
        self.cross4 = DCMA(dim*2**3)
        self.crossm = DCMA(dim*2**4)

        # Downsample
        self.down1to2 = Conv2d(dim, dim*2**1, 2, 2)
        self.down2to3 = Conv2d(dim*2**1, dim*2**2, 2, 2)
        self.down3to4 = Conv2d(dim*2**2, dim*2**3, 2, 2)        
        self.down4tom = Conv2d(dim*2**3, dim*2**4, 2, 2)

        # Middle Block
        self.middle = Sequential(*[SAGSP(dim*2**4) for _ in range(middle_blk_num)])

        # Upsample
        self.upmto4 = Sequential(Conv2d(dim*2**4, dim*2**5, 1, bias=False), PixelShuffle(2))
        self.up4to3 = Sequential(Conv2d(dim*2**3, dim*2**4, 1, bias=False), PixelShuffle(2))
        self.up3to2 = Sequential(Conv2d(dim*2**2, dim*2**3, 1, bias=False), PixelShuffle(2))
        self.up2to1 = Sequential(Conv2d(dim*2**1, dim*2**2, 1, bias=False), PixelShuffle(2))

        # Decoder
        self.decoder4 = Sequential(*[SAGSP(dim*2**3) for _ in range(blk_nums[3])])
        self.decoder3 = Sequential(*[SAGSP(dim*2**2) for _ in range(blk_nums[2])])
        self.decoder2 = Sequential(*[SAGSP(dim*2**1) for _ in range(blk_nums[1])])
        self.decoder1 = Sequential(*[SAGSP(dim) for _ in range(blk_nums[0])])
  
        # Comprehensive Feature Fusion
        self.fusion = MDAF(dim)

        # Multi-stage Feature Refinement
        self.refine = Sequential(*[SAGSP(dim*3) for _ in range(refine_blk_num)])

        # Generate Initial Output
        self.iniout = Conv2d(in_channels=3*dim, out_channels=1, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        # Precise Edge Enhancement
        self.edge = EADE()

    def forward(self, rgb, dp, hole):
        
        # Save the original RGB image (for EADE) and LR depth (for residual)
        rgb0 = rgb
        dp0 = dp
        
        # Spatial Padding: Ensure dimensions are divisible by the scale factor
        _, _, H, W = dp.shape
        pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        if pad_h != 0 or pad_w != 0:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h))
            dp = F.pad(dp, (0, pad_w, 0, pad_h))
        
        # Shallow Feature Extraction (Embedding)
        rgb = self.inrgb(rgb)
        dp = self.indp(dp)
        
        # ----- Encoder & Cross-Modal Interaction & Downsample-----
        # Encoder-L1: [C, H, W] → [2C, H/2, W/2]
        rgb = self.encoder1(rgb)
        dp = self.encoder1(dp)
        dp,rgb = self.cross1(dp,rgb)
        rgb1, dp1 = rgb, dp
        rgb = self.down1to2(rgb)
        dp = self.down1to2(dp)

        # Encoder-L2: [2C, H/2, W/2] → [4C, H/4, W/4]
        rgb = self.encoder2(rgb)
        dp = self.encoder2(dp)
        dp,rgb = self.cross2(dp,rgb)
        rgb2, dp2 = rgb, dp
        rgb = self.down2to3(rgb)
        dp = self.down2to3(dp)

        # Encoder-L3: [4C, H/4, W/4] → [8C, H/8, W/8]
        rgb = self.encoder3(rgb)
        dp = self.encoder3(dp)
        dp,rgb = self.cross3(dp,rgb)
        rgb3, dp3 = rgb, dp
        rgb = self.down3to4(rgb)
        dp = self.down3to4(dp)

        # Encoder-L4: [8C, H/8, W/8] → [16C, H/16, W/16]
        rgb = self.encoder4(rgb)
        dp = self.encoder4(dp)
        dp,rgb = self.cross4(dp,rgb)
        rgb4, dp4 = rgb, dp
        rgb = self.down4tom(rgb)
        dp = self.down4tom(dp)

        # -----Middle Block Transition: Encoder → Decoder-----
        rgb = self.middle(rgb)
        dp = self.middle(dp)
        dp,rgb = self.crossm(dp,rgb)

        # -----Decoder & Upsamle & Skip Connections & Cross-Modal Interaction-----
        # Decoder-L4: [16C, H/16, W/16] → [8C, H/8, W/8]
        rgb = self.upmto4(rgb) + rgb4
        dp = self.upmto4(dp) + dp4
        rgb = self.decoder4(rgb)
        dp = self.decoder4(dp)
        dp,rgb = self.cross4(dp,rgb)

        # Decoder-L3: [8C, H/8, W/8] → [4C, H/4, W/4
        rgb = self.up4to3(rgb) + rgb3
        dp = self.up4to3(dp) + dp3
        rgb = self.decoder3(rgb)
        dp = self.decoder3(dp)
        dp,rgb = self.cross3(dp,rgb)

        # Decoder-L2: [4C, H/4, W/4] → [2C, H/2, W/2]
        rgb = self.up3to2(rgb) + rgb2
        dp = self.up3to2(dp) + dp2
        rgb = self.decoder2(rgb)
        dp = self.decoder2(dp)
        dp,rgb = self.cross2(dp,rgb)

        # Decoder-L1: [2C, H/2, W/2] → [C, H, W]
        rgb = self.up2to1(rgb) + rgb1
        dp = self.up2to1(dp) + dp1
        rgb = self.decoder1(rgb)
        dp = self.decoder1(dp)
        dp,rgb = self.cross1(dp,rgb)

        # Multimodal Feature Fusion
        fs = self.fusion(dp,rgb)
        fs = torch.cat([dp, rgb, fs],1)

        # Multi-stage Feature Refinement
        rf = self.refine(fs)

        # Initial Output Generation
        out = self.iniout(rf)[:, :, :H, :W] + dp0

        # Dynamic Edge Sharpening
        depth = self.edge(rgb0, out)

        return depth
    

    