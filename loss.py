import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import interpolate, conv2d

# Compute mean and standard deviation of depth maps
class MaskedMS(Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, depth, hole):
        mask_num = torch.sum(hole, dim=(2, 3), keepdim=True)
        mask_num[mask_num == 0] = self.eps
        depth_mean = (torch.sum(depth * hole, dim=(2, 3), keepdim=True) / mask_num)
        depth_std = torch.sum(torch.abs((depth - depth_mean) * hole), dim=(2, 3), keepdim=True) / mask_num
        return depth_mean, depth_std + self.eps

# L1 loss
class MaskedDataLoss(Module):
    def __init__(
            self,
    ) -> None:
        super(MaskedDataLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred: Tensor, gt: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
        assert pred.shape == gt.shape, "pred size != gt size"
        residual = torch.where(mask == 1, pred - gt, torch.zeros_like(gt))
        residual = residual * weight
        loss = torch.sum(torch.abs(residual))
        number_valid = torch.sum(mask) + self.eps
        return loss / number_valid
    
# Compute image gradients (Sobel-like) using fixed convolution kernels
class Gradient2D(Module):
    def __init__(self):
        super(Gradient2D, self).__init__()
        kernel_x = torch.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kernel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        self.weight_x = Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x: Tensor):
        grad_x = conv2d(x, self.weight_x)
        grad_y = conv2d(x, self.weight_y)
        return grad_x, grad_y

# Multi-scale gradient loss (L_grd)
class MaskedMSGradLoss(Module):
    def __init__(self, k: int = 4):
        super(MaskedMSGradLoss, self).__init__()
        self.grad_fun = Gradient2D().cuda()
        self.eps = 1e-6
        self.k = k

    def __gradient_loss__(self, residual: Tensor) -> Tensor:
        loss_x, loss_y = self.grad_fun(residual)
        loss_x = torch.sum(torch.abs(loss_x))
        loss_y = torch.sum(torch.abs(loss_y))
        loss = loss_x + loss_y
        return loss

    def forward(self, pred: Tensor, gt: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
        assert pred.shape == gt.shape, "pred size != gt size"
        residual = torch.where(mask == 1, pred - gt, torch.zeros_like(gt))
        residual = residual * weight
        number_valid = torch.sum(mask) + self.eps
        loss = 0.
        for i in range(self.k):
            scale_factor = 1.0 / (2 ** i)
            if i == 0:
                k_residual = residual
            else:
                k_residual = interpolate(residual, scale_factor=scale_factor, recompute_scale_factor=True)
            loss += self.__gradient_loss__(k_residual)
        return loss / number_valid

# Edge-guided structure consistency loss (L_esc) using Gaussian similarity maps
class GsmapRankingLoss(Module): 
    def __init__(self,
                kernel_size=5,
                edge_kernel_size=5,
                sigma_rgb=35,
                sigma_depth=10,
                edge_sigma=10, 
                stride=1,
                point_num_pic=200,
                point_num_gsmap=40, 
                edge_thres=0.2,
                device='cuda', 
                sample_near_gt_edge=True,
                debug=False):
        
        super(GsmapRankingLoss, self).__init__()
        self.kernel_size = kernel_size
        self.edge_kernel_size = edge_kernel_size
        self.sigma_rgb = sigma_rgb
        self.sigma_depth = sigma_depth
        self.edge_sigma = edge_sigma
        self.stride = stride
        self.point_num_pic = point_num_pic
        self.point_num_gsmap = point_num_gsmap
        self.edge_thres = edge_thres
        self.device = device
        self.sample_near_gt_edge = sample_near_gt_edge
        self.debug = debug
    
    def forward(self, depth_t_b, rgb_t_b, gt_t_b):
        depth_gsmaps_t_b = self.gen_conc_gsmaps(depth_t_b, 
                                                kernel_size=self.kernel_size, 
                                                sigma=self.sigma_depth, 
                                                stride=self.stride, 
                                                mode='grey')
        rgb_gsmaps_t_b = self.gen_conc_gsmaps(rgb_t_b, 
                                            kernel_size=self.kernel_size, 
                                            sigma=self.sigma_rgb, 
                                            stride=self.stride, 
                                            mode='rgb')
        
        sampled_depth_gsmaps_t_b, \
            sampled_rgb_gsmaps_t_b = self.sample_rgbd_gsmaps(depth_gsmaps_t_b,
                                                            rgb_gsmaps_t_b,
                                                            gt_t_b,
                                                            self.sample_near_gt_edge)

        sampled_gsmap_positions = torch.floor(torch.rand(4, self.point_num_gsmap)\
                                              *torch.tensor([[self.kernel_size],
                                                            [self.kernel_size],
                                                            [self.kernel_size],
                                                            [self.kernel_size]]))

        loss = self.rankingloss_gsmap(sampled_depth_gsmaps_t_b, sampled_rgb_gsmaps_t_b, sampled_gsmap_positions)

        return loss

    def gen_conc_gsmaps(self, img_tensor, kernel_size=11, sigma=10, stride=1, mode='grey'):
        img_pad = torch.nn.ReflectionPad2d(kernel_size//2)

        if mode == 'grey': 
            img_tensor_pad = img_pad(img_tensor)
            img_tensor_unfold = torch.nn.functional.unfold(img_tensor_pad, (kernel_size, kernel_size), stride=stride)
            grey_tensor_unfold = img_tensor_unfold
            grey_all = img_tensor.reshape(img_tensor.size()[0], 1, -1)
            grey_diff_blocks = (grey_tensor_unfold - grey_all)*255.
            grey_diff_blocks_tranned = grey_diff_blocks.transpose(2,1)
            grey_rdiff_blocks_tranned = grey_diff_blocks_tranned ** 2
            blocks_grey_gsmaps_1d = torch.exp((-1)*grey_rdiff_blocks_tranned/(2*sigma**2))
            blocks_gsmaps_2d = blocks_grey_gsmaps_1d.reshape(blocks_grey_gsmaps_1d.size()[0], blocks_grey_gsmaps_1d.size()[1], kernel_size, kernel_size)

        if mode == 'rgb':
            img_tensor_pad = img_pad(img_tensor)
            img_tensor_unfold = torch.nn.functional.unfold(img_tensor_pad, (kernel_size, kernel_size), stride=stride)
            rgb_tensor_unfold = img_tensor_unfold.reshape(img_tensor_unfold.size()[0], 3, kernel_size*kernel_size, -1)
            rgb_all = img_tensor.reshape(img_tensor.size()[0], 3, 1, -1)
            rgb_diff_blocks = (rgb_tensor_unfold - rgb_all)*255.
            rgb_diff_blocks_tranned = rgb_diff_blocks.transpose(2,3)
            rgb_rdiff_blocks_tranned = rgb_diff_blocks_tranned ** 2
            rgbsum_rdiff_blocks_tranned = torch.sum(rgb_rdiff_blocks_tranned, dim=1) 
            blocks_rgb_gsmaps_1d = torch.exp((-1)*rgbsum_rdiff_blocks_tranned/(2*3*sigma**2))
            blocks_gsmaps_2d = blocks_rgb_gsmaps_1d.reshape(blocks_rgb_gsmaps_1d.size()[0], blocks_rgb_gsmaps_1d.size()[1], kernel_size, kernel_size)
        
        return blocks_gsmaps_2d
    
    def gsmap_edge_detection(self, depth_t_b, edge_kernel_size=11, edge_sigma=10, edge_thres=0.2):
        h = depth_t_b.size(2)
        w = depth_t_b.size(3)
        depth_invgsmaps_t_b = 1 - self.gen_conc_gsmaps(depth_t_b, 
                                                    kernel_size=edge_kernel_size, 
                                                    sigma=edge_sigma,
                                                    stride=1,
                                                    mode='grey')
        depth_invgsmaps1d_t_b = torch.reshape(depth_invgsmaps_t_b, (depth_invgsmaps_t_b.size(0),
                                                                    depth_invgsmaps_t_b.size(1), 
                                                                    depth_invgsmaps_t_b.size(2)*depth_invgsmaps_t_b.size(3)))
        depth_edge_weights = torch.reshape(torch.mean(depth_invgsmaps1d_t_b, dim=2), (depth_t_b.size(0),
                                                                                1,h,w))
        mask_t_b = torch.zeros_like(depth_edge_weights)
        mask_t_b[depth_edge_weights > edge_thres] = 1
        return mask_t_b

    def rankingloss_gsmap(self, pre_gsmaps_t_b, target_gsmaps_t_b, positions, t=0.03):
        idx1 = positions[0].long()
        idx2 = positions[1].long()
        idy1 = positions[2].long()
        idy2 = positions[3].long()

        z_1_pre = pre_gsmaps_t_b[:, :, idy1, idx1]
        z_2_pre = pre_gsmaps_t_b[:, :, idy2, idx2]
        z_1_target = target_gsmaps_t_b[:, :, idy1, idx1]
        z_2_target = target_gsmaps_t_b[:, :, idy2, idx2]

        rela_gt = z_1_target/(z_2_target+1e-6)
        label_gt = torch.zeros_like(rela_gt)
        label_gt[rela_gt>(1+t)] = 1
        label_gt[rela_gt<(1-t)] = -1

        mask = torch.abs(label_gt)
        ranking_loss_list = mask*torch.log(1+torch.exp(-label_gt*(z_1_pre-z_2_pre)))+(1-mask)*(z_1_pre-z_2_pre)*(z_1_pre-z_2_pre)
        ranking_loss = torch.mean(ranking_loss_list)
        return ranking_loss
    
    def sample_rgbd_gsmaps(self, depth_gsmaps_t_b, rgb_gsmaps_t_b, gt_t_b, sample_near_depth_edge=True):
        h = gt_t_b.size(2)
        w = gt_t_b.size(3)

        if sample_near_depth_edge:

            sampled_depth_gsmaps_t_b = torch.tensor([]).to(self.device)
            sampled_rgb_gsmaps_t_b = torch.tensor([]).to(self.device)
            if self.debug:
                hs_b = torch.tensor([]).to(self.device)
                ws_b = torch.tensor([]).to(self.device)

            mask_t_b = self.gsmap_edge_detection(depth_t_b=gt_t_b, 
                                                edge_kernel_size=self.edge_kernel_size,
                                                edge_sigma=self.edge_sigma,
                                                edge_thres=self.edge_thres)

            for batch_i, mask_t in enumerate(mask_t_b):

                indexs_t = torch.nonzero(mask_t)
                if indexs_t.size(0) != 0:
                    indexs_rp_t = indexs_t[torch.randperm(indexs_t.size(0))]
                    if indexs_rp_t.size(0) > self.point_num_pic:
                        indexs_sampled_t = indexs_rp_t[:self.point_num_pic]
                    else:
                        rate = self.point_num_pic//indexs_t.size(0)+1
                        

                        index_rp_repeat_t = indexs_rp_t.repeat(rate, 1)
                        indexs_sampled_t = index_rp_repeat_t[:self.point_num_pic]

                    indexs_sampled_resize_t = indexs_sampled_t.transpose(1,0)

                    hs = indexs_sampled_resize_t[1]
                    ws = indexs_sampled_resize_t[2]
                else:
                    hs = torch.randint(0, h, (1, self.point_num_pic)).squeeze(0).to(self.device)
                    ws = torch.randint(0, w, (1, self.point_num_pic)).squeeze(0).to(self.device)

                if self.debug:
                    hs_b = torch.cat((hs_b, hs.unsqueeze(0)), dim=0)
                    ws_b = torch.cat((ws_b, ws.unsqueeze(0)), dim=0)

                pos1d = w * hs + ws

                sampled_depth_gsmaps_t = depth_gsmaps_t_b[batch_i, pos1d].unsqueeze(0)
                sampled_rgb_gsmaps_t = rgb_gsmaps_t_b[batch_i, pos1d].unsqueeze(0)

                sampled_depth_gsmaps_t_b = torch.cat((sampled_depth_gsmaps_t_b, sampled_depth_gsmaps_t), dim=0)
                sampled_rgb_gsmaps_t_b = torch.cat((sampled_rgb_gsmaps_t_b, sampled_rgb_gsmaps_t), dim=0)

        else:
            sampled_positions = torch.randint(0, h*w, (self.point_num_pic,1)).squeeze(1)
            sampled_depth_gsmaps_t_b = depth_gsmaps_t_b[:, sampled_positions]
            sampled_rgb_gsmaps_t_b = rgb_gsmaps_t_b[:, sampled_positions]
        
        return sampled_depth_gsmaps_t_b, sampled_rgb_gsmaps_t_b

# Total Loss = L1 & L_std & L_grd & L_esc
class TotalLoss(Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.dataloss = MaskedDataLoss()
        self.msgradloss = MaskedMSGradLoss()
        self.maskedmeanstd = MaskedMS()
        self.escloss = GsmapRankingLoss()

    def standardize(self, pred, gt, hole_gt):
        depth = torch.cat([pred, gt], dim=1)
        hole = torch.cat([hole_gt, hole_gt], dim=1)
        t_d, s_d = self.maskedmeanstd(depth, hole)
        sta_depth = (depth - t_d) / s_d * hole
        return sta_depth.split(pred.shape[1], dim=1)

    def forward(self, pred, rgb, gt, hole_raw, hole_gt):
        # 1. Absolute-Domain Supervision Loss: L1
        loss_abs = self.dataloss(pred, gt, hole_raw, torch.ones_like(gt))
        sta_pred, sta_gt = self.standardize(pred, gt, hole_gt)
        # 2. Mixed-Scale Standardized Loss: L_std
        loss_std = self.dataloss(sta_pred, sta_gt, hole_gt, torch.ones_like(gt))
        # 3. Multi-Scale Gradient Loss: L_grd
        loss_grd = self.msgradloss(sta_pred, sta_gt, hole_gt, torch.ones_like(gt))
        # 4. Edge Structure Consistency Loss: L_esc
        loss_esc = self.escloss(pred, rgb, gt)
        # Total Loss
        loss = loss_abs + loss_std + 0.5 * loss_grd + 0.5*loss_esc
        return loss_abs, loss_std, loss_grd, loss_esc, loss
