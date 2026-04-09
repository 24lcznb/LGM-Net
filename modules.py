import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, LeakyReLU, AdaptiveAvgPool2d, Parameter, Sigmoid, Dropout, Identity

# SPN: Scale-Propagation Normalization
class SPN(Module):
    def __init__(self, dim):
        super(SPN, self).__init__()
        self.eps = 1e-6
        self.conv = Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        u = x.mean(dim=1, keepdim=True)
        s = x.std(dim=1, unbiased=False, keepdim=True)
        x_norm = (x - u) / (s + self.eps)
        scale = self.conv(x_norm)
        return x * scale


# SGM: Simple Gate Mechanism
class SGM(Module):
    def __init__(self):
        super(SGM, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# SCA: Simple Channel Attention
class SCA(Module):
    def __init__(self, dim):
        super(SCA, self).__init__()
        self.pool = AdaptiveAvgPool2d(1)
        self.conv = Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        attn = self.conv(self.pool(x))
        return x * attn


# SAGSP Block: Simple Attention and Gate mechanism with Scale-Propagation Block
class SAGSP(Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.01, gconv_groups=1):
        super(SAGSP, self).__init__()
        dw_channel = c * DW_Expand
        
        self.norm1 = SPN(c)
        self.conv1 = Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sgm1 = SGM()
        self.sca = SCA(dw_channel // 2)
        self.conv3 = Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=3, padding=1, stride=1, groups=gconv_groups, bias=True)
        self.dropout1 = Dropout(drop_out_rate) if drop_out_rate > 0. else Identity()
        
        self.norm2 = SPN(c)
        ffn_channel = FFN_Expand * c
        self.conv4 = Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv6 = Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1, groups=ffn_channel, bias=True)
        self.sgm2 = SGM()
        self.conv7 = Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dropout2 = Dropout(drop_out_rate) if drop_out_rate > 0. else Identity()
        
        self.theta = Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sgm1(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        
        y1 = inp + x * self.theta
        
        x = self.norm2(y1)
        x = self.conv4(x)
        x = self.conv6(x)
        x = self.sgm2(x)
        x = self.conv7(x)
        x = self.dropout2(x)
        
        y = y1 + x * self.gamma
        
        return y


# DCMA: Depth-separated Cross-Modal Attention Module
class DCMA(Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm = SPN(c)
        self.dw_shared = Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c)
        self.dw_r = Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c)
        self.dw_d = Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c)

        self.alpha = Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, f_dp, f_rgb):
        norm_dp = self.norm(f_dp)
        norm_rgb = self.norm(f_rgb)

        Q_D = self.dw_shared(norm_dp) + self.dw_d(norm_dp)
        Q_R = self.dw_shared(norm_rgb) + self.dw_r(norm_rgb)

        Q_D_perm = Q_D.permute(0, 2, 3, 1)  
        K_R_T_perm = Q_R.permute(0, 2, 1, 3) 

        V_D = f_dp.permute(0, 2, 3, 1) 
        V_R = f_rgb.permute(0, 2, 3, 1)

        attention = torch.matmul(Q_D_perm, K_R_T_perm) * self.scale

        F_r2d = torch.matmul(torch.softmax(attention, dim=-1), V_R)  
        F_d2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_D)  

        F_R = f_rgb + F_d2r.permute(0, 3, 1, 2) * self.alpha
        F_D = f_dp + F_r2d.permute(0, 3, 1, 2) * self.beta
        
        return F_D, F_R


# MDAF: Multimodal Dual-Attention Fusion Module
class MDAF(Module):
    def __init__(self, dim):
        super().__init__()
        
        self.ca_pool = AdaptiveAvgPool2d(1)
        self.ca_dw = Conv2d(2 * dim, 2 * dim, kernel_size=3, padding=1, groups=2 * dim)
        self.ca_relu = LeakyReLU()
        self.ca_1x1 = Conv2d(2 * dim, 2 * dim, kernel_size=1)
        self.ca_sig = Sigmoid()

        self.sa_conv = Conv2d(2, 1, kernel_size=3, padding=1)
        self.sa_sig = Sigmoid()

        self.mid_dw = Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sgm = SGM()
        self.mid_1x1 = Conv2d(dim // 2, dim, kernel_size=1)

        self.sagsp = SAGSP(5 * dim)
        self.out_1x1 = Conv2d(5 * dim, dim, kernel_size=1)

    def forward(self, f_dp, f_rgb):
        f_rd = torch.cat([f_rgb, f_dp], dim=1)

        a_c = self.ca_pool(f_rd)
        a_c = self.ca_dw(a_c)
        a_c = self.ca_relu(a_c)
        a_c = self.ca_1x1(a_c)
        a_c = self.ca_sig(a_c)

        a_c_r, a_c_d = a_c.chunk(2, dim=1)

        max_pool, _ = torch.max(f_rd, dim=1, keepdim=True)
        mean_pool = torch.mean(f_rd, dim=1, keepdim=True)
        a_s = torch.cat([max_pool, mean_pool], dim=1)
        a_s = self.sa_conv(a_s)
        a_s = self.sa_sig(a_s)

        f_r_da = f_rgb * a_c_r * a_s
        f_d_da = f_dp * a_c_d * a_s

        f_da = f_r_da + f_d_da
        f_da = self.mid_dw(f_da)
        f_da = self.sgm(f_da)
        f_da = self.mid_1x1(f_da)

        f_concat = torch.cat([f_rgb, f_dp, f_r_da, f_d_da, f_da], dim=1)
        d_r = self.sagsp(f_concat)
        d_r = self.out_1x1(d_r)

        return d_r


# EADE: Edge-Aware Dynamic Enhancement Module
class EADE(Module):
    def __init__(self):
        super().__init__()
        
        self.conv1x1 = Conv2d(3, 1, kernel_size=1)
        
        kernel = torch.tensor([[0., -1., 0.],
                               [-1., 4., -1.],
                               [0., -1., 0.]])
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.laplacian = Parameter(data=kernel, requires_grad=False)
        
        self.conv3x3_i = Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv5x5_i = Conv2d(1, 1, kernel_size=5, padding=2)
        
        self.conv3x3_d = Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv5x5_d = Conv2d(1, 1, kernel_size=5, padding=2)
        
        self.eta = Parameter(torch.zeros((1, 1, 1, 1)), requires_grad=True)
        self.lam = Parameter(torch.zeros((1, 1, 1, 1)), requires_grad=True)

    def forward(self, I_hr, D_hr):
        I_hr_1 = self.conv1x1(I_hr)
        
        I_edge = F.conv2d(I_hr_1, self.laplacian, padding=1)
        D_edge = F.conv2d(D_hr, self.laplacian, padding=1)
        
        delta_I = torch.abs(self.conv5x5_i(I_hr_1) - self.conv3x3_i(I_hr_1))
        delta_D = torch.abs(self.conv5x5_d(D_hr) - self.conv3x3_d(D_hr))
        
        M = torch.sigmoid(delta_I + delta_D)

        D_hr_edge = D_hr + self.eta * (I_edge * M) + self.lam * (D_edge * M)
        
        return D_hr_edge