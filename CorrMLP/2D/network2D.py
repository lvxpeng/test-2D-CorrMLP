import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal


########################################################
# Networks
########################################################

class CorrMLP(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 enc_channels: int = 8,
                 dec_channels: int = 16,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Encoder = Conv_encoder(in_channels=in_channels,
                                    channel_num=enc_channels,
                                    use_checkpoint=use_checkpoint)
        self.Decoder = MLP_decoder(in_channels=enc_channels,
                                   channel_num=dec_channels,
                                   use_checkpoint=use_checkpoint)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, fixed, moving):
        x_fix = self.Encoder(fixed) #固定图像编码器
        x_mov = self.Encoder(moving)  #移动图像编码器
        flow = self.Decoder(x_fix, x_mov)  #形变场
        warped = self.SpatialTransformer(moving, flow) #使用形变场对移动图像进行重构

        return warped, flow

    ########################################################


# Encoder/Decoder
########################################################

class Conv_encoder(nn.Module):
    def __init__(self,
                 in_channels: int,    #输入通道数
                 channel_num: int,    #输出通道数
                 use_checkpoint: bool = False):
        super().__init__()

        self.Convblock_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.Convblock_2 = Conv_block(channel_num, channel_num*2, use_checkpoint)
        self.Convblock_3 = Conv_block(channel_num*2, channel_num*4, use_checkpoint)
        self.Convblock_4 = Conv_block(channel_num*4, channel_num*8, use_checkpoint)
        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.Convblock_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.Convblock_2(x)
        x = self.downsample(x_2)
        x_3 = self.Convblock_3(x)
        x = self.downsample(x_3)
        x_4 = self.Convblock_4(x)

        return [x_1, x_2, x_3, x_4]



class MLP_decoder(nn.Module):

    # def __init__(self,
    #              in_channels: int,
    #              channel_num: int,
    #              use_checkpoint: bool = False):
    #     super().__init__()
    def __init__(self, in_channels: int, channel_num: int, use_checkpoint: bool = False,
                 corr_max_disp: int = 1):  # Added corr_max_disp
        super().__init__()

        self.mlp_11 = CMWMLP_block(in_channels, channel_num, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_12 = CMWMLP_block(in_channels * 2, channel_num * 2, use_corr=True,corr_max_disp=corr_max_disp, use_checkpoint=use_checkpoint)
        self.mlp_13 = CMWMLP_block(in_channels * 4, channel_num * 4, use_corr=True, corr_max_disp=corr_max_disp, use_checkpoint=use_checkpoint)
        self.mlp_14 = CMWMLP_block(in_channels * 8, channel_num * 8, use_corr=True, corr_max_disp=corr_max_disp, use_checkpoint=use_checkpoint)

        self.mlp_21 = CMWMLP_block(channel_num, channel_num, use_corr=True, corr_max_disp=corr_max_disp, use_checkpoint=use_checkpoint)
        self.mlp_22 = CMWMLP_block(channel_num * 2, channel_num * 2, use_corr=True, corr_max_disp=corr_max_disp,use_checkpoint=use_checkpoint)
        self.mlp_23 = CMWMLP_block(channel_num * 4, channel_num * 4, use_corr=True, corr_max_disp=corr_max_disp,use_checkpoint=use_checkpoint)

        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num * 2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num * 4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num * 8)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='bilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

        self.reghead_1 = RegHead_block(channel_num, use_checkpoint)
        self.reghead_2 = RegHead_block(channel_num * 2, use_checkpoint)
        self.reghead_3 = RegHead_block(channel_num * 4, use_checkpoint)
        self.reghead_4 = RegHead_block(channel_num * 8, use_checkpoint)

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4 = x_mov

        # Step 1
        x_4 = self.mlp_14(x_fix_4, x_mov_4)
        flow_4 = self.reghead_4(x_4)

        # Step 2
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)

        x = self.mlp_13(x_fix_3, x_mov_3)
        x_3 = self.mlp_23(x, self.upsample_3(x_4))

        x = self.reghead_3(x_3)
        flow_3 = x + flow_4_up

        # Step 3
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.mlp_12(x_fix_2, x_mov_2)
        x_2 = self.mlp_22(x, self.upsample_2(x_3))

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 4
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.mlp_11(x_fix_1, x_mov_1)
        x_1 = self.mlp_21(x, self.upsample_1(x_2))

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


########################################################
# Blocks
########################################################

class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        # new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class Conv_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels:int ,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm2d(out_channels)

        self.Conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm2d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)

        return x_out



class RegHead_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.reg_head = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding='same')  # 2 for 2D flow
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.reg_head, x_in)
        else:
            x_out = self.reg_head(x_in)

        return x_out


class PatchExpanding_block(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        # Upsample spatial dimensions (H, W) by 2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Halve the channel dimension
        self.conv = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(embed_dim // 2) # Or GroupNorm, or remove
        self.relu = nn.LeakyReLU(0.2) # Add activation

    def forward(self, x_in):
        # x_in shape: (b, c, h, w)
        x = self.upsample(x_in)
        x = self.conv(x)
        x = self.norm(x)
        x_out = self.relu(x)
        return x_out


class CMWMLP_block(nn.Module):
    def __init__(self, in_channels, num_channels, use_corr=True, corr_max_disp=1, use_checkpoint=False): # Added corr_max_disp
        super().__init__()
        self.use_corr = use_corr
        self.corr_channels = 0
        if use_corr:
            # Use the standard correlation implementation from Fix 3, Option 1
            self.Corr = Correlation(max_disp=corr_max_disp, use_checkpoint=use_checkpoint)
            self.corr_channels = (2 * corr_max_disp + 1)**2 # Calculate channels dynamically
            conv_in_channels = in_channels * 2 + self.corr_channels
            # If Simple Product was used instead:
            # self.Corr = ElementwiseProductMean(...)
            # self.corr_channels = 1
            # conv_in_channels = in_channels * 2 + self.corr_channels
        else:
            conv_in_channels = in_channels * 2

        self.Conv = nn.Conv2d(conv_in_channels, num_channels, kernel_size=3, stride=1, padding='same')
        self.mlpLayer = MultiWinMlpLayer(num_channels, use_checkpoint=use_checkpoint)
        # Ensure RCAB uses the correct num_channels
        self.channel_attention_block = RCAB(num_channels, use_checkpoint=use_checkpoint)

    def forward(self, x_1, x_2):
        if self.use_corr:
            x_corr = self.Corr(x_1, x_2) # Output shape (b, corr_channels, h, w)
            # No repeat needed if Correlation layer is correct
            x = torch.cat([x_1, x_corr, x_2], dim=1)
            x = self.Conv(x)
        else:
            x = torch.cat([x_1, x_2], dim=1)
            x = self.Conv(x)

        shortcut = x
        x = x.permute(0, 2, 3, 1)  # n,h,w,c
        x = self.mlpLayer(x)
        x = self.channel_attention_block(x) # Expects (n, h, w, c), ensure CALayer inside is fixed
        x = x.permute(0, 3, 1, 2)  # n,c,h,w
        x_out = x + shortcut
        return x_out


class MultiWinMlpLayer(nn.Module):  # input shape: n, h, w, d, c
    """The multi-window gated MLP block."""

    def __init__(self, num_channels, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.LayerNorm = nn.LayerNorm(num_channels)
        # self.WinGmlpLayer_1 = WinGmlpLayer(win_size=[3, 3, 3], num_channels=num_channels, use_bias=use_bias)
        # self.WinGmlpLayer_2 = WinGmlpLayer(win_size=[5, 5, 5], num_channels=num_channels, use_bias=use_bias)
        # self.WinGmlpLayer_3 = WinGmlpLayer(win_size=[7, 7, 7], num_channels=num_channels, use_bias=use_bias)
        self.WinGmlpLayer_1 = WinGmlpLayer(win_size=[3, 3], num_channels=num_channels, use_bias=use_bias)
        self.WinGmlpLayer_2 = WinGmlpLayer(win_size=[5, 5], num_channels=num_channels, use_bias=use_bias)
        self.WinGmlpLayer_3 = WinGmlpLayer(win_size=[7, 7], num_channels=num_channels, use_bias=use_bias)

        self.reweight = MLP(num_channels, num_channels // 4, num_channels * 3)
        self.out_project = nn.Linear(num_channels, num_channels, bias=use_bias)

    def forward_run(self, x_in):


            n, h, w, c = x_in.shape  # Use actual c from input
            x = self.LayerNorm(x_in)

            # Window gMLP
            x_1 = self.WinGmlpLayer_1(x)
            x_2 = self.WinGmlpLayer_2(x)
            x_3 = self.WinGmlpLayer_3(x)

            # Calculate adaptive weights 'a'
            # a_base shape: (n, c)
            a_base = (x_1 + x_2 + x_3).permute(0, 3, 1, 2).flatten(2).mean(2)

            # Project to get 3 weights per channel: output shape (n, c*3)
            # Reshape to (n, c, 3), permute to (3, n, c) for softmax over branches
            # Then unsqueeze for broadcasting with (n, h, w, c) -> target shape (3, n, 1, 1, c)
            weights = self.reweight(a_base).view(n, c, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(3)
            # weights shape is now (3, n, 1, 1, c)

            # Apply weights using broadcasting
            x = x_1 * weights[0] + x_2 * weights[1] + x_3 * weights[2]
            x = self.out_project(x)

            x_out = x + x_in
            return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)
        return x_out


class WinGmlpLayer(nn.Module):  # input shape: n, h, w, d, c

    def __init__(self, win_size, num_channels, factor=2, use_bias=True):
        super().__init__()
        assert len(win_size) == 2,"Window size must be 2D"
        self.fh = win_size[0]
        self.fw = win_size[1]
        # self.fd = win_size[2]

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels * factor, bias=use_bias)  # c->c*factor
        self.gelu = nn.GELU()
        self.SpatialGatingUnit = SpatialGatingUnit(num_channels * factor,
                                                   n=self.fh * self.fw)
        # self.SpatialGatingUnit = SpatialGatingUnit(num_channels * factor,
        #                                            n=self.fh * self.fw * self.fd) # c*factor->c*factor//2
        self.out_project = nn.Linear(num_channels * factor // 2, num_channels, bias=use_bias)  # c*factor//2->c

    def forward(self, x):

        _, h, w, _ = x.shape


        pad_l = pad_t = 0
        pad_b = (self.fh - h % self.fh) % self.fh
        pad_r = (self.fw - w % self.fw) % self.fw
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        gh, gw = x.shape[1] // self.fh, x.shape[2] // self.fw
        x = split_images(x, patch_size=(self.fh, self.fw))  # n (gh gw) (fh fw) c

        # gMLP: Local (block) mixing part, provides local block communication.
        shortcut = x
        x = self.LayerNorm(x)
        x = self.in_project(x)
        x = self.gelu(x)
        x = self.SpatialGatingUnit(x)
        x = self.out_project(x)
        x = x + shortcut

        x = unsplit_images(x, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        if pad_b > 0 or pad_r > 0:
            x = x[:, :h, :w, :].contiguous()

        return x


class SpatialGatingUnit(nn.Module):  # input shape: n (gh gw gd) (fh fw fd) c

    def __init__(self, c, n, use_bias=True):
        super().__init__()

        self.Dense_0 = nn.Linear(n, n, use_bias)
        self.LayerNorm = nn.LayerNorm(c // 2)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2
        u, v = torch.split(x, c, dim=-1)

        v = self.LayerNorm(v)
        v = v.permute(0, 1, 3, 2)  # n, (gh gw gd), c/2, (fh fw fd)
        v = self.Dense_0(v)
        v = v.permute(0, 1, 3, 2)  # n (gh gw gd) (fh fw fd) c/2

        return u * (v + 1.0)


class RCAB(nn.Module):  # input shape: n, h, w, c
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def __init__(self, num_channels, reduction=4, lrelu_slope=0.2, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.LayerNorm = nn.LayerNorm(num_channels)
        # Conv layers expect (N, C, H, W) input
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.leaky_relu = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        # CALayer expects (N, H, W, C) input (based on our previous fix)
        self.channel_attention = CALayer(num_channels=num_channels, reduction=reduction, use_bias=use_bias)

    def forward_run(self, x): # Input x shape: (n, h, w, c)
        shortcut = x
        x_norm = self.LayerNorm(x) # Output shape: (n, h, w, c)

        # Permute for Conv2d: (n, h, w, c) -> (n, c, h, w)
        x_conv_input = x_norm.permute(0, 3, 1, 2)

        x_conv = self.conv1(x_conv_input) # Input (n, c, h, w), Output (n, c, h, w)
        x_conv = self.leaky_relu(x_conv)
        x_conv = self.conv2(x_conv) # Output (n, c, h, w)

        # Permute back for CALayer: (n, c, h, w) -> (n, h, w, c)
        x_permuted_back = x_conv.permute(0, 2, 3, 1)

        # Apply channel attention
        x_attn = self.channel_attention(x_permuted_back) # Input (n, h, w, c), Output (n, h, w, c)

        # Add residual connection
        x_out = x_attn + shortcut # Both (n, h, w, c)

        return x_out

    def forward(self, x): # Input x shape: (n, h, w, c)
        if self.use_checkpoint and x.requires_grad:
            # Pass the input x directly to checkpoint, which calls forward_run
            x_out = checkpoint.checkpoint(self.forward_run, x, use_reentrant=False) # Use use_reentrant=False for newer PyTorch versions
        else:
            x_out = self.forward_run(x)
        return x_out



class CALayer(nn.Module):  # input shape: n, h, w, c
    """Squeeze-and-excitation block for channel attention."""
    def __init__(self, num_channels, reduction=4, use_bias=True):
        super().__init__()
        # Expect input N, H, W, C
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Efficient way to average H, W dimensions
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        # x_in shape: (n, h, w, c)
        b, h, w, c = x_in.shape
        # Global Average Pooling
        # y = torch.mean(x_in, dim=(1, 2), keepdim=True) # Manual average
        y = self.avg_pool(x_in.permute(0, 3, 1, 2)).view(b, c) # Use AdaptiveAvgPool2d: N, C, H, W -> N, C, 1, 1 -> N, C

        # Fully Connected Layers (Excitation)
        y = self.fc(y).view(b, 1, 1, c)   # N, C -> N, 1, 1, C

        # Scale original input
        x_out = x_in * y    # N, H, W, C * N, 1, 1, C (broadcasting)
        return x_out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Correlation(nn.Module):
    def __init__(self, max_disp=4, use_checkpoint=False):  # Increased default max_disp
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.max_disp = max_disp
        self.corr_patch_size = 2 * self.max_disp + 1
        # No learnable parameters needed, it's a calculation

    def forward_run(self, x1, x2):
        b, c, h, w = x1.shape
        # Pad x2 for searching
        x2_padded = nnf.pad(x2, [self.max_disp] * 4)  # Pad H and W dimensions

        # Create correlation volume/map
        corr_list = []
        # Iterate through all shifts in the search window (-max_disp to +max_disp)
        for i in range(self.corr_patch_size):  # Vertical shift
            for j in range(self.corr_patch_size):  # Horizontal shift
                # Crop the shifted version of x2_padded to match x1's size
                x2_shifted = x2_padded[:, :, i:(i + h), j:(j + w)]
                # Calculate dot product (mean over channels)
                corr = torch.mean(x1 * x2_shifted, dim=1, keepdim=True)  # Shape: (b, 1, h, w)
                corr_list.append(corr)

        # Stack correlations along the channel dimension
        out = torch.cat(corr_list, dim=1)  # Shape: (b, corr_patch_size*corr_patch_size, h, w)
        # Normalize? Optional, sometimes helpful e.g., / c
        return out / c

    def forward(self, x_1, x_2):

        if self.use_checkpoint and x_1.requires_grad and x_2.requires_grad:
            return checkpoint.checkpoint(self.forward_run, x_1, x_2)
        else:
            return self.forward_run(x_1, x_2)

########################################################
# Functions
########################################################

def split_images(x, patch_size):  # n, h, w, d, c
    """Image to patches."""

    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unsplit_images(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x

