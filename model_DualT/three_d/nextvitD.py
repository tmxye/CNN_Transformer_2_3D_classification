# Copyright (c) ByteDance Inc. All rights reserved.
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import nn
# from utils import merge_pre_bn


NORM_EPS = 1e-5



def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight *(pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv3d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias



class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm3d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool3d((2, 2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))

class PatchEmbedA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbedA, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool3d((1, 2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    # nn.AvgPool3d(kernel_size=pooling_r, stride=pooling_r),
                    nn.AvgPool3d(kernel_size=(1, pooling_r, pooling_r), stride=(1, pooling_r, pooling_r)),
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        # out = self.k4(out) # k4

        return out


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
                                       # padding=1, groups=1, bias=False)
        self.pooling_r = 4
        self.scconv = SCConv(out_channels, out_channels, stride=1, padding=1, dilation=1,
            groups=out_channels // head_dim, pooling_r=self.pooling_r, norm_layer=nn.BatchNorm3d)

        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # out = self.group_conv3x3(x)
        out = self.scconv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class Mlp(nn.Module):
    # def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
    def __init__(self, in_features, out_features=None, mlp_ratio=1, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv3d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        print("self.hidden_dim ",self.hidden_dim.shape)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class NCB(nn.Module):
    """
    Next Convolution Block
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        # print("x", x.shape)
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        # x = x + self.mlp_path_dropout(self.mlp(out))
        x = out
        # print("xout", x.shape)
        # # xout torch.Size([4, 96, 8, 56, 56])
        # # x torch.Size([4, 256, 8, 56, 56])
        # # xout torch.Size([4, 192, 4, 28, 28])
        # # x torch.Size([4, 256, 4, 28, 28])
        # # xout torch.Size([4, 384, 2, 14, 14])
        # # x torch.Size([4, 384, 2, 14, 14])
        # # xout torch.Size([4, 384, 2, 14, 14])
        # # x torch.Size([4, 512, 2, 14, 14])
        # # xout torch.Size([4, 768, 1, 7, 7])
        return x


class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NTB(nn.Module):
    """
    Next Transformer Block
    """
    def __init__(
            self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
            mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0,
    ):
        super(NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        self.mlpC = None
        norm_func = partial(nn.BatchNorm3d, eps=NORM_EPS)

        # self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhsa_out_channels = out_channels // 2
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x
        # out = rearrange(out, "b c h w -> b (h w) c")  # b n c
        out = out.reshape(B, C, T * H * W).permute(0, 2, 1)
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        # x = x + rearrange(out, "b (h w) c -> b c h w", h=H)
        # x = x + out.reshape(B, C, T, H, W)
        x = x + out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        # out = self.projection(x)
        out = x
        out = out + self.mhca_path_dropout(self.mhca(out))

        # x = torch.cat([x, out], dim=1)

        Trans1, Trans2 = torch.split(x, (self.mhsa_out_channels//2, self.mhsa_out_channels//2), dim=1)
        CNN1, CNN2 = torch.split(out, (self.mhsa_out_channels//2, self.mhsa_out_channels//2), dim=1)
        x = torch.cat([Trans1, CNN1, Trans2, CNN2], dim=1)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        if self.mlpC:
            x = x + self.mlp_path_dropout(self.mlp(out))
        else:
            x = out
        return x






class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_pure(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = int(C // 3)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x

class Cross_Attention_pure(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, tokens_q, memory_k, memory_v, shape=None):
        assert shape is not None
        attn = (tokens_q @ memory_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ memory_v).transpose(1, 2).reshape(shape[0], shape[1], shape[2])
        x = self.proj_drop(x)
        return x



class MixBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=2):
        super().__init__()
        dim = dim // 8
        #        pdb.set_trace()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.dim = dim
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim * 8, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim * 8, 1)
        # self.conv = nn.Conv3d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)
        self.dim_conv = int(dim * 0.5)
        self.dim_sa = dim - self.dim_conv
        self.norm_conv1 = nn.BatchNorm3d(self.dim_conv)
        self.norm_sa1 = nn.LayerNorm(self.dim_sa)
        self.conv = nn.Conv3d(self.dim_conv, self.dim_conv, 3, padding=1, groups=self.dim_conv)
        #        self.attn_down = nn.Conv3d(dim // 2, dim // 2, (2 * downsample + 1), padding=downsample, groups=dim // 2, stride=downsample)
        # self.channel_up = nn.Conv3d(dim // 2, 3 * dim // 2, 1)
        self.channel_up = nn.Linear(self.dim_sa, 3 * self.dim_sa)
        self.cross_channel_up_conv = nn.Conv3d(self.dim_conv, 3 * self.dim_conv, 1)
        self.cross_channel_up_sa = nn.Linear(self.dim_sa, 3 * self.dim_sa)
        self.fuse_channel_conv = nn.Linear(self.dim_conv, self.dim_conv)
        self.fuse_channel_sa = nn.Linear(self.dim_sa, self.dim_sa)
        self.num_heads = num_heads
        self.attn = Attention_pure(
            self.dim_sa,
            num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=0.1, proj_drop=drop)
        self.cross_attn = Cross_Attention_pure(
            self.dim_sa,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=0.1, proj_drop=drop)
        self.norm_conv2 = nn.BatchNorm3d(self.dim_conv)
        self.norm_sa2 = nn.LayerNorm(self.dim_sa)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        self.downsample = downsample
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    #        self.sa_weight = nn.Parameter(torch.Tensor([0.5]))
    #        self.conv_weight = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x):
        #        pdb.set_trace()
        # x = x + self.pos_embed(x)
        B, _, T, H, W = x.shape
        residual = x
        # x = self.norm1(x)
        x = self.conv1(x)

        # qkv = x[:, :(self.dim // 2), :]
        # conv = x[:, (self.dim // 2):, :, :]
        qkv = x[:, :self.dim_sa, :, :, :]
        conv = x[:, self.dim_sa:, :, :, :]
        residual_conv = conv
        conv = residual_conv + self.conv(self.norm_conv1(conv))

        #        sa = self.attn_down(qkv)
        #         sa = nn.functional.interpolate(qkv, size=(T // self.downsample, H // self.downsample, W // self.downsample), mode='bilinear')
        T_downsample = T // self.downsample
        H_downsample = H // self.downsample
        # print("H_downsample", H_downsample)       # 7  7  7   7
        W_downsample = W // self.downsample
        if T_downsample < 2:
            T_downsample = 1
        if H_downsample < 7:
            H_downsample = 7
            W_downsample = 7
        sa = nn.functional.interpolate(qkv, size=(T_downsample, H_downsample, W_downsample), mode='trilinear')
        B, _, T_down, H_down, W_down = sa.shape
        sa = sa.flatten(2).transpose(1, 2)
        residual_sa = sa
        sa = self.norm_sa1(sa)
        sa = self.channel_up(sa)
        # print('aaa:', residual_sa.shape, sa.shape)
        # input()
        sa = residual_sa + self.attn(sa)

        ### cross attention ###
        residual_conv_co = conv
        residual_sa_co = sa
        conv_qkv = self.cross_channel_up_conv(self.norm_conv2(conv))
        conv_qkv = conv_qkv.flatten(2).transpose(1, 2)
        # print('sa:', self.norm_sa2(sa).shape)
        # input()
        sa_qkv = self.cross_channel_up_sa(self.norm_sa2(sa))
        # print('sa_qkv', sa_qkv.shape)
        # input()


        B_conv, N_conv, C_conv = conv_qkv.shape
        C_conv = int(C_conv // 3)
        conv_qkv = conv_qkv.reshape(B_conv, N_conv, 3, self.num_heads, C_conv // self.num_heads).permute(2, 0, 3, 1, 4)
        conv_q, conv_k, conv_v = conv_qkv[0], conv_qkv[1], conv_qkv[2]

        B_sa, N_sa, C_sa = sa_qkv.shape
        C_sa = int(C_sa // 3)
        sa_qkv = sa_qkv.reshape(B_sa, N_sa, 3, self.num_heads, C_sa // self.num_heads).permute(2, 0, 3, 1, 4)
        sa_q, sa_k, sa_v = sa_qkv[0], sa_qkv[1], sa_qkv[2]

        # sa -> conv
        conv = self.cross_attn(conv_q, sa_k, sa_v, shape=(B_conv, N_conv, C_conv))
        conv = self.fuse_channel_conv(conv)
        conv = conv.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        conv = residual_conv_co + conv
        # print('conv:', conv.shape)
        # input()
        # conv -> sa
        sa = self.cross_attn(sa_q, conv_k, conv_v, shape=(B_sa, N_sa, C_sa))
        sa = residual_sa_co + self.fuse_channel_sa(sa)
        sa = sa.reshape(B, T_down, H_down, W_down, -1).permute(0, 4, 1, 2, 3).contiguous()
        # print('sa:', sa.shape)
        # input()
        # sa = nn.functional.interpolate(sa, size=(T, H, W), mode='bilinear')
        sa = nn.functional.interpolate(sa, size=(T, H, W), mode='trilinear')
        x = torch.cat([conv, sa], dim=1)
        x = residual + self.drop_path(self.conv2(x))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NextViT(nn.Module):
    # def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000,
    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=2,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 # strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=16, mix_block_ratio=0.75,
                 use_checkpoint=False):
        super(NextViT, self).__init__()
        self.use_checkpoint = use_checkpoint

        # self.stage_out_channels = [[96] * (depths[0]),
        #                            [192] * (depths[1] - 1) + [256],
        #                            [384, 384, 384, 384, 512] * (depths[2] // 5),
        #                            [768] * (depths[3] - 1) + [1024]]
        self.stage_out_channels = [[96] * (depths[0] - 1) + [256],
                                   [192] * (depths[1] - 1) + [256],
                                   # [384, 512] * (depths[2]//2),
                                   # [384, 384, 384, 512] * (depths[2]//4),
                                   [384, 384, 384, 512],
                                   [768] * (depths[3] - 1) + [1024]]

        # Next Hybrid Strategy
        # self.stage_block_types = [[NCB] * depths[0],
        #                           [NCB] * (depths[1] - 1) + [NTB],
        #                           [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
        #                           [NCB] * (depths[3] - 1) + [NTB]]
        self.stage_block_types = [[NCB] * (depths[0] - 1) + [NTB],
                                  [NCB] * (depths[1] - 1) + [NTB],
                                  # [NCB, NTB] * (depths[2]//2),
                                  # [NCB, NTB, NCB, NTB] * (depths[2]//4),
                                  [NCB, NTB, NCB, NTB],
                                  [NCB] * (depths[3] - 1) + [NTB]]

        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=(2, 2, 2)),
            # ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            # ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[0], stem_chs[2], kernel_size=3, stride=(1, 2, 2)),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                # print('output_channels', output_channels)
                # print('block_id', block_id)
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    features.append(layer)
                elif block_type is NTB:
                    layer = NTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm3d(output_channel, eps=NORM_EPS)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
        print('initialize_weights...')
        self._initialize_weights()
        embed_dim = [256, 256, 512, 1024]
        num_heads = [8, 8, 8, 8]            # [1, 2, 5, 8]
        downsamples = [8, 4, 2, 1]
        self.blocks1 = MixBlock(dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=drop, attn_drop=attn_drop, drop_path=0., norm_layer=nn.LayerNorm, downsample=downsamples[0])
        self.blocks2 = MixBlock(dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=drop, attn_drop=attn_drop, drop_path=0., norm_layer=nn.LayerNorm, downsample=downsamples[1])
        self.blocks3 = MixBlock(dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=drop, attn_drop=attn_drop, drop_path=0., norm_layer=nn.LayerNorm, downsample=downsamples[2])
        self.blocks4 = MixBlock(dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=drop, attn_drop=attn_drop, drop_path=0., norm_layer=nn.LayerNorm, downsample=downsamples[3])





    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, NCB) or isinstance(module, NTB):
                module.merge_bn()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, PET=None, CT=None):
        if PET !=None:
            PET = PET
            CT = CT
        else:
            PET = x
            CT = x
        x = self.stem(x)
        PET = self.stem(PET)
        CT = self.stem(CT)
        for idx, layer in enumerate(self.features):
            if idx < 2:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(layer, x)
                    PET = checkpoint.checkpoint(layer, PET)
                    CT = checkpoint.checkpoint(layer, CT)
                else:
                    x = layer(x)
                    PET = layer(PET)
                    CT = layer(CT)
                if idx == 1:
                    x = x + PET + CT
                    # x = self.blocks1(x)
                    # print("xA", x.shape)
            elif 2 <= idx <4:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(layer, x)
                    PET = checkpoint.checkpoint(layer, PET)
                    CT = checkpoint.checkpoint(layer, CT)
                else:
                    x = layer(x)
                    PET = layer(PET)
                    CT = layer(CT)
                if idx == 3:
                    x = x + PET + CT
                    # x = self.blocks2(x)
                    # print("xB", x.shape)
            elif 4 <= idx < 8:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(layer, x)
                    PET = checkpoint.checkpoint(layer, PET)
                    CT = checkpoint.checkpoint(layer, CT)
                else:
                    x = layer(x)
                    PET = layer(PET)
                    CT = layer(CT)
                if idx == 7:
                    x = x + PET + CT
                    # x = self.blocks3(x)
                    # print("xC", x.shape)
            else:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(layer, x)
                    PET = checkpoint.checkpoint(layer, PET)
                    CT = checkpoint.checkpoint(layer, CT)
                else:
                    x = layer(x)
                    PET = layer(PET)
                    CT = layer(CT)
                if idx == 9:
                    x = x + PET + CT
                    # x = self.blocks4(x)
                    # print("xD", x.shape)

        # center.append(x)
        # print("center[1]", center[1].shape)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x


@register_model
def nextvit_small(pretrained=False, pretrained_cfg=None, **kwargs):
    # model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, **kwargs)
    model = NextViT(stem_chs=[32, 64, 64], depths=[2, 2, 4, 2], path_dropout=0.1, **kwargs)
    return model


@register_model
def nextvit_base(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2, **kwargs)
    return model


@register_model
def nextvit_large(pretrained=False, pretrained_cfg=None, **kwargs):
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 30, 3], path_dropout=0.2, **kwargs)
    return model
