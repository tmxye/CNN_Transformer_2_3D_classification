from typing import Callable, Any, Optional, List
from einops import rearrange

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm3d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.SiLU,
        dilation: int = 1
    ):
        super(ConvNormAct, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation = dilation, groups = groups, bias = norm_layer is None)

        self.norm_layer = nn.BatchNorm3d(out_channels) if norm_layer is None else norm_layer(out_channels)
        self.act = activation_layer() if activation_layer is not None else activation_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.act is not None:
            x = self.act(x)
        return x



import torch.nn.functional as F
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_reduce = nn.Conv3d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv3d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv3d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm3d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv3d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm3d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv3d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm3d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm3d(in_chs),
                nn.Conv3d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x

# GhostBottleneck(input_channel, hidden_channel, output_channel, k, s,
#                               se_ratio=se_ratio)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        xout=self.net(x)
        return xout


class MultiHeadSelfAttention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    """
    def __init__(self, dim, num_heads = 8, dim_head = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        _weight_dim = self.num_heads * self.dim_head
        self.to_qvk = nn.Linear(dim, _weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5

        self.scale_factor = dim ** -0.5

        # Weight matrix for output, Size: num_heads*dim_head X dim
        # Final linear transformation layer
        self.w_out = nn.Linear(_weight_dim, dim, bias = False)

    def forward(self, x):
        qkv = self.to_qvk(x).chunk(3, dim = -1)
        #q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.num_heads), qkv)
        q, k, v =qkv[0],qkv[1],qkv[2]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        attn = torch.softmax(dots, dim = -1)
        out = torch.matmul(attn, v)
        #out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.w_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadSelfAttention(dim, heads, dim_head)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            b, c, h, w = x.shape
            x=x.reshape(b*c*h,w)#add reshape
            x = ff(x) + x
            x=x.reshape(b,c,h,w)
        return x


class InvertedResidual(nn.Module):
    """
    MobileNetv2 InvertedResidual block
    """
    def __init__(self, in_channels, out_channels, stride = 1, expand_ratio = 2, act_layer = nn.SiLU):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_channels, hidden_dim, kernel_size = 1, activation_layer = None))

        # Depth-wise convolution
        layers.append(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size = 3, stride = stride,
                        padding = 1, groups = hidden_dim, activation_layer = act_layer)
        )
        # Point-wise convolution
        layers.append(
            nn.Conv3d(hidden_dim, out_channels, kernel_size = 1, stride = 1, bias = False)
        )
        layers.append(nn.BatchNorm3d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# class PfAAMLayer(nn.Module):
#     def __init__(self, ratio=16):
#         super(PfAAMLayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, _, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c, 1, 1, 1).expand_as(x)
#         z = torch.mean(x, dim=1, keepdim=True).expand_as(x)
#         return x * self.sigmoid(y * z)

class PfAAMLayer(nn.Module):
    def __init__(self, ratio=16):
        super(PfAAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()

        e_lambda = 1e-4
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda


    def forward(self, x):
        b, c, _, _, _ = x.size()
        # print("\view(b, c, 1, 1, 1)", self.avg_pool(x).view(b, c, 1, 1, 1).size())
        # ([2, 384, 1, 1, 1])
        y = self.avg_pool(x).view(b, c, 1, 1, 1).expand_as(x)
        # print("1", torch.mean(x, dim=1, keepdim=True).size())       # ([2, 1, 2, 28, 28])
        z = torch.mean(x, dim=1, keepdim=True).expand_as(x)
        # s = torch.mean(x, dim=0, keepdim=True).expand_as(x)


        #### Sim
        b, c, t, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        # print("x_minus_mu_square", x_minus_mu_square.shape)     # ([16, 256, 32, 32])
        y1 = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5


        # return x * self.sigmoid(y * z * s)
        # return x * self.activaton(y1) * self.sigmoid(y * z * s)
        return x * self.activaton(y1) * self.sigmoid(y * z)


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv3d(dim, dim, 1),
            nn.GELU(),
            # nn.Conv3d(dim, dim, (1,7,7), padding=(0,3,3), groups=dim)     #   54.888M
            nn.Conv3d(dim, dim, (1,11,11), padding=(0,5,5), groups=dim)     #  69.931M
            # nn.Conv3d(dim, dim, (3,11,11), padding=(1,5,5), groups=dim)     # 120.494M
        )

        self.v = nn.Conv3d(dim, dim, 1)
        self.proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        B, C, T, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x



class MobileVitBlockD(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, layers, mlp_dim, stride):
        super(MobileVitBlockD, self).__init__()
        # Local representation
        self.local_representation = nn.Sequential(
            ConvNormAct(in_channels, out_channels // 2, 1),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels // 4

        # self.transformer = Transformer(d_model, layers, 1, 32, mlp_dim, 0.1)
        # self.transformer = Transformer(out_channels // 2, layers, 1, 32, mlp_dim, 0.1)
        self.transformer = ConvMod(out_channels // 4)


        # Down block
        self.Down = nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride=stride, padding=1, groups=out_channels)
        # self.Down = nn.Conv3d(out_channels*2, out_channels, kernel_size = 3, stride=stride, padding=1, groups=out_channels)

        # self.local_representationA = nn.Sequential(
        #     ConvNormAct(out_channels, out_channels // 2, 1),
        #     nn.BatchNorm3d(out_channels // 2),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.transformerA = Transformer(out_channels // 2, layers, 1, 32, mlp_dim, 0.1)

        self.CrossA = PfAAMLayer()


    def forward(self, x):
        # print("AAA", x.shape)
        local_repr = self.local_representation(x)
        local_repr1, local_repr2 = torch.split(local_repr, [self.out_channels, self.out_channels], dim=1)

        _, _, t, h, w = local_repr.shape

        # global_repr = rearrange(local_repr, 'b d (t pt) (h ph) (w pw) -> b (pt ph pw) (t h w) d', ph=1, pw=1, pt=1)
        global_repr = self.transformer(local_repr2)
        # global_repr = rearrange(global_repr, 'b (pt ph pw) (t h w) d -> b d (t pt) (h ph) (w pw)', h=h, w=w, t=t, ph=1, pw=1, pt=1)

        # # Fuse the local and gloval features in the concatenation tensor
        # # fuse_repr = self.fusion_block1(global_repr)
        # # # result = self.fusion_block2(torch.cat([x, fuse_repr], dim = 1))
        # result = torch.cat([local_repr, global_repr], dim = 1)

        global_repr1 = self.transformer(global_repr)
        global_repr2 = self.transformer(global_repr1)

        result = torch.cat([local_repr1, global_repr, global_repr1, global_repr2], dim = 1)

        result = self.Down(result)

        result = self.CrossA(result)
        

        # local_repr = self.local_representationA(result)
        # _, _, t, h, w = local_repr.shape
        #
        # global_repr = rearrange(local_repr, 'b d (t pt) (h ph) (w pw) -> b (pt ph pw) (t h w) d', ph=1, pw=1, pt=1)
        # global_repr = self.transformerA(global_repr)
        # global_repr = rearrange(global_repr, 'b (pt ph pw) (t h w) d -> b d (t pt) (h ph) (w pw)', h=h, w=w, t=t, ph=1, pw=1, pt=1)
        #
        # result = torch.cat([local_repr, global_repr], dim = 1)


        return result




class MobileVitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, layers, mlp_dim, stride):
        super(MobileVitBlock, self).__init__()
        # Local representation
        self.local_representation = nn.Sequential(
            ConvNormAct(in_channels, out_channels // 4, 1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        # self.transformer = Transformer(d_model, layers, 1, 32, mlp_dim, 0.1)
        # self.transformer = Transformer(out_channels // 2, layers, 1, 32, mlp_dim, 0.1)
        self.transformer = ConvMod(out_channels // 4)


        # Down block
        self.Down = nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride=stride, padding=1, groups=out_channels)
        # self.Down = nn.Conv3d(out_channels*2, out_channels, kernel_size = 3, stride=stride, padding=1, groups=out_channels)

        self.CrossA = PfAAMLayer()


    def forward(self, x):
        # print("AAA", x.shape)
        local_repr = self.local_representation(x)

        _, _, t, h, w = local_repr.shape

        # global_repr = rearrange(local_repr, 'b d (t pt) (h ph) (w pw) -> b (pt ph pw) (t h w) d', ph=1, pw=1, pt=1)
        global_repr = self.transformer(local_repr)
        # global_repr = rearrange(global_repr, 'b (pt ph pw) (t h w) d -> b d (t pt) (h ph) (w pw)', h=h, w=w, t=t, ph=1, pw=1, pt=1)

        # # Fuse the local and gloval features in the concatenation tensor
        # # fuse_repr = self.fusion_block1(global_repr)
        # # # result = self.fusion_block2(torch.cat([x, fuse_repr], dim = 1))
        # result = torch.cat([local_repr, global_repr], dim = 1)

        global_repr1 = self.transformer(global_repr)
        global_repr2 = self.transformer(global_repr1)

        result = torch.cat([local_repr, global_repr, global_repr1, global_repr2], dim = 1)

        # result = self.Down(result)

        result = self.CrossA(result)


        return result


import torch
import torch.nn as nn

# from .module_opt import InvertedResidual, MobileVitBlock
model_cfg = {
    # "xxs":{
    #     "features": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
    #     "d": [64, 80, 96],
    #     "expansion_ratio": 2,
    #     "layers": [2, 4, 3]
    # },
    "xxs":{
        # "features": [16, 96, 192, 64, 384, 768],
        "features": [16, 48, 96, 64, 192, 384, 8, 24, 48, 32, 96, 192],
        "d": [64, 80, 96],
        "expansion_ratio": 2,
        # "layers": [1, 3, 2]
        "layers": [1, 1, 1]
    },
    "xs":{
        "features": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        "d": [96, 120, 144],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
    "s":{
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
}










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

from timm.models.layers import DropPath, trunc_normal_

class MixBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=2):
        super().__init__()
        dim = dim // 8 * 3 // 2
        #        pdb.set_trace()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.dim = dim
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim * 8 * 2, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim * 8 * 2, 1)
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



class MobileViT(nn.Module):
    def __init__(self, img_size, features_list, d_list, transformer_depth, expansion, num_classes = 1000):
        super(MobileViT, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels = 3, out_channels = features_list[0], kernel_size = 3, stride = 4, padding = 1),
            # nn.AvgPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2))
            # nn.Conv3d(in_channels = features_list[1], out_channels = features_list[2], kernel_size = 3, stride = 2, padding = 1),
            # InvertedResidual(in_channels = features_list[0], out_channels = features_list[1], stride = 1, expand_ratio = expansion),
        )
        self.stemCP = nn.Sequential(
            nn.Conv3d(in_channels = 3, out_channels = features_list[6], kernel_size = 3, stride = 4, padding = 1),
        )

        self.stage1 = nn.Sequential(
            # GhostBottleneck(features_list[1], features_list[2]//2, features_list[2], 3, 2, se_ratio=0),
            # InvertedResidual(in_channels = features_list[1], out_channels = features_list[2], stride = 2, expand_ratio = expansion),
            # InvertedResidual(in_channels = features_list[2], out_channels = features_list[2], stride = 1, expand_ratio = expansion),
            # InvertedResidual(in_channels = features_list[2], out_channels = features_list[3], stride = 1, expand_ratio = expansion)
            MobileVitBlock(in_channels=features_list[0], out_channels=features_list[0], d_model=d_list[0] // 4,
                           layers=transformer_depth[0], mlp_dim=d_list[0] // 4, stride=1),
            MobileVitBlockD(in_channels=features_list[0], out_channels=features_list[1], d_model=d_list[0] // 4,
                           layers=transformer_depth[0], mlp_dim=d_list[0] // 4, stride=2),
        )
        self.stage1CP = nn.Sequential(
            MobileVitBlock(in_channels=features_list[6], out_channels=features_list[6], d_model=d_list[0] // 4,
                           layers=transformer_depth[0], mlp_dim=d_list[0] // 4, stride=1),
            MobileVitBlockD(in_channels=features_list[6], out_channels=features_list[7], d_model=d_list[0] // 4,
                           layers=transformer_depth[0], mlp_dim=d_list[0] // 4, stride=2),
        )

        self.stage2 = nn.Sequential(
            # GhostBottleneck(features_list[3], features_list[4]//2, features_list[4], 3, 2, se_ratio=0),
            MobileVitBlock(in_channels = features_list[1], out_channels = features_list[1], d_model = d_list[0] // 2,
                           layers = transformer_depth[0], mlp_dim = d_list[0] // 2, stride=1),
            MobileVitBlockD(in_channels = features_list[1], out_channels = features_list[2], d_model = d_list[0] // 2,
                           layers = transformer_depth[0], mlp_dim = d_list[0] // 2, stride=2),
        )
        self.stage2CP = nn.Sequential(
            # GhostBottleneck(features_list[3], features_list[4]//2, features_list[4], 3, 2, se_ratio=0),
            MobileVitBlock(in_channels = features_list[7], out_channels = features_list[7], d_model = d_list[0] // 2,
                           layers = transformer_depth[0], mlp_dim = d_list[0] // 2, stride=1),
            MobileVitBlockD(in_channels = features_list[7], out_channels = features_list[8], d_model = d_list[0] // 2,
                           layers = transformer_depth[0], mlp_dim = d_list[0] // 2, stride=2),
        )

        self.stage3 = nn.Sequential(
            # GhostBottleneck(features_list[5], features_list[6]//2, features_list[6], 3, 2, se_ratio=0),
            # InvertedResidual(in_channels = features_list[5], out_channels = features_list[6], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[2], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[3], out_channels = features_list[3], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlockD(in_channels = features_list[3], out_channels = features_list[4], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=2)
        )

        self.stage3CP = nn.Sequential(
            # GhostBottleneck(features_list[5], features_list[6]//2, features_list[6], 3, 2, se_ratio=0),
            # InvertedResidual(in_channels = features_list[5], out_channels = features_list[6], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[8], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[9], out_channels = features_list[9], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=1),
            MobileVitBlockD(in_channels = features_list[9], out_channels = features_list[10], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 1, stride=2)
        )

        self.stage4 = nn.Sequential(
            # GhostBottleneck(features_list[7], features_list[8]//2, features_list[8], 3, 2, se_ratio=0),
            # InvertedResidual(in_channels = features_list[7], out_channels = features_list[8], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[4], out_channels = features_list[4], d_model = d_list[2],
                           layers = transformer_depth[2], mlp_dim = d_list[2] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[4], out_channels = features_list[5], d_model = d_list[2],
                           layers = transformer_depth[2], mlp_dim = d_list[2] * 1, stride=1),
            # nn.Conv3d(in_channels = features_list[9], out_channels = features_list[10], kernel_size = 1, stride = 1, padding = 0)
        )
        self.stage4CP = nn.Sequential(
            MobileVitBlock(in_channels = features_list[10], out_channels = features_list[10], d_model = d_list[2],
                           layers = transformer_depth[2], mlp_dim = d_list[2] * 1, stride=1),
            MobileVitBlock(in_channels = features_list[10], out_channels = features_list[11], d_model = d_list[2],
                           layers = transformer_depth[2], mlp_dim = d_list[2] * 1, stride=1),
            # nn.Conv3d(in_channels = features_list[9], out_channels = features_list[10], kernel_size = 1, stride = 1, padding = 0)
        )

        self.avgpool = nn.AvgPool3d(kernel_size = (16 // 16, img_size // 32, img_size // 32))
        self.fc1 = nn.Linear(features_list[5] * 2, num_classes)
        self.fc = nn.Linear(features_list[5], num_classes)


    def forward(self, x, PET=None, CT=None):
        # Stem
        if PET !=None:
            PET = PET
            CT = CT
        else:
            PET = x
            CT = x
        x = self.stem(x)
        PET = self.stemCP(PET)
        CT = self.stemCP(CT)
        # Body
        x = self.stage1(x)
        PET = self.stage1CP(PET)
        CT = self.stage1CP(CT)
        x = self.stage2(x)
        PET = self.stage2CP(PET)
        CT = self.stage2CP(CT)
        x = self.stage3(x)
        PET = self.stage3CP(PET)
        CT = self.stage3CP(CT)
        x = self.stage4(x)
        PET = self.stage4CP(PET)
        CT = self.stage4CP(CT)

        # Body
        # Head
        # # print("AAAA", x.shape)
        # x = torch.cat([x, PET, CT], dim=1)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # # x = self.fc(x)
        # x = self.fc1(x)

        # x = torch.cat([x, x], dim=1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.fc1(x)
        return x


def MobileViT_XXS(img_size = 224, num_classes = 1000):
    cfg_xxs = model_cfg["xxs"]
    model_xxs = MobileViT(img_size, cfg_xxs["features"], cfg_xxs["d"], cfg_xxs["layers"], cfg_xxs["expansion_ratio"], num_classes)
    return model_xxs

def MobileViT_XS(img_size = 224, num_classes = 1000):
    cfg_xs = model_cfg["xs"]
    model_xs = MobileViT(img_size, cfg_xs["features"], cfg_xs["d"], cfg_xs["layers"], cfg_xs["expansion_ratio"], num_classes)
    return model_xs

def MobileViT_S(img_size = 224, num_classes = 1000):
    cfg_s = model_cfg["s"]
    model_s = MobileViT(img_size, cfg_s["features"], cfg_s["d"], cfg_s["layers"], cfg_s["expansion_ratio"], num_classes)
    return model_s


if __name__ == "__main__":
    img = torch.randn(1, 3, 256, 256)

    cfg_xxs = model_cfg["xxs"]
    model_xxs = MobileViT(256, cfg_xxs["features"], cfg_xxs["d"], cfg_xxs["layers"], cfg_xxs["expansion_ratio"])

    cfg_xs = model_cfg["xs"]
    model_xs = MobileViT(256, cfg_xs["features"], cfg_xs["d"], cfg_xs["layers"], cfg_xs["expansion_ratio"])

    cfg_s = model_cfg["s"]
    model_s = MobileViT(256, cfg_s["features"], cfg_s["d"], cfg_s["layers"], cfg_s["expansion_ratio"])

    print(model_s)

    # XXS: 1.3M 、 XS: 2.3M 、 S: 5.6M
    print("XXS params: ", sum(p.numel() for p in model_xxs.parameters()))
    print(" XS params: ", sum(p.numel() for p in model_xs.parameters()))
    print("  S params: ", sum(p.numel() for p in model_s.parameters()))

