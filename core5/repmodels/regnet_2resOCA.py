"""RegNet

Paper: `Designing Network Design Spaces` - https://arxiv.org/abs/2003.13678
Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py

Based on original PyTorch impl linked above, but re-wrote to use my own blocks (adapted from ResNet here)
and cleaned up with more descriptive variable names.

Weights from original impl have been modified
* first layer from BGR -> RGB as most PyTorch models are
* removed training specific dict entries from checkpoints and keep model state_dict only
* remap names to match the ones here

Hacked together by / Copyright 2020 Ross Wightman
"""
import numpy as np
import torch.nn as nn
import torch

from models_32.model_YXY_fullgood_13.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, AvgPool2dSame, ConvBnAct, SEModule
from .layers import DropPathAA
from .registry import register_modelAA

from timm.models.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model
from .natten import NeighborhoodAttention

from core5.convnet_utils import conv_bn, conv_bn_relu


def _mcfg(**kwargs):
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32)
    cfg.update(**kwargs)
    return cfg


# Model FLOPS = three trailing digits * 10^8
model_cfgs = dict(
    regnetx_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13),
    regnetx_004=_mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22),
    regnetx_006=_mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16),
    regnetx_008=_mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16),
    regnetx_016=_mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    regnetx_032=_mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25),
    regnetx_040=_mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23),
    regnetx_064=_mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17),
    regnetx_080=_mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23),
    regnetx_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19),
    regnetx_160=_mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22),
    regnetx_320=_mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23),
    regnety_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    regnety_004=_mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25),
    regnety_006=_mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    regnety_008=_mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),
    regnety_016=_mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    regnety_032=_mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    regnety_040=_mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    regnety_064=_mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    regnety_080=_mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),
    regnety_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    regnety_160=_mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    regnety_320=_mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25),
)


def _cfg(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
    }


default_cfgs = dict(
    regnetx_002=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_002-e7e85e5c.pth'),
    regnetx_004=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_004-7d0e9424.pth'),
    regnetx_006=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_006-85ec1baa.pth'),
    regnetx_008=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_008-d8b470eb.pth'),
    regnetx_016=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_016-65ca972a.pth'),
    regnetx_032=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_032-ed0c7f7e.pth'),
    regnetx_040=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_040-73c2a654.pth'),
    regnetx_064=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_064-29278baa.pth'),
    regnetx_080=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_080-7c7fcab1.pth'),
    regnetx_120=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_120-65d5521e.pth'),
    regnetx_160=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_160-c98c4112.pth'),
    regnetx_320=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_320-8ea38b93.pth'),
    regnety_002=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_002-e68ca334.pth'),
    regnety_004=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_004-0db870e6.pth'),
    regnety_006=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_006-c67e57ec.pth'),
    regnety_008=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_008-dc900dbe.pth'),
    regnety_016=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_016-54367f74.pth'),
    regnety_032=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/regnety_032_ra-7f2439f9.pth'),
    regnety_040=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_040-f0d569f9.pth'),
    regnety_064=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_064-0a48325c.pth'),
    regnety_080=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_080-e7f3eb93.pth'),
    regnety_120=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_120-721ba79a.pth'),
    regnety_160=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_160-d64013cd.pth'),
    regnety_320=_cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_320-ba464b29.pth'),
)



# model_urls = {
#     "nat_mini_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_mini.pth",
#     "nat_tiny_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_tiny.pth",
#     "nat_small_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_small.pth",
#     "nat_base_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_base.pth",
# }

class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # self.proj = nn.Sequential(
        #     nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        #     nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        # )
        self.proj = nn.Sequential(
            # nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            # nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
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


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x
        return self.downsample(x)

def channel_shuffleYXY(x, groups):
    batchsize, height, width, num_channels = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batchsize, height, width, -1)
    return x

class NAT(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, depths, num_heads, drop_path_rate=0.2, in_chans=3, kernel_size=7,
                 num_classes=2, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, layer_scale=None, **kwargs):
        super(NAT, self).__init__()
        # super().__init__()

        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.normLN = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        self.level2ress = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(dim=int(embed_dim * 2 ** i),
                             depth=depths[i],
                             num_heads=num_heads[i],
                             kernel_size=kernel_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             norm_layer=norm_layer,
                             downsample=(i < self.num_levels - 1),
                             layer_scale=layer_scale)
            self.levels.append(level)

        # 这个代码写的属于无效
        for i in range(self.num_levels):
            level2res = NATBlock(dim=int(embed_dim / 4),
                                 depth=1,
                                 num_heads=num_heads[i],
                                 kernel_size=kernel_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                 norm_layer=norm_layer,
                                 downsample=False,
                                 layer_scale=layer_scale)
            self.level2ress.append(level2res)

        self.num_levels = depths[0]
        self.level2res_O = NATBlock(dim=int(embed_dim / 4),
                                    depth=1,
                                    num_heads=num_heads[i],
                                    kernel_size=kernel_size,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                    norm_layer=norm_layer,
                                    downsample=False,
                                    layer_scale=layer_scale)

        self.num_features = 1024
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)


        x = channel_shuffleYXY(x, 2)
        x_A, x_B, x_C, x_D =  x.split([int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1], dim=3)
        x_B = self.level2res_O(x_A)
        x_C = self.level2res_O(x_B + x_C)
        x_D = self.level2res_O(x_C + x_D)
        shortcut = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        # shortcut_A = x + shortcut
        shortcut_A = shortcut
        shortcut_A = self.normLN(shortcut_A)

        shortcut_B = channel_shuffleYXY(shortcut_A, 2)
        x_A, x_B, x_C, x_D = shortcut_B.split([int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1], dim=3)
        x_B = self.level2res_O(x_A)
        x_C = self.level2res_O(x_B + x_C)
        x_D = self.level2res_O(x_C + x_D)
        shortcut_B = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        # shortcut_B = shortcut_A + shortcut_B
        shortcut_B = self.normLN(shortcut_B)

        shortcut_C = channel_shuffleYXY(shortcut_B, 2)
        x_A, x_B, x_C, x_D = shortcut_C.split([int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1], dim=3)
        x_B = self.level2res_O(x_A)
        x_C = self.level2res_O(x_B + x_C)
        x_D = self.level2res_O(x_C + x_D)
        shortcut_C = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        # shortcut_C = shortcut_B + shortcut_C
        shortcut_C = self.normLN(shortcut_C)

        shortcut_D = channel_shuffleYXY(shortcut_C, 2)
        x_A, x_B, x_C, x_D = shortcut_D.split([int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1], dim=3)
        x_B = self.level2res_O(x_A)
        x_C = self.level2res_O(x_B + x_C)
        x_D = self.level2res_O(x_C + x_D)
        shortcut_D = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        # shortcut_D = shortcut_C + shortcut_D
        shortcut_D = self.normLN(shortcut_D)

        x = torch.cat((shortcut_A, shortcut_B, shortcut_C, shortcut_D), dim=3)


        # print("x", x.shape)     # ([2, 2, 2, 512])
        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
    return widths, num_stages, max_stage, widths_cont


class Bottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, bottleneck_ratio=1, group_width=1, se_ratio=0.25,
                 downsample=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()
        bottleneck_chs = int(round(out_chs * bottleneck_ratio))
        groups = bottleneck_chs // group_width

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        self.conv2 = ConvBnAct(
            bottleneck_chs, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation,
            groups=groups, **cargs)
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, reduction_channels=se_channels)
        else:
            self.se = None
        cargs['act_layer'] = None
        self.conv3 = ConvBnAct(bottleneck_chs, out_chs, kernel_size=1, **cargs)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

        self.convA = conv_bn_relu(in_chs, bottleneck_chs, kernel_size=1)
        self.convB = conv_bn_relu(bottleneck_chs, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation, groups=groups, padding=1)
        self.convC = conv_bn(bottleneck_chs, out_chs, kernel_size=1)


    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)
        # nn.init.zeros_(self.convC.bn.weight)

    def forward(self, x):
        shortcut = x
        # x = self.conv1(x)
        # x = self.conv2(x)
        x = self.convA(x)
        x = self.convB(x)
        if self.se is not None:
            x = self.se(x)
        # x = self.conv3(x)
        x = self.convC(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        return x


def downsample_conv(
        in_chs, out_chs, kernel_size, stride=1, dilation=1, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    # # return ConvBnAct(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, norm_layer=norm_layer, act_layer=None)
    # if kernel_size != 1:
    #     print("**************************这里的卷积核不应该是其它的，而应该是一个1X1  **************************")
    return conv_bn(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation)


def downsample_avg(
        in_chs, out_chs, kernel_size, stride=1, dilation=1, norm_layer=None):
    """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    pool = nn.Identity()
    if stride > 1 or dilation > 1:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    # return nn.Sequential(*[pool, ConvBnAct(in_chs, out_chs, 1, stride=1, norm_layer=norm_layer, act_layer=None)])
    return nn.Sequential(*[pool, conv_bn(in_chs, out_chs, kernel_size=1, stride=1)])


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio, group_width,
                 block_fn=Bottleneck, se_ratio=0., drop_path_rates=None, drop_block=None):
        super(RegStage, self).__init__()
        block_kwargs = {}  # FIXME setup to pass various aa, norm, act layer common args
        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = first_dilation if i == 0 else dilation
            if drop_path_rates is not None and drop_path_rates[i] > 0.:
                drop_path = DropPathAA(drop_path_rates[i])
            else:
                drop_path = None
            if (block_in_chs != out_chs) or (block_stride != 1):
                proj_block = downsample_conv(block_in_chs, out_chs, 1, block_stride, block_dilation)
            else:
                proj_block = None

            name = "b{}".format(i + 1)
            self.add_module(
                name, block_fn(
                    block_in_chs, out_chs, block_stride, block_dilation, bottle_ratio, group_width, se_ratio,
                    downsample=proj_block, drop_block=drop_block, drop_path=drop_path, **block_kwargs)
            )


    def forward(self, x):
        for block in self.children():
            x = block(x)
            # print("x", x.shape)
        return x


class RegNet(nn.Module):
    """RegNet model.

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, output_stride=32, global_pool='avg', drop_rate=0.,
                 drop_path_rate=0., zero_init_last_bn=True):
        super().__init__()
        # TODO add drop block, drop path, anti-aliasing, custom bn/act args
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)

        # Construct the stem
        stem_width = cfg['stem_width']
        # self.stem = ConvBnAct(in_chans, stem_width, 3, stride=2)
        self.stem = conv_bn_relu(in_chans, stem_width, kernel_size=3, stride=2, padding=1)
        self.feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]

        # Construct the stages
        prev_width = stem_width
        curr_stride = 2
        stage_params = self._get_stage_params(cfg, output_stride=output_stride, drop_path_rate=drop_path_rate)
        se_ratio = cfg['se_ratio']
        for i, stage_args in enumerate(stage_params):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(prev_width, **stage_args, se_ratio=se_ratio))
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]


        block_in_chs = 432
        self.downsample_avg = downsample_avg(in_chs=432, out_chs=block_in_chs, kernel_size=1)
        self.NAT = NAT(depths=[1], num_heads=[16], embed_dim=256, mlp_ratio=3, drop_path_rate=0.2, in_chans=block_in_chs, kernel_size=7, num_classes=num_classes)


        # Construct the head
        self.num_features = prev_width
        self.head = ClassifierHead(
            in_chs=prev_width, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _get_stage_params(self, cfg, default_stride=2, output_stride=32, drop_path_rate=0.):
        # Generate RegNet ws per block
        w_a, w_0, w_m, d = cfg['wa'], cfg['w0'], cfg['wm'], cfg['depth']
        widths, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)

        # Convert to per stage format
        stage_widths, stage_depths = np.unique(widths, return_counts=True)

        # Use the same group width, bottleneck mult and stride for each stage
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_bottle_ratios = [cfg['bottle_ratio'] for _ in range(num_stages)]
        stage_strides = []
        stage_dilations = []
        net_stride = 2
        dilation = 1
        for _ in range(num_stages):
            if net_stride >= output_stride:
                dilation *= default_stride
                stride = 1
            else:
                stride = default_stride
                net_stride *= stride
            stage_strides.append(stride)
            stage_dilations.append(dilation)
        stage_dpr = np.split(np.linspace(0, drop_path_rate, d), np.cumsum(stage_depths[:-1]))

        # Adjust the compatibility of ws and gws
        stage_widths, stage_groups = adjust_widths_groups_comp(stage_widths, stage_bottle_ratios, stage_groups)
        param_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_width', 'drop_path_rates']
        stage_params = [
            dict(zip(param_names, params)) for params in
            zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_bottle_ratios, stage_groups,
                stage_dpr)]
        return stage_params

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        for block in list(self.children())[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        # for block in self.children():
        #     x = block(x)
        for block in list(self.children())[:-4]:
            x = block(x)
        #     ([16, 432, 14, 14])
        x = self.downsample_avg(x)
        x = self.NAT(x)

        return x


def _create_regnet(variant, pretrained, **kwargs):
    return build_model_with_cfg(
        RegNet, variant, pretrained, default_cfg=default_cfgs[variant], model_cfg=model_cfgs[variant], **kwargs)


@register_modelAA
def regnetx_002(pretrained=False, **kwargs):
    """RegNetX-200MF"""
    return _create_regnet('regnetx_002', pretrained, **kwargs)


@register_modelAA
def regnetx_004(pretrained=False, **kwargs):
    """RegNetX-400MF"""
    return _create_regnet('regnetx_004', pretrained, **kwargs)


@register_modelAA
def regnetx_006(pretrained=False, **kwargs):
    """RegNetX-600MF"""
    return _create_regnet('regnetx_006', pretrained, **kwargs)


@register_modelAA
def regnetx_008(pretrained=False, **kwargs):
    """RegNetX-800MF"""
    return _create_regnet('regnetx_008', pretrained, **kwargs)


@register_modelAA
def regnetx_016(pretrained=False, **kwargs):
    """RegNetX-1.6GF"""
    return _create_regnet('regnetx_016', pretrained, **kwargs)


@register_modelAA
def regnetx_032(pretrained=False, **kwargs):
    """RegNetX-3.2GF"""
    return _create_regnet('regnetx_032', pretrained, **kwargs)


@register_modelAA
def regnetx_040(pretrained=False, **kwargs):
    """RegNetX-4.0GF"""
    return _create_regnet('regnetx_040', pretrained, **kwargs)


@register_modelAA
def regnetx_064(pretrained=False, **kwargs):
    """RegNetX-6.4GF"""
    return _create_regnet('regnetx_064', pretrained, **kwargs)


@register_modelAA
def regnetx_080(pretrained=False, **kwargs):
    """RegNetX-8.0GF"""
    return _create_regnet('regnetx_080', pretrained, **kwargs)


@register_modelAA
def regnetx_120(pretrained=False, **kwargs):
    """RegNetX-12GF"""
    return _create_regnet('regnetx_120', pretrained, **kwargs)


@register_modelAA
def regnetx_160(pretrained=False, **kwargs):
    """RegNetX-16GF"""
    return _create_regnet('regnetx_160', pretrained, **kwargs)


@register_modelAA
def regnetx_320(pretrained=False, **kwargs):
    """RegNetX-32GF"""
    return _create_regnet('regnetx_320', pretrained, **kwargs)


@register_modelAA
def regnety_002(pretrained=False, **kwargs):
    """RegNetY-200MF"""
    return _create_regnet('regnety_002', pretrained, **kwargs)


@register_modelAA
def regnety_004(pretrained=False, **kwargs):
    """RegNetY-400MF"""
    return _create_regnet('regnety_004', pretrained, **kwargs)


@register_modelAA
def regnety_006(pretrained=False, **kwargs):
    """RegNetY-600MF"""
    return _create_regnet('regnety_006', pretrained, **kwargs)


@register_modelAA
def regnety_008(pretrained=False, **kwargs):
    """RegNetY-800MF"""
    return _create_regnet('regnety_008', pretrained, **kwargs)


@register_modelAA
def regnety_016(pretrained=False, **kwargs):
    """RegNetY-1.6GF"""
    return _create_regnet('regnety_016', pretrained, **kwargs)


@register_modelAA
def regnety_032(pretrained=False, **kwargs):
    """RegNetY-3.2GF"""
    return _create_regnet('regnety_032', pretrained, **kwargs)


@register_modelAA
def regnety_040(pretrained=False, **kwargs):
    """RegNetY-4.0GF"""
    return _create_regnet('regnety_040', pretrained, **kwargs)


@register_modelAA
def regnety_064(pretrained=False, **kwargs):
    """RegNetY-6.4GF"""
    return _create_regnet('regnety_064', pretrained, **kwargs)


@register_modelAA
def regnety_080(pretrained=False, **kwargs):
    """RegNetY-8.0GF"""
    return _create_regnet('regnety_080', pretrained, **kwargs)


@register_modelAA
def regnety_120(pretrained=False, **kwargs):
    """RegNetY-12GF"""
    return _create_regnet('regnety_120', pretrained, **kwargs)


@register_modelAA
def regnety_160(pretrained=False, **kwargs):
    """RegNetY-16GF"""
    return _create_regnet('regnety_160', pretrained, **kwargs)


@register_modelAA
def regnety_320(pretrained=False, **kwargs):
    """RegNetY-32GF"""
    return _create_regnet('regnety_320', pretrained, **kwargs)
