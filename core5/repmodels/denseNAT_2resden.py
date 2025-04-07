"""添加了senet模块,去除transition的senet模块,仅仅在denseblock中添加senet模块"""
import sys
sys.path.append("F:/car_classify_abnormal")
# 这个和full_in loop ,几乎一样，在171行FUll_loop注释了一行
#  # Add a SELayer behind each transition block这行也注释了   206行

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from core.se_module import SELayer
from core5.convnet_utils import conv_bn, conv_bn_relu
from pandas import DataFrame

# Total params: 9,827,908
# Trainable params: 9,827,908
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 375.09
# Params size (MB): 37.49
# Estimated Total Size (MB): 413.15


__all__ = ['SEDenseNet', 'se_densenet121', 'se_densenet169', 'se_densenet201', 'se_densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def se_densenet121(pretrained=False, is_strict=False, num_classes=2, drop_rate=0.2, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes, drop_rate=drop_rate,
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        print('模型从本地导入成功')
        model.load_state_dict(state_dict, strict=is_strict)
    return model


def se_densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        print('模型从本地导入成功')
        model.load_state_dict(state_dict, strict=False)
    return model


def se_densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        print('模型从本地导入成功')
        model.load_state_dict(state_dict, strict=False)
    return model


def se_densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEDenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        print('模型从本地导入成功')
        model.load_state_dict(state_dict, strict=False)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.SElayer = SELayer(channel=num_input_features)

        self.convA = conv_bn_relu(num_input_features, bn_size * growth_rate, kernel_size=1)
        self.convB = conv_bn_relu(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        # new_features = self.SElayer(x)
        # new_features = self.bn1(x)
        # new_features = self.relu1(new_features)
        # new_features = self.conv1(new_features)
        # new_features = self.bn3(new_features)
        # new_features = self.relu3(new_features)
        # new_features = self.conv3(new_features)

        new_features = self.convA(x)
        new_features = self.convB(new_features)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

'''
@author: yali
@time:2021-08-01  16:09

'''
# (6, 12, 24, 16)
class _DenseBlockA(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlockA, self).__init__()
        self.DenseLayerA = _DenseLayer(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerB = _DenseLayer(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerC = _DenseLayer(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerD = _DenseLayer(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerE = _DenseLayer(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerF = _DenseLayer(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)

    def forward(self, x):
        xA = self.DenseLayerA(x)
        xB = self.DenseLayerB(xA)
        xC = self.DenseLayerC(xB)
        xD = self.DenseLayerD(xC)
        xE = self.DenseLayerE(xD)
        xF = self.DenseLayerF(xE)
        # x = torch.cat([x, xA, xB, xC, xD, xE, xF], 1)
        del x, xA, xB, xC, xD, xE
        torch.cuda.empty_cache()
        return xF


class _DenseBlockB(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlockB, self).__init__()
        self.DenseLayerA = _DenseLayer(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerB = _DenseLayer(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerC = _DenseLayer(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerD = _DenseLayer(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerE = _DenseLayer(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerF = _DenseLayer(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerG = _DenseLayer(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerH = _DenseLayer(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerI = _DenseLayer(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerJ = _DenseLayer(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerK = _DenseLayer(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerL = _DenseLayer(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)


    def forward(self, x):
        xA = self.DenseLayerA(x)
        xB = self.DenseLayerB(xA)
        xC = self.DenseLayerC(xB)
        xD = self.DenseLayerD(xC)
        xE = self.DenseLayerE(xD)
        xF = self.DenseLayerF(xE)
        xG = self.DenseLayerG(xF)
        xH = self.DenseLayerH(xG)
        xI = self.DenseLayerI(xH)
        xJ = self.DenseLayerJ(xI)
        xK = self.DenseLayerK(xJ)
        xL = self.DenseLayerL(xK)

        # x = torch.cat([x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL], 1)
        del x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK
        torch.cuda.empty_cache()
        return xL



class _DenseBlockC(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlockC, self).__init__()
        self.DenseLayerA = _DenseLayer(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerB = _DenseLayer(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerC = _DenseLayer(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerD = _DenseLayer(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerE = _DenseLayer(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerF = _DenseLayer(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerG = _DenseLayer(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerH = _DenseLayer(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerI = _DenseLayer(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerJ = _DenseLayer(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerK = _DenseLayer(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerL = _DenseLayer(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerM = _DenseLayer(num_input_features + 12 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerN = _DenseLayer(num_input_features + 13 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerO = _DenseLayer(num_input_features + 14 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerP = _DenseLayer(num_input_features + 15 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerQ = _DenseLayer(num_input_features + 16 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerR = _DenseLayer(num_input_features + 17 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerS = _DenseLayer(num_input_features + 18 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerT = _DenseLayer(num_input_features + 19 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerU = _DenseLayer(num_input_features + 20 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerV = _DenseLayer(num_input_features + 21 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerW = _DenseLayer(num_input_features + 22 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerX = _DenseLayer(num_input_features + 23 * growth_rate, growth_rate, bn_size, drop_rate)

    def forward(self, x):
        xA = self.DenseLayerA(x)
        xB = self.DenseLayerB(xA)
        xC = self.DenseLayerC(xB)
        xD = self.DenseLayerD(xC)
        xE = self.DenseLayerE(xD)
        xF = self.DenseLayerF(xE)
        xG = self.DenseLayerG(xF)
        xH = self.DenseLayerH(xG)
        xI = self.DenseLayerI(xH)
        xJ = self.DenseLayerJ(xI)
        xK = self.DenseLayerK(xJ)
        xL = self.DenseLayerL(xK)
        xM = self.DenseLayerM(xL)
        xN = self.DenseLayerN(xM)
        xO = self.DenseLayerO(xN)
        xP = self.DenseLayerP(xO)
        xQ = self.DenseLayerQ(xP)
        xR = self.DenseLayerR(xQ)
        xS = self.DenseLayerS(xR)
        xT = self.DenseLayerT(xS)
        xU = self.DenseLayerU(xT)
        xV = self.DenseLayerV(xU)
        xW = self.DenseLayerW(xV)
        xX = self.DenseLayerX(xW)

        # x = torch.cat([x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL, xM, xN, xO, xP, xQ, xR, xS, xT, xU, xV, xW, xX], 1)
        del x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL, xM, xN, xO, xP, xQ, xR, xS, xT, xU, xV, xW
        # torch.cuda.empty_cache()()
        return xX

class _DenseBlockD(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlockD, self).__init__()
        self.DenseLayerA = _DenseLayer(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerB = _DenseLayer(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerC = _DenseLayer(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerD = _DenseLayer(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerE = _DenseLayer(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerF = _DenseLayer(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerG = _DenseLayer(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerH = _DenseLayer(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerI = _DenseLayer(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerJ = _DenseLayer(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerK = _DenseLayer(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerL = _DenseLayer(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerM = _DenseLayer(num_input_features + 12 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerN = _DenseLayer(num_input_features + 13 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerO = _DenseLayer(num_input_features + 14 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerP = _DenseLayer(num_input_features + 15 * growth_rate, growth_rate, bn_size, drop_rate)


    def forward(self, x):
        xA = self.DenseLayerA(x)
        xB = self.DenseLayerB(xA)
        xC = self.DenseLayerC(xB)
        xD = self.DenseLayerD(xC)
        xE = self.DenseLayerE(xD)
        xF = self.DenseLayerF(xE)
        xG = self.DenseLayerG(xF)
        xH = self.DenseLayerH(xG)
        xI = self.DenseLayerI(xH)
        xJ = self.DenseLayerJ(xI)
        xK = self.DenseLayerK(xJ)
        xL = self.DenseLayerL(xK)
        xM = self.DenseLayerM(xL)
        xN = self.DenseLayerN(xM)
        xO = self.DenseLayerO(xN)
        xP = self.DenseLayerP(xO)

        # x = torch.cat([x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL, xM, xN, xO, xP], 1)
        del x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL, xM, xN, xO
        torch.cuda.empty_cache()
        return xP





from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from .natten import NeighborhoodAttention
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
        self.level2res_OA = NATBlock(dim=int(embed_dim / 2),
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

        self.num_features = 512
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
        # print("Ax", x.shape)     # Ax torch.Size([1, 512, 7, 7])
        x = self.patch_embed(x)
        # print("Bx", x.shape)     # Bx torch.Size([1, 2, 2, 512])
        x = self.pos_drop(x)

        # for level in self.levels:
        #     x = level(x)

        # for i in range(self.num_levels):
        #     x_A, x_B, x_C, x_D =  x.split([int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1], dim=3)
        #     x_A = self.level2res_O(x_A)
        #     x_B = self.level2res_O(x_A + x_B)
        #     x_C = self.level2res_O(x_B + x_C)
        #     x_D = self.level2res_O(x_C + x_D)
        #     # x = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        #     shortcut = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        #     # exec ("shortcut%s=DataFrame(k[%s])" % (i, torch.cat((x_A, x_B, x_C, x_D), dim=3)))
        #     # exec ("shortcut%s=%i") = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        #     x = x + shortcut


        x_A, x_B, x_C, x_D =  x.split([int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1, int(x.shape[3]/4) * 1], dim=3)
        x_A = self.level2res_O(x_A)
        x_B = self.level2res_O(x_A + x_B)
        x_C = self.level2res_O(x_B + x_C)
        x_D = self.level2res_O(x_C + x_D)
        shortcut_A = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        shortcut_A = x + shortcut_A
        shortcut_B = torch.cat((x, shortcut_A), dim=3)

        x_A, x_B, x_C, x_D =  shortcut_B.split([int(x.shape[3]/2) * 1, int(x.shape[3]/2) * 1, int(x.shape[3]/2) * 1, int(x.shape[3]/2) * 1], dim=3)
        x_A = self.level2res_OA(x_A)
        x_B = self.level2res_OA(x_A + x_B)
        x_C = self.level2res_OA(x_B + x_C)
        x_D = self.level2res_OA(x_C + x_D)
        shortcut_C = torch.cat((x_A, x_B, x_C, x_D), dim=3)
        shortcut_C = shortcut_B + shortcut_C
        shortcut_C = torch.cat((shortcut_B, shortcut_C), dim=3)
        x = shortcut_C

        # print("x", x.shape)     # ([2, 2, 2, 512])
        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.SElayer1 = SELayer(num_input_features)
        # self.bn1 = nn.BatchNorm2d(num_input_features)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)

        self.convA = conv_bn_relu(num_input_features, num_output_features, kernel_size=1)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # # x = self.SElayer1(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        # x = self.conv1(x)

        x = self.convA(x)
        x = self.pool1(x)
        return x

class SEDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2):

        super(SEDenseNet, self).__init__()

        # # First convolution
        # self.First = nn.Sequential(OrderedDict([
        # # self.features = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))

        self.First = conv_bn_relu(3, num_init_features, kernel_size=7, stride=2, padding=3)
        self.Pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        # Add SELayer at first convolution
        # self.features.add_module("SELayer_0a", SELayer(channel=num_init_features))

        # Each denseblock
        num_features = num_init_features
        self._DenseBlockA = _DenseBlockA(num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + 6 * growth_rate
        # num_features = 256
        self._TransitionA = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self._DenseBlockB = _DenseBlockB(num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + 12 * growth_rate
        # num_features = 512
        self._TransitionB = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self._DenseBlockC = _DenseBlockC(num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + 24 * growth_rate
        # num_features = 1024
        self._TransitionC = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self._DenseBlockD = _DenseBlockD(num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        # num_features = num_features + 16 * growth_rate
        # # num_features = 1024

        # self.NAT = NAT(depths=[3, 4, 6, 5], num_heads=[2, 4, 8, 16], embed_dim=64, mlp_ratio=3, drop_path_rate=0.2, kernel_size=7)
        # self.NAT = NAT(depths=[5], num_heads=[16], embed_dim=512, mlp_ratio=3, drop_path_rate=0.2, in_chans=num_features, kernel_size=7, num_classes=num_classes)
        # self.NAT = NAT(depths=[5], num_heads=[16], embed_dim=512, mlp_ratio=3, drop_path_rate=0.2, in_chans=num_features, kernel_size=7, num_classes=num_classes)
        self.NAT = NAT(depths=[1], num_heads=[16], embed_dim=128, mlp_ratio=3, drop_path_rate=0.2, in_chans=num_features, kernel_size=7, num_classes=num_classes)

        # depths=2, 29MB            kernel_size=3,  计算量只有与三分之一了  66.162M

        num_features = num_features + 16 * growth_rate
        # Final batch norm
        self.Final_bn = nn.BatchNorm2d(num_features)
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Add SELayer
        # self.features.add_module("SELayer_0b", SELayer(channel=num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)




        # # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # 6. xavier_uniform 初始化
        #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #         # 7. xavier_normal 初始化
        #         # nn.init.xavier_normal_(m.weight)
        #
        #         # nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, -100)

        # Official init from torch repo.
        for m in self.modules():
            # 如果使用SeparableConv2d会出现错误AttributeError: 'SeparableConv2d' object has no attribute 'weight'
            # https://blog.csdn.net/qq_37297763/article/details/116430049       Pytorch对声明的网络结构进行初始化
            if isinstance(m, nn.Conv2d):
            # if isinstance(m, SeparableConv2d):      # 阅读代码以后只是里面有一个weight，但是改起来还是挺复杂的，又报一个错误，这个需要修改挺多初始化参数的
                # 修改SeparableConv2d里面，添加weight后报错误是TypeError: __init__() missing 1 required positional argument: 'weight'
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        First = self.First(x)
        First = self.Pool(First)
        DenseBlockA = self._DenseBlockA(First)
        TransitionA = self._TransitionA(DenseBlockA)
        DenseBlockB = self._DenseBlockB(TransitionA)
        TransitionB = self._TransitionB(DenseBlockB)
        DenseBlockC = self._DenseBlockC(TransitionB)
        TransitionC = self._TransitionC(DenseBlockC)
        # # DenseBlockD = self._DenseBlockD(TransitionC)
        # DenseBlockD = self.Final_bn(DenseBlockD)
        # out = F.relu(DenseBlockD, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(DenseBlockD.size(0), -1)     # 这个是原始代码里的
        # out = self.classifier(out)

        out = self.NAT(TransitionC)
        # print("DenseBlockD", DenseBlockD.shape)     # ([2, 512])
        return out


def test_se_densenet(pretrained=False):
    X = torch.Tensor(32, 3, 224, 224)

    if pretrained:
        model = se_densenet121(pretrained=pretrained)
        net_state_dict = {key: value for key, value in model_zoo.load_url("https://download.pytorch.org/models/densenet121-a639ec97.pth").items()}
        model.load_state_dict(net_state_dict, strict=False)

    else:
        model = se_densenet121(pretrained=pretrained)

    print(model)
    if torch.cuda.is_available():
        X = X.cuda()
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        output = model(X)
        print(output.shape)

if __name__ == "__main__":
    test_se_densenet()