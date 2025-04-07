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


def se_densenet121(pretrained=False, is_strict=False, num_classes=1000, drop_rate=0.2, **kwargs):
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

# # https://blog.csdn.net/weixin_43731225/article/details/114778008 CNN与Transformer的结合-BoTNet
# class MHSA(nn.Module):
#     def __init__(self, n_dims, width, height):
#         super(MHSA, self).__init__()
#
#         self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
#         self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
#         self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
#         # nn.Parameter 含义是将一个固定不可训练的tensor转换成可以训练的类型parameter
#         self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
#         self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         n_batch, C, width, height = x.size()
#         q = self.query(x).view(n_batch, C, -1)
#         k = self.key(x).view(n_batch, C, -1)
#         v = self.value(x).view(n_batch, C, -1)
#         # 对存储在两个批bach1和batch内的矩阵进行批矩阵乘操作。batch1和2都包含相同数量矩阵的三维张量
#         content_content = torch.bmm(q.permute(0, 2, 1), k)
#
#         content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
#         content_position = torch.matmul(content_position, q)
#
#         energy = content_content + content_position
#         attention = self.softmax(energy)
#
#         out = torch.bmm(v, attention.permute(0, 2, 1))
#         out = out.view(n_batch, C, width, height)
#
#         return out

# # https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=2):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.CA = CoordAtt(num_input_features, num_input_features)
        # self.SElayer = SELayer(channel=num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv3 = MHSA(bn_size * growth_rate)
        self.drop_rate = drop_rate

    def forward(self, x):
        # new_features = self.CA(x)
        # new_features = self.bn1(new_features)
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        new_features = self.conv3(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


# class _DenseLayerA(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseLayerA, self).__init__()
#         self.bn1 = nn.BatchNorm2d(num_input_features)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
#         self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv3 = MHSA(bn_size * growth_rate, 56, 56)
#         self.drop_rate = drop_rate


from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=7, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # self.scale = qk_scale or head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), vis=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class _DenseLayerA(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerA, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        # self.trans = MHSA(growth_rate, 7, 7)
        self.conv3 = nn.Conv2d(growth_rate//2, growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.trans = Block(dim=28*28, num_heads=7, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),)
        # self.trans = MHSA(growth_rate, 28, 28)
        self.drop_rate = drop_rate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = nn.ConvTranspose2d(in_channels=growth_rate//2,out_channels=growth_rate//2,kernel_size=3,stride=2,padding=1,output_padding=1),


    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        x_A, x_B = new_features.split([int(new_features.shape[1] / 2) * 1, int(new_features.shape[1] / 2) * 1], dim=1)
        # x_A = self.conv3(x_A)
        x_B = self.pool(new_features)
        B, C, W, H  = x_B.shape
        x_B = x_B.flatten(2)
        x_B = self.trans(x_B).reshape(B, C, W, H)
        new_features = self.up(x_B)
        # new_features = torch.cat([x_A, x_B], 1)
        # new_features = channel_shuffle(new_features, 2)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)



class _DenseLayerB(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerB, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        # self.trans = MHSA(growth_rate, 7, 7)
        self.conv3 = nn.Conv2d(growth_rate//2, growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.trans = Block(dim=14*14, num_heads=7, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),)
        # self.trans = MHSA(growth_rate, 14, 14)
        self.drop_rate = drop_rate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        # x_A, x_B = new_features.split([int(new_features.shape[1] / 2) * 1, int(new_features.shape[1] / 2) * 1], dim=1)
        # x_A = self.conv3(x_A)
        x_B = self.pool(new_features)
        B, C, W, H  = x_B.shape
        x_B = x_B.flatten(2)
        x_B = self.trans(x_B).reshape(B, C, W, H)
        new_features = self.up(x_B)
        # new_features = torch.cat([x_A, x_B], 1)
        # new_features = channel_shuffle(new_features, 2)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseLayerC(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerC, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        # self.trans = MHSA(growth_rate, 7, 7)
        self.conv3 = nn.Conv2d(growth_rate//2, growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.trans = Block(dim=7*7, num_heads=7, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),)
        # self.trans = MHSA(growth_rate, 7, 7)
        self.drop_rate = drop_rate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        # x_A, x_B = new_features.split([int(new_features.shape[1] / 2) * 1, int(new_features.shape[1] / 2) * 1], dim=1)
        # x_A = self.conv3(x_A)
        x_B = self.pool(new_features)
        B, C, W, H  = x_B.shape
        x_B = x_B.flatten(2)
        x_B = self.trans(x_B).reshape(B, C, W, H)
        new_features = self.up(x_B)
        # new_features = torch.cat([x_A, x_B], 1)
        # new_features = channel_shuffle(new_features, 2)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)



# class _DenseLayerD(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseLayerD, self).__init__()
#         self.bn1 = nn.BatchNorm2d(num_input_features)
#         self.relu1 = nn.ReLU(inplace=True)
#         # self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
#         # self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
#         # self.relu3 = nn.ReLU(inplace=True)
#         # self.conv3 = MHSA(bn_size * growth_rate, 7, 7)
#         # self.drop_rate = drop_rate
#
#         self.bn1 = nn.BatchNorm2d(num_input_features)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(growth_rate)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv3 = MHSA(growth_rate, 7, 7)
#         self.drop_rate = drop_rate
#
#     def forward(self, x):
#         new_features = self.bn1(x)
#         new_features = self.relu1(new_features)
#         new_features = self.conv1(new_features)
#         new_features = self.bn3(new_features)
#         new_features = self.relu3(new_features)
#         new_features = self.conv3(new_features)
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
#         return torch.cat([x, new_features], 1)
#
#


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class _DenseLayerD(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerD, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        # self.trans = MHSA(growth_rate, 7, 7)
        self.conv3 = nn.Conv2d(growth_rate//2, growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.trans = Block(dim=7*7, num_heads=7, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),)
        # self.trans = MHSA(growth_rate, 7, 7)
        self.drop_rate = drop_rate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        # x_A, x_B = new_features.split([int(new_features.shape[1] / 2) * 1, int(new_features.shape[1] / 2) * 1], dim=1)
        # x_A = self.conv3(x_A)
        x_B = new_features
        B, C, W, H  = x_B.shape
        x_B = x_B.flatten(2)
        new_features = self.trans(x_B).reshape(B, C, W, H)
        # new_features = torch.cat([x_A, x_B], 1)
        # new_features = channel_shuffle(new_features, 2)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)




'''
@author: yali
@time:2021-08-01  16:09

'''
# # (6, 12, 24, 16)
class _DenseBlockA(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlockA, self).__init__()
        self.DenseLayerA = _DenseLayerA(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerB = _DenseLayerA(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerC = _DenseLayerA(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerD = _DenseLayerA(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerE = _DenseLayerA(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerF = _DenseLayerA(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)

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
        self.DenseLayerA = _DenseLayerB(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerB = _DenseLayerB(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerC = _DenseLayerB(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerD = _DenseLayerB(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerE = _DenseLayerB(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerF = _DenseLayerB(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerG = _DenseLayerB(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerH = _DenseLayerB(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerI = _DenseLayerB(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerJ = _DenseLayerB(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerK = _DenseLayerB(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerL = _DenseLayerB(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)


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
        self.DenseLayerA = _DenseLayerC(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerB = _DenseLayerC(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerC = _DenseLayerC(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerD = _DenseLayerC(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerE = _DenseLayerC(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerF = _DenseLayerC(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerG = _DenseLayerC(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerH = _DenseLayerC(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerI = _DenseLayerC(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerJ = _DenseLayerC(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerK = _DenseLayerC(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerL = _DenseLayerC(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerM = _DenseLayerC(num_input_features + 12 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerN = _DenseLayerC(num_input_features + 13 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerO = _DenseLayerC(num_input_features + 14 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerP = _DenseLayerC(num_input_features + 15 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerQ = _DenseLayerC(num_input_features + 16 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerR = _DenseLayerC(num_input_features + 17 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerS = _DenseLayerC(num_input_features + 18 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerT = _DenseLayerC(num_input_features + 19 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerU = _DenseLayerC(num_input_features + 20 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerV = _DenseLayerC(num_input_features + 21 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerW = _DenseLayerC(num_input_features + 22 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerX = _DenseLayerC(num_input_features + 23 * growth_rate, growth_rate, bn_size, drop_rate)

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
        torch.cuda.empty_cache()
        return xX

class _DenseBlockD(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlockD, self).__init__()
        self.DenseLayerA = _DenseLayerD(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerB = _DenseLayerD(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerC = _DenseLayerD(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerD = _DenseLayerD(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerE = _DenseLayerD(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerF = _DenseLayerD(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerG = _DenseLayerD(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerH = _DenseLayerD(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerI = _DenseLayerD(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerJ = _DenseLayerD(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerK = _DenseLayerD(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerL = _DenseLayerD(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerM = _DenseLayerD(num_input_features + 12 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerN = _DenseLayerD(num_input_features + 13 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerO = _DenseLayerD(num_input_features + 14 * growth_rate, growth_rate, bn_size, drop_rate)
        self.DenseLayerP = _DenseLayerD(num_input_features + 15 * growth_rate, growth_rate, bn_size, drop_rate)


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



#
#
# class _DenseBlockA(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseBlockA, self).__init__()
#         self.DenseLayerA = _DenseLayer(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerB = _DenseLayer(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerC = _DenseLayer(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerD = _DenseLayer(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerE = _DenseLayer(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerF = _DenseLayer(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
#
#     def forward(self, x):
#         xA = self.DenseLayerA(x)
#         xB = self.DenseLayerB(xA)
#         xC = self.DenseLayerC(xB)
#         xD = self.DenseLayerD(xC)
#         xE = self.DenseLayerE(xD)
#         xF = self.DenseLayerF(xE)
#         # x = torch.cat([x, xA, xB, xC, xD, xE, xF], 1)
#         del x, xA, xB, xC, xD, xE
#         torch.cuda.empty_cache()
#         return xF
#
#
# class _DenseBlockB(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseBlockB, self).__init__()
#         self.DenseLayerA = _DenseLayer(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerB = _DenseLayer(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerC = _DenseLayer(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerD = _DenseLayer(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerE = _DenseLayer(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerF = _DenseLayer(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerG = _DenseLayer(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerH = _DenseLayer(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerI = _DenseLayer(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerJ = _DenseLayer(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerK = _DenseLayer(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerL = _DenseLayer(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)
#
#
#     def forward(self, x):
#         xA = self.DenseLayerA(x)
#         xB = self.DenseLayerB(xA)
#         xC = self.DenseLayerC(xB)
#         xD = self.DenseLayerD(xC)
#         xE = self.DenseLayerE(xD)
#         xF = self.DenseLayerF(xE)
#         xG = self.DenseLayerG(xF)
#         xH = self.DenseLayerH(xG)
#         xI = self.DenseLayerI(xH)
#         xJ = self.DenseLayerJ(xI)
#         xK = self.DenseLayerK(xJ)
#         xL = self.DenseLayerL(xK)
#
#         # x = torch.cat([x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL], 1)
#         del x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK
#         torch.cuda.empty_cache()
#         return xL
#
#
#
# class _DenseBlockC(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseBlockC, self).__init__()
#         self.DenseLayerA = _DenseLayer(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerB = _DenseLayer(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerC = _DenseLayer(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerD = _DenseLayer(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerE = _DenseLayer(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerF = _DenseLayer(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerG = _DenseLayer(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerH = _DenseLayer(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerI = _DenseLayer(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerJ = _DenseLayer(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerK = _DenseLayer(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerL = _DenseLayer(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerM = _DenseLayer(num_input_features + 12 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerN = _DenseLayer(num_input_features + 13 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerO = _DenseLayer(num_input_features + 14 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerP = _DenseLayer(num_input_features + 15 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerQ = _DenseLayer(num_input_features + 16 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerR = _DenseLayer(num_input_features + 17 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerS = _DenseLayer(num_input_features + 18 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerT = _DenseLayer(num_input_features + 19 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerU = _DenseLayer(num_input_features + 20 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerV = _DenseLayer(num_input_features + 21 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerW = _DenseLayer(num_input_features + 22 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerX = _DenseLayer(num_input_features + 23 * growth_rate, growth_rate, bn_size, drop_rate)
#
#     def forward(self, x):
#         xA = self.DenseLayerA(x)
#         xB = self.DenseLayerB(xA)
#         xC = self.DenseLayerC(xB)
#         xD = self.DenseLayerD(xC)
#         xE = self.DenseLayerE(xD)
#         xF = self.DenseLayerF(xE)
#         xG = self.DenseLayerG(xF)
#         xH = self.DenseLayerH(xG)
#         xI = self.DenseLayerI(xH)
#         xJ = self.DenseLayerJ(xI)
#         xK = self.DenseLayerK(xJ)
#         xL = self.DenseLayerL(xK)
#         xM = self.DenseLayerM(xL)
#         xN = self.DenseLayerN(xM)
#         xO = self.DenseLayerO(xN)
#         xP = self.DenseLayerP(xO)
#         xQ = self.DenseLayerQ(xP)
#         xR = self.DenseLayerR(xQ)
#         xS = self.DenseLayerS(xR)
#         xT = self.DenseLayerT(xS)
#         xU = self.DenseLayerU(xT)
#         xV = self.DenseLayerV(xU)
#         xW = self.DenseLayerW(xV)
#         xX = self.DenseLayerX(xW)
#
#         # x = torch.cat([x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL, xM, xN, xO, xP, xQ, xR, xS, xT, xU, xV, xW, xX], 1)
#         del x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL, xM, xN, xO, xP, xQ, xR, xS, xT, xU, xV, xW
#         torch.cuda.empty_cache()
#         return xX
#
# class _DenseBlockD(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseBlockD, self).__init__()
#         self.DenseLayerA = _DenseLayer(num_input_features + 0 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerB = _DenseLayer(num_input_features + 1 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerC = _DenseLayer(num_input_features + 2 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerD = _DenseLayer(num_input_features + 3 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerE = _DenseLayer(num_input_features + 4 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerF = _DenseLayer(num_input_features + 5 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerG = _DenseLayer(num_input_features + 6 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerH = _DenseLayer(num_input_features + 7 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerI = _DenseLayer(num_input_features + 8 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerJ = _DenseLayer(num_input_features + 9 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerK = _DenseLayer(num_input_features + 10 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerL = _DenseLayer(num_input_features + 11 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerM = _DenseLayer(num_input_features + 12 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerN = _DenseLayer(num_input_features + 13 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerO = _DenseLayer(num_input_features + 14 * growth_rate, growth_rate, bn_size, drop_rate)
#         self.DenseLayerP = _DenseLayer(num_input_features + 15 * growth_rate, growth_rate, bn_size, drop_rate)
#
#
#     def forward(self, x):
#         xA = self.DenseLayerA(x)
#         xB = self.DenseLayerB(xA)
#         xC = self.DenseLayerC(xB)
#         xD = self.DenseLayerD(xC)
#         xE = self.DenseLayerE(xD)
#         xF = self.DenseLayerF(xE)
#         xG = self.DenseLayerG(xF)
#         xH = self.DenseLayerH(xG)
#         xI = self.DenseLayerI(xH)
#         xJ = self.DenseLayerJ(xI)
#         xK = self.DenseLayerK(xJ)
#         xL = self.DenseLayerL(xK)
#         xM = self.DenseLayerM(xL)
#         xN = self.DenseLayerN(xM)
#         xO = self.DenseLayerO(xN)
#         xP = self.DenseLayerP(xO)
#
#         # x = torch.cat([x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL, xM, xN, xO, xP], 1)
#         del x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL, xM, xN, xO
#         torch.cuda.empty_cache()
#         return xP



import numpy as np
class _Transition2(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition2, self).__init__()
        # self.CA1 = CoordAtt(num_input_features, num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        #
        # if np.size(x, 2) == 56:
        #     # Crop_trans = x[firstbatchsize, 512, 1:-1, 1:-1]      #a 这个操作是切除最外层的一个皮
        #     Crop_trans = x[..., 4:-4, 4:-4].cuda()      #a 这个操作是切除最外层的一个皮      直接切片索引就ok
        #     b, c, h, w = Crop_trans.shape
        #     out = torch.zeros((b, c, h * 7 // 6, w * 7 // 6))
        #     # 第一排的数据，6个值变成7个值
        #     out[..., 0::7, 0::7] = Crop_trans[..., 0::6, 0::6].cuda()
        #     out[..., 0::7, 1::7] = Crop_trans[..., 0::6, 1::6].cuda()
        #     out[..., 0::7, 2::7] = Crop_trans[..., 0::6, 2::6].cuda()
        #     out[..., 0::7, 3::7] = Crop_trans[..., 0::6, 3::6].cuda()
        #     out[..., 0::7, 4::7] = Crop_trans[..., 0::6, 4::6].cuda()
        #     out[..., 0::7, 5::7] = Crop_trans[..., 0::6, 5::6].cuda()
        #     out[..., 0::7, 6::7] = Crop_trans[..., 0::6, 5::6].cuda()
        #
        #     # 第二排的数据
        #     out[..., 1::7, 0::7] = Crop_trans[..., 1::6, 0::6].cuda()
        #     out[..., 1::7, 1::7] = Crop_trans[..., 1::6, 1::6].cuda()
        #     out[..., 1::7, 2::7] = Crop_trans[..., 1::6, 2::6].cuda()
        #     out[..., 1::7, 3::7] = Crop_trans[..., 1::6, 3::6].cuda()
        #     out[..., 1::7, 4::7] = Crop_trans[..., 1::6, 4::6].cuda()
        #     out[..., 1::7, 5::7] = Crop_trans[..., 1::6, 5::6].cuda()
        #     out[..., 1::7, 6::7] = Crop_trans[..., 1::6, 5::6].cuda()
        #
        #     # 第三排的数据
        #     out[..., 2::7, 0::7] = Crop_trans[..., 2::6, 0::6].cuda()
        #     out[..., 2::7, 1::7] = Crop_trans[..., 2::6, 1::6].cuda()
        #     out[..., 2::7, 2::7] = Crop_trans[..., 2::6, 2::6].cuda()
        #     out[..., 2::7, 3::7] = Crop_trans[..., 2::6, 3::6].cuda()
        #     out[..., 2::7, 4::7] = Crop_trans[..., 2::6, 4::6].cuda()
        #     out[..., 2::7, 5::7] = Crop_trans[..., 2::6, 5::6].cuda()
        #     out[..., 2::7, 6::7] = Crop_trans[..., 2::6, 5::6].cuda()
        #
        #     # 第四排的数据
        #     out[..., 3::7, 0::7] = Crop_trans[..., 3::6, 0::6].cuda()
        #     out[..., 3::7, 1::7] = Crop_trans[..., 3::6, 1::6].cuda()
        #     out[..., 3::7, 2::7] = Crop_trans[..., 3::6, 2::6].cuda()
        #     out[..., 3::7, 3::7] = Crop_trans[..., 3::6, 3::6].cuda()
        #     out[..., 3::7, 4::7] = Crop_trans[..., 3::6, 4::6].cuda()
        #     out[..., 3::7, 5::7] = Crop_trans[..., 3::6, 5::6].cuda()
        #     out[..., 3::7, 6::7] = Crop_trans[..., 3::6, 5::6].cuda()
        #
        #     # 第五排的数据
        #     out[..., 4::7, 0::7] = Crop_trans[..., 4::6, 0::6].cuda()
        #     out[..., 4::7, 1::7] = Crop_trans[..., 4::6, 1::6].cuda()
        #     out[..., 4::7, 2::7] = Crop_trans[..., 4::6, 2::6].cuda()
        #     out[..., 4::7, 3::7] = Crop_trans[..., 4::6, 3::6].cuda()
        #     out[..., 4::7, 4::7] = Crop_trans[..., 4::6, 4::6].cuda()
        #     out[..., 4::7, 5::7] = Crop_trans[..., 4::6, 5::6].cuda()
        #     out[..., 4::7, 6::7] = Crop_trans[..., 4::6, 5::6].cuda()
        #
        #     # 第六排的数据
        #     out[..., 5::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
        #     out[..., 5::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
        #     out[..., 5::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
        #     out[..., 5::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
        #     out[..., 5::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
        #     out[..., 5::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #     out[..., 5::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #
        #     # 第七排的数据
        #     out[..., 6::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
        #     out[..., 6::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
        #     out[..., 6::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
        #     out[..., 6::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
        #     out[..., 6::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
        #     out[..., 6::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #     out[..., 6::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #
        #
        # #  下面直接对特征图进行裁切
        # if np.size(x, 2) == 28:
        #     # x = x
        #     Crop_trans = x[..., 2:-2, 2:-2].cuda()      #a 这个操作是切除最外层的一个皮      直接切片索引就ok
        #     b, c, h, w = Crop_trans.shape
        #     out = torch.zeros((b, c, h * 7 // 6, w * 7 // 6))
        #     # 第一排的数据，6个值变成7个值
        #     out[..., 0::7, 0::7] = Crop_trans[..., 0::6, 0::6].cuda()
        #     out[..., 0::7, 1::7] = Crop_trans[..., 0::6, 1::6].cuda()
        #     out[..., 0::7, 2::7] = Crop_trans[..., 0::6, 2::6].cuda()
        #     out[..., 0::7, 3::7] = Crop_trans[..., 0::6, 3::6].cuda()
        #     out[..., 0::7, 4::7] = Crop_trans[..., 0::6, 4::6].cuda()
        #     out[..., 0::7, 5::7] = Crop_trans[..., 0::6, 5::6].cuda()
        #     out[..., 0::7, 6::7] = Crop_trans[..., 0::6, 5::6].cuda()
        #
        #     # 第二排的数据
        #     out[..., 1::7, 0::7] = Crop_trans[..., 1::6, 0::6].cuda()
        #     out[..., 1::7, 1::7] = Crop_trans[..., 1::6, 1::6].cuda()
        #     out[..., 1::7, 2::7] = Crop_trans[..., 1::6, 2::6].cuda()
        #     out[..., 1::7, 3::7] = Crop_trans[..., 1::6, 3::6].cuda()
        #     out[..., 1::7, 4::7] = Crop_trans[..., 1::6, 4::6].cuda()
        #     out[..., 1::7, 5::7] = Crop_trans[..., 1::6, 5::6].cuda()
        #     out[..., 1::7, 6::7] = Crop_trans[..., 1::6, 5::6].cuda()
        #
        #     # 第三排的数据
        #     out[..., 2::7, 0::7] = Crop_trans[..., 2::6, 0::6].cuda()
        #     out[..., 2::7, 1::7] = Crop_trans[..., 2::6, 1::6].cuda()
        #     out[..., 2::7, 2::7] = Crop_trans[..., 2::6, 2::6].cuda()
        #     out[..., 2::7, 3::7] = Crop_trans[..., 2::6, 3::6].cuda()
        #     out[..., 2::7, 4::7] = Crop_trans[..., 2::6, 4::6].cuda()
        #     out[..., 2::7, 5::7] = Crop_trans[..., 2::6, 5::6].cuda()
        #     out[..., 2::7, 6::7] = Crop_trans[..., 2::6, 5::6].cuda()
        #
        #     # 第四排的数据
        #     out[..., 3::7, 0::7] = Crop_trans[..., 3::6, 0::6].cuda()
        #     out[..., 3::7, 1::7] = Crop_trans[..., 3::6, 1::6].cuda()
        #     out[..., 3::7, 2::7] = Crop_trans[..., 3::6, 2::6].cuda()
        #     out[..., 3::7, 3::7] = Crop_trans[..., 3::6, 3::6].cuda()
        #     out[..., 3::7, 4::7] = Crop_trans[..., 3::6, 4::6].cuda()
        #     out[..., 3::7, 5::7] = Crop_trans[..., 3::6, 5::6].cuda()
        #     out[..., 3::7, 6::7] = Crop_trans[..., 3::6, 5::6].cuda()
        #
        #     # 第五排的数据
        #     out[..., 4::7, 0::7] = Crop_trans[..., 4::6, 0::6].cuda()
        #     out[..., 4::7, 1::7] = Crop_trans[..., 4::6, 1::6].cuda()
        #     out[..., 4::7, 2::7] = Crop_trans[..., 4::6, 2::6].cuda()
        #     out[..., 4::7, 3::7] = Crop_trans[..., 4::6, 3::6].cuda()
        #     out[..., 4::7, 4::7] = Crop_trans[..., 4::6, 4::6].cuda()
        #     out[..., 4::7, 5::7] = Crop_trans[..., 4::6, 5::6].cuda()
        #     out[..., 4::7, 6::7] = Crop_trans[..., 4::6, 5::6].cuda()
        #
        #     # 第六排的数据
        #     out[..., 5::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
        #     out[..., 5::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
        #     out[..., 5::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
        #     out[..., 5::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
        #     out[..., 5::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
        #     out[..., 5::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #     out[..., 5::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #
        #     # 第七排的数据
        #     out[..., 6::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
        #     out[..., 6::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
        #     out[..., 6::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
        #     out[..., 6::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
        #     out[..., 6::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
        #     out[..., 6::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #     out[..., 6::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #
        #     # out = out
        #
        #
        # #  下面直接对特征图进行裁切
        # if np.size(x, 2) == 14:
        #     # x = x
        #     # Crop_trans = x[..., 1:-1, 1:-1]      #a 这个操作是切除最外层的一个皮      直接切片索引就ok
        #     Crop_trans = x[..., 1:-1, 1:-1].cuda()      #a 这个操作是切除最外层的一个皮      直接切片索引就ok
        #     b, c, h, w = Crop_trans.shape
        #     out = torch.zeros((b, c, h * 7 // 6, w * 7 // 6))
        #     # 第一排的数据，6个值变成7个值
        #     out[..., 0::7, 0::7] = Crop_trans[..., 0::6, 0::6].cuda()
        #     out[..., 0::7, 1::7] = Crop_trans[..., 0::6, 1::6].cuda()
        #     out[..., 0::7, 2::7] = Crop_trans[..., 0::6, 2::6].cuda()
        #     out[..., 0::7, 3::7] = Crop_trans[..., 0::6, 3::6].cuda()
        #     out[..., 0::7, 4::7] = Crop_trans[..., 0::6, 4::6].cuda()
        #     out[..., 0::7, 5::7] = Crop_trans[..., 0::6, 5::6].cuda()
        #     out[..., 0::7, 6::7] = Crop_trans[..., 0::6, 5::6].cuda()
        #
        #     # 第二排的数据
        #     out[..., 1::7, 0::7] = Crop_trans[..., 1::6, 0::6].cuda()
        #     out[..., 1::7, 1::7] = Crop_trans[..., 1::6, 1::6].cuda()
        #     out[..., 1::7, 2::7] = Crop_trans[..., 1::6, 2::6].cuda()
        #     out[..., 1::7, 3::7] = Crop_trans[..., 1::6, 3::6].cuda()
        #     out[..., 1::7, 4::7] = Crop_trans[..., 1::6, 4::6].cuda()
        #     out[..., 1::7, 5::7] = Crop_trans[..., 1::6, 5::6].cuda()
        #     out[..., 1::7, 6::7] = Crop_trans[..., 1::6, 5::6].cuda()
        #
        #     # 第三排的数据
        #     out[..., 2::7, 0::7] = Crop_trans[..., 2::6, 0::6].cuda()
        #     out[..., 2::7, 1::7] = Crop_trans[..., 2::6, 1::6].cuda()
        #     out[..., 2::7, 2::7] = Crop_trans[..., 2::6, 2::6].cuda()
        #     out[..., 2::7, 3::7] = Crop_trans[..., 2::6, 3::6].cuda()
        #     out[..., 2::7, 4::7] = Crop_trans[..., 2::6, 4::6].cuda()
        #     out[..., 2::7, 5::7] = Crop_trans[..., 2::6, 5::6].cuda()
        #     out[..., 2::7, 6::7] = Crop_trans[..., 2::6, 5::6].cuda()
        #
        #     # 第四排的数据
        #     out[..., 3::7, 0::7] = Crop_trans[..., 3::6, 0::6].cuda()
        #     out[..., 3::7, 1::7] = Crop_trans[..., 3::6, 1::6].cuda()
        #     out[..., 3::7, 2::7] = Crop_trans[..., 3::6, 2::6].cuda()
        #     out[..., 3::7, 3::7] = Crop_trans[..., 3::6, 3::6].cuda()
        #     out[..., 3::7, 4::7] = Crop_trans[..., 3::6, 4::6].cuda()
        #     out[..., 3::7, 5::7] = Crop_trans[..., 3::6, 5::6].cuda()
        #     out[..., 3::7, 6::7] = Crop_trans[..., 3::6, 5::6].cuda()
        #
        #     # 第五排的数据
        #     out[..., 4::7, 0::7] = Crop_trans[..., 4::6, 0::6].cuda()
        #     out[..., 4::7, 1::7] = Crop_trans[..., 4::6, 1::6].cuda()
        #     out[..., 4::7, 2::7] = Crop_trans[..., 4::6, 2::6].cuda()
        #     out[..., 4::7, 3::7] = Crop_trans[..., 4::6, 3::6].cuda()
        #     out[..., 4::7, 4::7] = Crop_trans[..., 4::6, 4::6].cuda()
        #     out[..., 4::7, 5::7] = Crop_trans[..., 4::6, 5::6].cuda()
        #     out[..., 4::7, 6::7] = Crop_trans[..., 4::6, 5::6].cuda()
        #
        #     # 第六排的数据
        #     out[..., 5::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
        #     out[..., 5::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
        #     out[..., 5::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
        #     out[..., 5::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
        #     out[..., 5::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
        #     out[..., 5::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #     out[..., 5::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #
        #     # 第七排的数据
        #     out[..., 6::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
        #     out[..., 6::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
        #     out[..., 6::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
        #     out[..., 6::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
        #     out[..., 6::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
        #     out[..., 6::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #     out[..., 6::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()
        #
        #     # out = out
        # x = out.cuda()

        # x = self.CA1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.pool1(x)
        return x


import numpy as np
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.CA1 = CoordAtt(num_input_features, num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        input_channels = num_input_features
        output_channels = num_output_features
        self.branch3x3stack = nn.Sequential(
            nn.Conv2d(input_channels, int(((input_channels*output_channels)**0.5)*0.125), kernel_size=3, stride=2, padding=1),
            nn.Conv2d(int(((input_channels*output_channels)**0.5)*0.125), int(output_channels*1.0), kernel_size=3, stride=1, padding=1),
        )
        self.branch3s = nn.Sequential(
            nn.Conv2d(input_channels, int(((input_channels*output_channels)**0.5)*0.125), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(((input_channels*output_channels)**0.5)*0.125), int(output_channels), kernel_size=3, stride=1, padding=1),
        )
        self.branch3x3 = nn.Conv2d(input_channels, int(output_channels*1.0), kernel_size=3, stride=1, padding=1)
        self.branch3 = nn.Conv2d(input_channels, int(output_channels), kernel_size=3, stride=1, padding=1)
        self.branch1x1 = nn.Conv2d(input_channels, int(output_channels*1.0), kernel_size=1, stride=2)
        self.branch1 = nn.Conv2d(input_channels, int(output_channels*1.0), kernel_size=1, stride=1)
        self.resbranch1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2, bias=False)
        self.branchpool = nn.Sequential(
            # BasicConv2d(input_channels, int(output_channels*1.0), kernel_size=1),
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, int(output_channels*1.0), kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )           #这里是 256        最后合起来是1536          现在合算起来是64-48=16     每个补8即可

        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.apool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x = self.apool(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        # x = self.branch3s(x)
        x = self.pool1(x)
        return x

        # # branch3x3stack_output = self.branch3x3stack(x)
        # # branch3x3_output = self.branch3x3(x)
        # # branch1x1_output = self.branch1(x)
        # # resbranch1x1_output = self.resbranch1x1(x)
        # branchpool = self.branchpool(x)
        # output = [
        #     # branch1x1_output,     # 这个缓慢爬升
        #     # branch3x3_output,
        #     # branch3x3stack_output,
        #     branchpool      # 这个效果不错呢
        # ]
        # # InceptionA_1_5_res_out = self.bn2(torch.cat(output, 1))+resbranch1x1_output
        # # InceptionA_1_5_res_out = (torch.cat(output, 1))+resbranch1x1_output
        # InceptionA_1_5_res_out = (torch.cat(output, 1))
        # # InceptionA_1_5_res_out = self.relu(InceptionA_1_5_res_out)
        # # del branch3x3stack_output
        # # del branch3x3_output
        # # del branch1x1_output
        # # del resbranch1x1_output
        # del branchpool
        # del output
        #
        # return InceptionA_1_5_res_out

from torch.autograd import Function
# new layer
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data
        return grad_output

def hash_layer(input):
    return hash.apply(input)

#
# from torch.autograd import Function
# # new layer
# class hash(Function):
#     @staticmethod
#     def forward(ctx, input):
#         # ctx.save_for_backward(input)
#         return torch.sign(input)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # input,  = ctx.saved_tensors
#         # grad_output = grad_output.data
#         return grad_output
#
# def hash_layer(input):
#     return hash.apply(input)
#


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


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
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
                 # num_init_features=64, bn_size=1, drop_rate=0, num_classes=1000):

        super(SEDenseNet, self).__init__()

        # First convolution
        self.First = nn.Sequential(OrderedDict([
        # self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

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
        num_features = num_features + 16 * growth_rate
        # num_features = 1024

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

        # self.fc_encode = nn.Linear(num_features, encode_length)
        self.fc_encode = nn.Linear(num_features, 32)


    def forward(self, x):
        First = self.First(x)
        DenseBlockA = self._DenseBlockA(First)
        TransitionA = self._TransitionA(DenseBlockA)
        DenseBlockB = self._DenseBlockB(TransitionA)
        TransitionB = self._TransitionB(DenseBlockB)
        DenseBlockC = self._DenseBlockC(TransitionB)
        TransitionC = self._TransitionC(DenseBlockC)
        DenseBlockD = self._DenseBlockD(TransitionC)
        DenseBlockD = self.Final_bn(DenseBlockD)
        out = F.relu(DenseBlockD, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(DenseBlockD.size(0), -1)     # 这个是原始代码里的
        # # out = F.avg_pool2d(out, kernel_size=7, stride=1)     # 这个是原始代码里的
        # # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(out.size(0), -1)
        # # 出现问题RuntimeError: size mismatch, m1: [2 x 320000], m2: [1024 x 2] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:283
        # # out = out.view(out.size(0), -1)       很明显，size不匹配，原因有很多，请检查卷积核的尺寸和输入尺寸是否匹配，padding数是否正确。
        # # out = self.Final_bn(out).view(out.size(0), -1)
        # h = self.fc_encode(out)
        # out = self.classifier(out)
        # b = hash_layer(h)
        # # return out, h, b
        #

        out = self.classifier(out)
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