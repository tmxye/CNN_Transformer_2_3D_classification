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
from .layer import MultiSpectralAttentionLayer

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
        self.pool_h = nn.AdaptiveAvgPool2d((28, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, 28))

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
        # print("y", y.shape)         # torch.Size([1, 128, 56, 1])
        y = self.conv1(y)
        # print("y", y.shape)         # torch.Size([1, 8, 56, 1])
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



class CoordAttA(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((56, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, 56))
        self.pool_z = nn.AdaptiveAvgPool2d((1, 1))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        # print("x_h", x_h.shape)         # torch.Size([1, 128, 56, 1])
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_z = self.pool_z(x_h)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_z = self.conv1(x_z)
        x_z = self.bn1(x_z)
        x_z = self.relu(x_z)
        # print("x_z", x_z.shape)     # ([1, 8, 1, 1])
        x_z = x_z.expand(-1, -1, h, -1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_h = x_h * x_z
        x_w = x_w * x_z
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        # print("x_z", x_z.shape)      # ([1, 64, 56, 56])

        y = identity * x_w * x_h

        return y





class CoordAttB(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttB, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((28, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, 28))
        self.pool_z = nn.AdaptiveAvgPool2d((1, 1))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        # print("x_h", x_h.shape)         # torch.Size([1, 128, 56, 1])
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_z = self.pool_z(x_h)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_z = self.conv1(x_z)
        x_z = self.bn1(x_z)
        x_z = self.relu(x_z)
        # print("x_z", x_z.shape)     # ([1, 8, 1, 1])
        x_z = x_z.expand(-1, -1, h, -1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_h = x_h * x_z
        x_w = x_w * x_z
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        # print("x_z", x_z.shape)      # ([1, 64, 56, 56])

        y = identity * x_w * x_h

        return y





class CoordAttC(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttC, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((14, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, 14))
        self.pool_z = nn.AdaptiveAvgPool2d((1, 1))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        # print("x_h", x_h.shape)         # torch.Size([1, 128, 56, 1])
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_z = self.pool_z(x_h)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_z = self.conv1(x_z)
        x_z = self.bn1(x_z)
        x_z = self.relu(x_z)
        # print("x_z", x_z.shape)     # ([1, 8, 1, 1])
        x_z = x_z.expand(-1, -1, h, -1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_h = x_h * x_z
        x_w = x_w * x_z
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        # print("x_z", x_z.shape)      # ([1, 64, 56, 56])

        y = identity * x_w * x_h

        return y






class CoordAttD(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttD, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((7, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, 7))
        self.pool_z = nn.AdaptiveAvgPool2d((1, 1))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        # print("x_h", x_h.shape)         # torch.Size([1, 128, 56, 1])
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_z = self.pool_z(x_h)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_z = self.conv1(x_z)
        x_z = self.bn1(x_z)
        x_z = self.relu(x_z)
        # print("x_z", x_z.shape)     # ([1, 8, 1, 1])
        x_z = x_z.expand(-1, -1, h, -1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_h = x_h * x_z
        x_w = x_w * x_z
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        # print("x_z", x_z.shape)      # ([1, 64, 56, 56])

        y = identity * x_w * x_h

        return y






class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        self.SElayer = SELayer(channel=num_input_features)
        # c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # self.SElayer = MultiSpectralAttentionLayer(num_input_features, 56, 56, reduction=16, freq_sel_method='top16')
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.SElayer(x)
        new_features = self.bn1(new_features)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        new_features = self.conv3(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseLayerA(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerA, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.SElayer = SELayer(channel=num_input_features)
        self.SElayer = CoordAttA(num_input_features, num_input_features)
        # c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # self.SElayer = MultiSpectralAttentionLayer(num_input_features, 56, 56, reduction=16, freq_sel_method='top16')
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.SElayer(x)
        new_features = self.bn1(new_features)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        new_features = self.conv3(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseLayerB(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerB, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.SElayer = SELayer(channel=num_input_features)
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # self.SElayer = MultiSpectralAttentionLayer(num_input_features, 28, 28, reduction=16, freq_sel_method='top16')
        self.SElayer = CoordAttB(num_input_features, num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.SElayer(x)
        new_features = self.bn1(new_features)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        new_features = self.conv3(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)



class _DenseLayerC(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerC, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.SElayer = SELayer(channel=num_input_features)
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # self.SElayer = MultiSpectralAttentionLayer(num_input_features, 28, 28, reduction=16, freq_sel_method='top16')
        self.SElayer = CoordAttC(num_input_features, num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.SElayer(x)
        new_features = self.bn1(new_features)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        new_features = self.conv3(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)



class _DenseLayerD(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerD, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.SElayer = SELayer(channel=num_input_features)
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        # self.SElayer = MultiSpectralAttentionLayer(num_input_features, 28, 28, reduction=16, freq_sel_method='top16')
        self.SElayer = CoordAttD(num_input_features, num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.SElayer(x)
        new_features = self.bn1(new_features)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.bn3(new_features)
        new_features = self.relu3(new_features)
        new_features = self.conv3(new_features)
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

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.SElayer1 = SELayer(num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.SElayer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
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
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

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
            # elif isinstance(m, nn.Linear):
            #     nn.init.constant_(m.bias, 0)

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
        # out = F.avg_pool2d(out, kernel_size=7, stride=1)     # 这个是原始代码里的
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(out.size(0), -1)
        # 出现问题RuntimeError: size mismatch, m1: [2 x 320000], m2: [1024 x 2] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:283
        # out = out.view(out.size(0), -1)       很明显，size不匹配，原因有很多，请检查卷积核的尺寸和输入尺寸是否匹配，padding数是否正确。
        # out = self.Final_bn(out).view(out.size(0), -1)
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