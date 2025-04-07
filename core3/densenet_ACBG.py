"""添加了senet模块,去除transition的senet模块,仅仅在denseblock中添加senet模块"""
import sys
sys.path.append("F:/car_classify_abnormal")
# 这个和full_in loop ,几乎一样，在171行FUll_loop注释了一行
#  # Add a SELayer behind each transition block这行也注释了   206行

import re
import torch
import torch.nn as nn
import torch.nn.init as init
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


class ACBlock(nn.Module):
    # Y:\05DenseNet_Dense_SA\data\21_10_旧\ACNet         来自于官方的ACNet
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, gamma_init=None ):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)


            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            if reduce_gamma:
                self.init_gamma(1.0 / 3)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)


    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b


    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels, out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size, stride=self.square_conv.stride,
                                    padding=self.square_conv.padding, dilation=self.square_conv.dilation, groups=self.square_conv.groups, bias=True,
                                    padding_mode=self.square_conv.padding_mode)
        self.__delattr__('square_conv')
        self.__delattr__('square_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b


    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result





class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.SElayer = SELayer(channel=num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, groups=growth_rate, bias=False)
        self.bn3 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv3 = ACBlock(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,  padding=1, dilation=1, groups=1,
        #                       padding_mode='zeros', deploy=False, use_last_bn=False, gamma_init=1/3)
        # self.conv3 = ACBlock(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,  padding=1, dilation=1, groups=1,
        #                       padding_mode='zeros', deploy=False, gamma_init=1/3)       #参数量64MB
        self.conv3 = ACBlock(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,  padding=1, dilation=1, groups=growth_rate,
                              padding_mode='zeros', deploy=False, gamma_init=1/3)       #参数量32MB
        self.drop_rate = drop_rate

    def forward(self, x):
        # new_features = self.SElayer(x)
        new_features = self.bn1(x)
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

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.SElayer1 = SELayer(num_input_features)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, groups=num_output_features, bias=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x = self.SElayer1(x)
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
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

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