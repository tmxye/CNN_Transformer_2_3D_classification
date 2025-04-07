"""添加了senet模块,transition与block均有senet"""
import sys
sys.path.append("F:/car_classify_abnormal")

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from core.se_module import SELayer


__all__ = ['SEDenseNet', 'se_densenet121', 'se_densenet169', 'se_densenet201', 'se_densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class Inception(nn.Module):         #这里都添加了     , stride=2
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, stride=1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, stride=1, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

        # 'conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
        # bias = False      这两份的代码区别在于卷积后面有没有接BN操作

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class InceptionC(nn.Module):
    #输出尺寸是1536
    def __init__(self, input_channels):
        #"""Figure 6. The schema for 8×8 grid modules of the pure
        #Inceptionv4 network. This is the Inception-C block of Figure 9."""

        super().__init__()

        # self.branch3x3stack = nn.Sequential(
        #     BasicConv2d(input_channels, 384, kernel_size=1),
        #     BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
        #     BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),
        # )
        # self.branch3x3stacka = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        # self.branch3x3stackb = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))        #这里是 256+256
        #
        # self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=1)
        # self.branch3x3a = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))
        # self.branch3x3b = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))     #这里是256 +256
        #
        # self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1)        #这里是256
        #
        # self.branchpool = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        #     BasicConv2d(input_channels, 256, kernel_size=1)
        # )           #这里是 256        最后合起来是1536        x下面需要1536除以64=24，   也就是数据要降低24倍

        # 下面是修改为输出通道数为64的，但是没有步长，原始的输入应该将步长改为2
        # self.branch3x3stack = nn.Sequential(
        #     BasicConv2d(input_channels, 16, kernel_size=1),        #384---16
        #     BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
        #     BasicConv2d(20, 24, kernel_size=(3, 1), padding=(1, 0)),     #512-----21.33333333       给24算
        # )
        # self.branch3x3stacka = BasicConv2d(24, 12, kernel_size=(1, 3), padding=(0, 1))          #256----10.666667       给12算
        # self.branch3x3stackb = BasicConv2d(24, 12, kernel_size=(3, 1), padding=(1, 0))        #这里是 256+256     现在是12+12
        #
        # self.branch3x3 = BasicConv2d(input_channels, 16, kernel_size=1)
        # self.branch3x3a = BasicConv2d(16, 12, kernel_size=(3, 1), padding=(1, 0))          #256---10.667       给12算
        # self.branch3x3b = BasicConv2d(16, 12, kernel_size=(1, 3), padding=(0, 1))     #这里是256 +256      现在是12+12        目前是48
        #
        # self.branch1x1 = BasicConv2d(input_channels, 8, kernel_size=1)        #这里是256     256-----10.6667     待会用这个补  8
        #
        # self.branchpool = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        #     BasicConv2d(input_channels, 8, kernel_size=1)         #这里是256     256-----10.6667         待会用这个补    8
        # )           #这里是 256        最后合起来是1536          现在合算起来是64-48=16     每个补8即可


        # 现在将步长改为2
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 16, kernel_size=1),        #384---16
            BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
            BasicConv2d(20, 24, kernel_size=(3, 1), padding=(1, 0)),     #512-----21.33333333       给24算
        )
        self.branch3x3stacka = BasicConv2d(24, 12, kernel_size=(1, 3), stride=1, padding=(0, 1))          #256----10.666667       给12算
        self.branch3x3stackb = BasicConv2d(24, 12, kernel_size=(3, 1), stride=1, padding=(1, 0))        #这里是 256+256     现在是12+12

        self.branch3x3 = BasicConv2d(input_channels, 16, kernel_size=1)
        self.branch3x3a = BasicConv2d(16, 12, kernel_size=(3, 1), stride=1, padding=(1, 0))          #256---10.667       给12算
        self.branch3x3b = BasicConv2d(16, 12, kernel_size=(1, 3), stride=1, padding=(0, 1))     #这里是256 +256      现在是12+12        目前是48

        self.branch1x1 = BasicConv2d(input_channels, 8, kernel_size=1, stride=1)        #这里是256     256-----10.6667     待会用这个补  8

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 8, kernel_size=1)         #这里是256     256-----10.6667         待会用这个补    8
        )           #这里是 256        最后合起来是1536          现在合算起来是64-48=16     每个补8即可

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [
            self.branch3x3stacka(branch3x3stack_output),
            self.branch3x3stackb(branch3x3stack_output)
        ]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)

        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [
            self.branch3x3a(branch3x3_output),
            self.branch3x3b(branch3x3_output)
        ]
        branch3x3_output = torch.cat(branch3x3_output, 1)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)



def se_densenet121(pretrained=False, is_strict=False, drop_rate=0, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), drop_rate=drop_rate,
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
        print("模型从本地装载成功")
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
        print("模型从本地装载成功")
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
        print("模型从本地装载成功")
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
        print("模型从本地装载成功")
        model.load_state_dict(state_dict, strict=False)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        self.add_module("selayer", SELayer(channel=num_input_features)),

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),

        # by YXY        没有太大必要
        # self.add_module("selayer", SELayer(channel=num_input_features)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        # by YXY
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)),
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("selayer", SELayer(channel=num_input_features))
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class SEDenseNet(nn.Module):

    # by YXY
    # def _generate_inception_module(input_channels, output_channels, block_num, block):
    # # def _generate_inception_module(input_channels, output_channels):
    #     layers = nn.Sequential()
    #     for l in range(1):
    #         layers.add_module("{}_{}".format(InceptionC.__name__, l), InceptionC(input_channels))
    #         input_channels = output_channels
    #     return layers
    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels
        return layers

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

        # # by YXY    这里准备用inception替代，原文是一个1*1，  1*1、3*3，    1*1、5*5，     3*3最大池化、1*1卷积
        # # self.a3 = Inception(3, 64, 96, 128, 16, 32, 32)     #这样通道数是256      #64+128+32+32
        # self.a3 = Inception(3, 16, 24, 32, 4, 8, 8)     #这样通道数是256      现在改为了64
        # # # self.features = Inception(3, 16, 24, 32, 4, 8, 8)     #这样通道数是256
        # # # Inception输出特征图数量
        # # # torch.Size([16, 64, 224, 224])
        # # # torch.Size([16, 64, 224, 224])
        # # # 4
        # # # 第1个输出特征图数量
        # # # torch.Size([16, 64, 56, 56])
        # # # torch.Size([16, 64, 56, 56])      注意这里的图像大小是56


        # self.inception_c = self._generate_inception_module(1536, 1536, C, InceptionC)       #C:block_num
        self.features = self._generate_inception_module(3, 64, 1, InceptionC)
        #C:block_num        这里把3改为1即可

        #
        # # self.a4 = nn.Sequential(OrderedDict([
        # #     ('norm0', nn.BatchNorm2d(num_init_features)),
        # #     ('relu0', nn.ReLU(inplace=True)),
        # #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # # ]))
        # ### 这里是不需要池化，如果需要修改就只需要把下面这个注释删除掉
        # self.features = nn.Sequential(OrderedDict([
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))

        # self.a5 = nn.Sequential(OrderedDict([
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))
        # #
        # # First convolution test by YXY
        # self.a3 = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))
        # # # #'conv0', nn.Conv2d(3, num_ini      这里的3是输入的通道数
        # #
        # #
        # # First convolution
        # self.featuresa = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))


        # # Add SELayer at first convolution
        # # self.features.add_module("SELayer_0a", SELayer(channel=num_init_features))
        # # self.features = SELayer(channel=num_init_features)
        # # self.features = nn.Sequential(['norm0', nn.BatchNorm2d(num_init_features)])
        # self.featuresb = nn.Sequential(OrderedDict([
        #     # ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))
        # # #
        # # # Add SELayer at first convolution      by YXY,作为开始
        # # self.features("SELayer_0a", SELayer(channel=num_init_features))
        #
        # # Each denseblock
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            # Add a SELayer
            # self.features.add_module("SELayer_%da" % (i + 1), SELayer(channel=num_features))

            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                # Add a SELayer behind each transition block
                # self.features.add_module("SELayer_%db" % (i + 1), SELayer(channel=num_features))

                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Add SELayer
        self.features.add_module("SELayer_0b", SELayer(channel=num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # a3 = self.a3(x)         # 16    64      112     112         #初始的输出是：

        # featuresa = self.featuresa(x)
        # a = self.a(x)         # 16    64      112     112         #初始的输出是：
        # Inception_c = self.Inception_c(x)
        # print(featuresa.shape)# 尺寸
        # print(Inception_c.size()) # 形状
        # print(Inception_c.ndim) # 维数
        # a4 = self.a5(x)
        # print(a4.shape)# 尺寸
        # # # a4 = self.a4(a3)
        # # # # a4 = self.a5(a4)
        # # # print('\nInception输出特征图数量')
        # # # # print(x)
        # print(a.shape)# 尺寸
        # # # print(a3.size()) # 形状
        # # # print(a3.ndim) # 维数
        # # print('\nInception输出特征图数量')
        # # # print(x)
        # # print(Inception_c.shape)# 尺寸
        # # print(Inception_c.size()) # 形状
        # # print(Inception_c.ndim) # 维数
        # # # print('\nInception输出特征图数量')
        # # # # print(x)
        # # # print(a4.shape)# 尺寸
        # # # print(a4.size()) # 形状
        # # # print(a4.ndim) # 维数
        # # # # b1 = self.b1(x)
        # # # ## print('\nInception输出特征图数量')
        # # # ## # print(x)
        # # # ## print(b1.shape)# 尺寸
        # # # ## print(b1.size()) # 形状
        # # # ## print(b1.ndim) # 维数
        # # #
        # # # print('第一个输入特征图数量')
        # # # print(x.shape)# 尺寸
        # # # print(x.size()) # 形状
        # # # print(x.ndim) # 维数)
        # # #
        # # # # a = self.a(x)
        # # # # print('第1个输出特征图数量')
        # # # # print(a.shape)# 尺寸
        # # # # print(a.size()) # 形状
        # # # # print(a.ndim) # 维数)
        # # #
        # # # # features = self.features(a4)
        # # # features = self.features(a3)

        features = self.features(x)
        # print(features.size())
        # #
        # # print('第n个输出特征图数量')
        # # print(features.shape)# 尺寸
        # # print(features.size()) # 形状
        # # print(features.ndim) # 维数)

        # features = self.features(x)

        out = F.relu(features, inplace=True)
        # print(out.size())
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        # print(out.size())
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

