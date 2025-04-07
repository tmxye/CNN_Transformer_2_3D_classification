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



class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        return x


class InceptionC(nn.Module):
    #输出尺寸是1536
    def __init__(self, input_channels):
        #"""Figure 6. The schema for 8×8 grid modules of the pure
        #Inceptionv4 network. This is the Inception-C block of Figure 9."""

        super().__init__()


        # 现在将步长改为2
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 16, kernel_size=1),        #384---16
            BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
            BasicConv2d(20, 24, kernel_size=(3, 1), padding=(1, 0)),     #512-----21.33333333       给24算
        )
        self.branch3x3stacka = BasicConv2d(24, 12, kernel_size=(1, 3), stride=2, padding=(0, 1))          #256----10.666667       给12算
        self.branch3x3stackb = BasicConv2d(24, 12, kernel_size=(3, 1), stride=2, padding=(1, 0))        #这里是 256+256     现在是12+12

        self.branch3x3 = BasicConv2d(input_channels, 16, kernel_size=1)
        self.branch3x3a = BasicConv2d(16, 12, kernel_size=(3, 1), stride=2, padding=(1, 0))          #256---10.667       给12算
        self.branch3x3b = BasicConv2d(16, 12, kernel_size=(1, 3), stride=2, padding=(0, 1))     #这里是256 +256      现在是12+12        目前是48

        self.branch1x1 = BasicConv2d(input_channels, 8, kernel_size=1, stride=2)        #这里是256     256-----10.6667     待会用这个补  8

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
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



class InceptionA_1_1(nn.Module):
    #输出尺寸是1536
    def __init__(self, input_channels):
        #"""Figure 6. The schema for 8×8 grid modules of the pure
        #Inceptionv4 network. This is the Inception-C block of Figure 9."""

        super().__init__()

        # 现在将步长改为2
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 16, kernel_size=1),        #384---16
            BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
            BasicConv2d(20, 16, kernel_size=(3, 1), padding=(1, 0)),     #512-----21.33333333       给24算
            BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
            BasicConv2d(20, 16, kernel_size=(3, 1), padding=(1, 0)),     #512-----21.33333333       给24算
            BasicConv2d(16, 16, kernel_size=1, stride=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 16, kernel_size=1),        #384---16
            BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
            BasicConv2d(20, 16, kernel_size=(3, 1), padding=(1, 0)),     #512-----21.33333333       给24算
            BasicConv2d(16, 16, kernel_size=1, stride=2)
        )

        self.branch1x1 = BasicConv2d(input_channels, 16, kernel_size=1, stride=2)        #这里是256     256-----10.6667     待会用这个补  8

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(input_channels, 16, kernel_size=1)         #这里是256     256-----10.6667         待会用这个补    8
        )           #这里是 256        最后合起来是1536          现在合算起来是64-48=16     每个补8即可

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)

        branch3x3_output = self.branch3x3(x)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)




class InceptionA_1_4(nn.Module):
    #输出尺寸是1536
    def __init__(self, input_channels, output_channels):
        #"""Figure 6. The schema for 8×8 grid modules of the pure
        #Inceptionv4 network. This is the Inception-C block of Figure 9."""

        super().__init__()

        # 现在将步长改为2
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, int(output_channels*0.25), kernel_size=1),        #384---16
            # BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
            # BasicConv2d(20, 16, kernel_size=(3, 1), padding=(1, 0)),     #512-----21.33333333       给24算
            # BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
            # BasicConv2d(20, 16, kernel_size=(3, 1), padding=(1, 0)),     #512-----21.33333333       给24算
            BasicConv2d(int(output_channels*0.25), int(output_channels*0.25), kernel_size=3, stride=1, padding=1),
            BasicConv2d(int(output_channels*0.25), int(output_channels*0.25), kernel_size=3, stride=2, padding=1),
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, int(output_channels*0.25), kernel_size=1),        #384---16
            # BasicConv2d(16, 20, kernel_size=(1, 3), padding=(0, 1)),       # 448---18.666666667     给20算
            # BasicConv2d(20, 16, kernel_size=(3, 1), stride=2, padding=(1, 0)),     #512-----21.33333333       给24算
            BasicConv2d(int(output_channels*0.25), int(output_channels*0.25), kernel_size=3, stride=2, padding=1),
            # BasicConv2d(16, 16, kernel_size=1, stride=2)
        )

        self.branch1x1 = BasicConv2d(input_channels, int(output_channels*0.25), kernel_size=1, stride=2)        #这里是256     256-----10.6667     待会用这个补  8

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(input_channels, int(output_channels*0.25), kernel_size=1)         #这里是256     256-----10.6667         待会用这个补    8
        )           #这里是 256        最后合起来是1536          现在合算起来是64-48=16     每个补8即可


    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)

        branch3x3_output = self.branch3x3(x)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)




def se_densenet121(pretrained=False, is_strict=False, drop_rate=0.2, num_classes=2, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), drop_rate=drop_rate, num_classes=num_classes,
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
        # # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.add_module("selayer", SELayer(channel=num_input_features)),
        #
        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                 growth_rate, kernel_size=1, stride=1, bias=False)),
        #
        # # by YXY        没有太大必要
        # # by YXY        这里是原文的残差连接，文章是https://www.sciencedirect.com/science/article/pii/S0925231218309147   04 0226-40 A Deep Learning Approach to Identify Blepharoptosis by Convolutional Neural Networks – ScienceDirect
        # # 因此需要把SE放在1*1和3*3之后
        # self.add_module("selayer", SELayer(channel=num_input_features)),
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        #
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, padding=1, bias=False)),



        # # by YXY，这个是准备使用过在e上面的结构
        # # 因此需要把SE放在1*1和3*3之后
        # # 这样调整之后SE里面的特征图输入大小需要进行如下更改
        # #   1、在1*1和3*3之间的需要改为bn_size *  growth_rate
        # #   1、在3*3之后的需要改为growth_rate，不能改为bn_size *  growth_rate，是因为在下一个块中SE后面接的是下一个单元的1*1卷积，因此应该修改为那个特征图通道数
        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                 growth_rate, kernel_size=1, stride=1, bias=False)),
        # # by YXY        这里是原文的残差连接，文章是https://www.sciencedirect.com/science/article/pii/S0925231218309147   04 0226-40 A Deep Learning Approach to Identify Blepharoptosis by Convolutional Neural Networks – ScienceDirect
        # # 因此需要把SE放在1*1和3*3之后
        # self.add_module("selayer", SELayer(channel=bn_size *  growth_rate)),
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        #
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, padding=1, bias=False)),
        #
        # self.add_module("selayer", SELayer(channel=bn_size *  growth_rate)),





        # # 现在把残差的关键代码放在这里,再看看如何添加进去
        # # def forward(self, x):
        # #     identity = x        #
        # #     out = self.conv1(x)        #     out = self.bn1(out)        #     out = self.relu(out)        #
        # #     out = self.conv2(out)        #     out = self.bn2(out)        #     out = self.relu(out)        #
        # #     out = self.conv3(out)        #     out = self.bn3(out)
        # #
        # #     if self.downsample is not None:        #         identity = self.downsample(x)        #
        # #     out += identity           #     out = self.relu(out)
        # # 可以观察到残差连接的核心在于残差连接,这个操作的实现在反向传播函数里面
        # # 在densenet中使用cat去拼接通道,因此我要在块内实现特征的拼接,现在开始对代码进行再次修改,修改原作者的原始代码
        #
        # # 先对原始代码进行修改，改为普通的卷积操作，而不是add_model
        # self.a = nn.Sequential(
        #     nn.ReLU(num_input_features),
        #     nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
        # )
        # self.b = SELayer(bn_size * growth_rate)
        #
        # self.c = nn.Sequential(
        #     nn.ReLU(num_input_features),
        #     nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        # )
        # self.d = SELayer(growth_rate)
        #
        # # self.add_module('relu1', nn.ReLU(num_input_features)),
        # # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        # #
        # # self.add_module("selayer", SELayer(channel=bn_size *  growth_rate)),
        # #
        # # self.add_module('relu2', nn.ReLU(num_input_features)),
        # # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        # #
        # # self.add_module("selayer", SELayer(channel=bn_size *  growth_rate)),



        # # Add SELayer at here, like SE-PRE block in original paper illustrates
        # self.add_module("selayer", SELayer(channel=num_input_features)),
        #
        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                 growth_rate, kernel_size=1, stride=1, bias=False)),
        #
        # # by YXY        没有太大必要
        # # by YXY        这里是原文的残差连接，文章是https://www.sciencedirect.com/science/article/pii/S0925231218309147   04 0226-40 A Deep Learning Approach to Identify Blepharoptosis by Convolutional Neural Networks – ScienceDirect
        # # 因此需要把SE放在1*1和3*3之后
        # self.add_module("selayer", SELayer(channel=num_input_features)),
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        #
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, padding=1, bias=False)),


        # 经过试验验证，下面这样的写法是是合适的，因为要使用残差连接，bn_size需要修改为1，和growth_rate要保持一样的倍速增长
        # # 先实现在1*1之间的残差连接,在SE模块之前,在第二个3*3卷积之后
        # 现在已经修改好了原始的，如上一个所示，同样的反向传播函数也被修改了，现在开始考虑添加第一个残差连接，发现这里只有一个残差连接
        self.a = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.e = nn.Sequential(
            # nn.BatchNorm2d(num_input_features),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(num_input_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
            nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False)
        )
        # 第一个残差连接在这里开始,     刚看之后才发现残差连接只有一下，在卷积之后连接到卷积之后
        # self.b = SELayer(bn_size * growth_rate)

        self.c = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        # 第一个残差连接到这里结束
        # self.d = SELayer(growth_rate)

        self.drop_rate = drop_rate

    def forward(self, x):
        # # new_features = super(_DenseLayer, self).forward(x)
        a = self.a(x)
        e = self.e(x)
        # b = self.b(a)
        c = self.c(a)
        # print(a.size())
        # print(c.size())

        c += e
        # new_features = self.d(c)
        # # #     不添加se
        # # a = self.a(x)
        # # new_features = self.c(a)
        #
        # # new_features = super(_DenseLayer, self).forward(x)
        # print('\nx')
        # print(x.size())
        # print('\na')
        # print(a.size())
        # print('b')
        # print(b.size())
        # print('c')
        # print(c.size())
        # #a  torch.Size([1, 128, 56, 56])        # b   # torch.Size([1, 128, 56, 56])        # c  # torch.Size([1, 32, 56, 56])        # new_features  # torch.Size([1, 32, 56, 56])
        # print('new_features')
        # print(new_features.size())
        if self.drop_rate > 0:
            new_features = F.dropout(c, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
        # return torch.cat([x, a, b, c, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        # self.add_module("selayer", SELayer(channel=num_output_features))


class SEDenseNet(nn.Module):

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

    # def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
    #              num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=1, drop_rate=0, num_classes=1000):       #经过计算，为了残差连接能实现，把这里的bn_size改为1合适，通道数全部改为32了     r如果需要加大必须要保证参数同步增加倍数

        super(SEDenseNet, self).__init__()

        # by YXY    这里准备用inception替代，原文是一个1*1，  1*1、3*3，    1*1、5*5，     3*3最大池化、1*1卷积
        # self.Inception_c = self._generate_inception_module(3, 64, 1, InceptionC)
        self.Inception_c = self._generate_inception_module(3, 64, 1, InceptionA_1_4)


        self.features = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # # First convolution
        # self.features = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))

        # Add SELayer at first convolution
        # self.features.add_module("SELayer_0a", SELayer(channel=num_init_features))

        # Each denseblock
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
        # self.features.add_module("SELayer_0b", SELayer(channel=num_features))

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
        Inception_c = self.Inception_c(x)
        features = self.features(Inception_c)
        # features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
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

