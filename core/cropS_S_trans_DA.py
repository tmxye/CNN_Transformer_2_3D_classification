"""添加了senet模块,transition与block均有senet"""
import sys

import torchvision

sys.path.append("F:/car_classify_abnormal")

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from core.se_module import SELayer
import numpy as np


__all__ = ['SEDenseNet', 'se_densenet121']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}



class InceptionA_1_1_res(nn.Module):
    def __init__(self, input_channels, output_channels):

        super().__init__()

        self.branch3x3stack = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, int(output_channels*0.25), kernel_size=3, stride=1, padding=1),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=3, stride=2, padding=1),
        )
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=1, stride=2),        #这里是256     256-----10.6667     待会用这个补  8
        )
        self.resbranch1x1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2, bias=False),
        )
        self.branchpool = nn.Sequential(
            # BasicConv2d(input_channels, int(output_channels*0.25), kernel_size=1),
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )           #这里是 256        最后合起来是1536          现在合算起来是64-48=16     每个补8即可

        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        # print("branch3x3stack_output", branch3x3stack_output.shape)
        branch3x3_output = self.branch3x3(x)
        branch1x1_output = self.branch1x1(x)
        resbranch1x1_output = self.resbranch1x1(x)
        branchpool = self.branchpool(x)
        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]
        InceptionA_1_5_res_out = self.bn2(torch.cat(output, 1))+resbranch1x1_output
        InceptionA_1_5_res_out = self.relu(InceptionA_1_5_res_out)
        del branch3x3stack_output
        del branch3x3_output
        del branch1x1_output
        del resbranch1x1_output
        del branchpool
        del output

        return InceptionA_1_5_res_out




class InceptionA_1_4_res(nn.Module):
    def __init__(self, input_channels, output_channels):

        super().__init__()

        self.branch3x3stack = nn.Sequential(
            nn.Conv2d(input_channels, int(((input_channels*output_channels)**0.5)*0.125), kernel_size=3, stride=2, padding=1),
            nn.Conv2d(int(((input_channels*output_channels)**0.5)*0.125), int(output_channels*0.25), kernel_size=3, stride=1, padding=1),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=3, stride=2, padding=1),
        )
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=1, stride=2),        #这里是256     256-----10.6667     待会用这个补  8
        )
        self.resbranch1x1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2, bias=False),
        )
        self.branchpool = nn.Sequential(
            # BasicConv2d(input_channels, int(output_channels*0.25), kernel_size=1),
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )           #这里是 256        最后合起来是1536          现在合算起来是64-48=16     每个补8即可

        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3_output = self.branch3x3(x)
        branch1x1_output = self.branch1x1(x)
        resbranch1x1_output = self.resbranch1x1(x)
        branchpool = self.branchpool(x)
        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]
        InceptionA_1_5_res_out = self.bn2(torch.cat(output, 1))+resbranch1x1_output
        InceptionA_1_5_res_out = self.relu(InceptionA_1_5_res_out)
        del branch3x3stack_output
        del branch3x3_output
        del branch1x1_output
        del resbranch1x1_output
        del branchpool
        del output

        return InceptionA_1_5_res_out



class InceptionA_1_5_res(nn.Module):
    def __init__(self, input_channels, output_channels):

        super().__init__()

        self.branch3x3stack = nn.Sequential(
            nn.Conv2d(input_channels, int(((input_channels*output_channels)**0.5)*0.125), kernel_size=3, stride=2, padding=1),
            nn.Conv2d(int(((input_channels*output_channels)**0.5)*0.125), int(output_channels*0.25), kernel_size=3, stride=1, padding=1),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=3, stride=2, padding=1),
        )
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=1, stride=2),        #这里是256     256-----10.6667     待会用这个补  8
        )
        self.resbranch1x1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2, bias=False),
        )
        self.branchpool = nn.Sequential(
            # BasicConv2d(input_channels, int(output_channels*0.25), kernel_size=1),
            nn.Conv2d(input_channels, int(output_channels*0.25), kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )           #这里是 256        最后合起来是1536          现在合算起来是64-48=16     每个补8即可

        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3_output = self.branch3x3(x)
        branch1x1_output = self.branch1x1(x)
        resbranch1x1_output = self.resbranch1x1(x)
        branchpool = self.branchpool(x)
        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]
        InceptionA_1_5_res_out = self.bn2(torch.cat(output, 1))+resbranch1x1_output
        InceptionA_1_5_res_out = self.relu(InceptionA_1_5_res_out)
        del branch3x3stack_output
        del branch3x3_output
        del branch1x1_output
        del resbranch1x1_output
        del branchpool
        del output

        return InceptionA_1_5_res_out




def se_densenet121(pretrained=False, is_strict=False, drop_rate=0.2, num_classes=2, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), drop_rate=drop_rate, num_classes=num_classes,
    # model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(2, 6, 12, 8), drop_rate=drop_rate, num_classes=num_classes,
    # model = SEDenseNet(num_init_features=64, growth_rate=16, block_config=(6, 12, 24, 16), drop_rate=drop_rate, num_classes=num_classes,
    # model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(2, 6, 15, 2), drop_rate=drop_rate, num_classes=num_classes,
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

#
#
# class _DenseLayer(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseLayer, self).__init__()
#         # 经过试验验证，下面这样的写法是是合适的，因为要使用残差连接，bn_size需要修改为1，和growth_rate要保持一样的倍速增长
#         # # 先实现在1*1之间的残差连接,在SE模块之前,在第二个3*3卷积之后
#         # 现在已经修改好了原始的，如上一个所示，同样的反向传播函数也被修改了，现在开始考虑添加第一个残差连接，发现这里只有一个残差连接
#         self.a = nn.Sequential(
#             nn.BatchNorm2d(num_input_features),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
#         )
#
#         self.c = nn.Sequential(
#             nn.BatchNorm2d(bn_size * growth_rate),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
#         )
#
#         self.drop_rate = drop_rate
#
#     def forward(self, x):
#         a = self.a(x)
#         a = self.c(a)
#
#         # a += e
#         if self.drop_rate > 0:
#             a = F.dropout(a, p=self.drop_rate, training=self.training)
#         # del e
#         return torch.cat([x, a], 1)
#         # return torch.cat([x, a, b, c, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _TransitionLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionLayer, self).__init__()
        # self.Inception_c = generate_inception_module(num_input_features*2, num_output_features, 1, InceptionA_1_5_res)
        self.Inception_c = generate_inception_module(num_input_features, num_output_features, 1, InceptionA_1_5_res)

    def forward(self, x, firstbatchsize=16):
        x = x
        # print("x" , x)
        if np.size(x, 2) == 56:
            # Crop_trans = x[firstbatchsize, 512, 1:-1, 1:-1]      #a 这个操作是切除最外层的一个皮
            Crop_trans = x[..., 4:-4, 4:-4]      #a 这个操作是切除最外层的一个皮      直接切片索引就ok
            b, c, h, w = Crop_trans.shape
            out = torch.zeros((b, c, h * 7 // 6, w * 7 // 6))
            # 第一排的数据，6个值变成7个值
            out[..., 0::7, 0::7] = Crop_trans[..., 0::6, 0::6].cuda()
            out[..., 0::7, 1::7] = Crop_trans[..., 0::6, 1::6].cuda()
            out[..., 0::7, 2::7] = Crop_trans[..., 0::6, 2::6].cuda()
            out[..., 0::7, 3::7] = Crop_trans[..., 0::6, 3::6].cuda()
            out[..., 0::7, 4::7] = Crop_trans[..., 0::6, 4::6].cuda()
            out[..., 0::7, 5::7] = Crop_trans[..., 0::6, 5::6].cuda()
            out[..., 0::7, 6::7] = Crop_trans[..., 0::6, 5::6].cuda()

            # 第二排的数据
            out[..., 1::7, 0::7] = Crop_trans[..., 1::6, 0::6].cuda()
            out[..., 1::7, 1::7] = Crop_trans[..., 1::6, 1::6].cuda()
            out[..., 1::7, 2::7] = Crop_trans[..., 1::6, 2::6].cuda()
            out[..., 1::7, 3::7] = Crop_trans[..., 1::6, 3::6].cuda()
            out[..., 1::7, 4::7] = Crop_trans[..., 1::6, 4::6].cuda()
            out[..., 1::7, 5::7] = Crop_trans[..., 1::6, 5::6].cuda()
            out[..., 1::7, 6::7] = Crop_trans[..., 1::6, 5::6].cuda()

            # 第三排的数据
            out[..., 2::7, 0::7] = Crop_trans[..., 2::6, 0::6].cuda()
            out[..., 2::7, 1::7] = Crop_trans[..., 2::6, 1::6].cuda()
            out[..., 2::7, 2::7] = Crop_trans[..., 2::6, 2::6].cuda()
            out[..., 2::7, 3::7] = Crop_trans[..., 2::6, 3::6].cuda()
            out[..., 2::7, 4::7] = Crop_trans[..., 2::6, 4::6].cuda()
            out[..., 2::7, 5::7] = Crop_trans[..., 2::6, 5::6].cuda()
            out[..., 2::7, 6::7] = Crop_trans[..., 2::6, 5::6].cuda()

            # 第四排的数据
            out[..., 3::7, 0::7] = Crop_trans[..., 3::6, 0::6].cuda()
            out[..., 3::7, 1::7] = Crop_trans[..., 3::6, 1::6].cuda()
            out[..., 3::7, 2::7] = Crop_trans[..., 3::6, 2::6].cuda()
            out[..., 3::7, 3::7] = Crop_trans[..., 3::6, 3::6].cuda()
            out[..., 3::7, 4::7] = Crop_trans[..., 3::6, 4::6].cuda()
            out[..., 3::7, 5::7] = Crop_trans[..., 3::6, 5::6].cuda()
            out[..., 3::7, 6::7] = Crop_trans[..., 3::6, 5::6].cuda()

            # 第五排的数据
            out[..., 4::7, 0::7] = Crop_trans[..., 4::6, 0::6].cuda()
            out[..., 4::7, 1::7] = Crop_trans[..., 4::6, 1::6].cuda()
            out[..., 4::7, 2::7] = Crop_trans[..., 4::6, 2::6].cuda()
            out[..., 4::7, 3::7] = Crop_trans[..., 4::6, 3::6].cuda()
            out[..., 4::7, 4::7] = Crop_trans[..., 4::6, 4::6].cuda()
            out[..., 4::7, 5::7] = Crop_trans[..., 4::6, 5::6].cuda()
            out[..., 4::7, 6::7] = Crop_trans[..., 4::6, 5::6].cuda()

            # 第六排的数据
            out[..., 5::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
            out[..., 5::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
            out[..., 5::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
            out[..., 5::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
            out[..., 5::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
            out[..., 5::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
            out[..., 5::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()

            # 第七排的数据
            out[..., 6::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
            out[..., 6::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
            out[..., 6::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
            out[..., 6::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
            out[..., 6::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
            out[..., 6::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
            out[..., 6::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()


        #  下面直接对特征图进行裁切
        if np.size(x, 2) == 28:
            # x = x
            Crop_trans = x[..., 2:-2, 2:-2]      #a 这个操作是切除最外层的一个皮      直接切片索引就ok
            b, c, h, w = Crop_trans.shape
            out = torch.zeros((b, c, h * 7 // 6, w * 7 // 6))
            # 第一排的数据，6个值变成7个值
            out[..., 0::7, 0::7] = Crop_trans[..., 0::6, 0::6].cuda()
            out[..., 0::7, 1::7] = Crop_trans[..., 0::6, 1::6].cuda()
            out[..., 0::7, 2::7] = Crop_trans[..., 0::6, 2::6].cuda()
            out[..., 0::7, 3::7] = Crop_trans[..., 0::6, 3::6].cuda()
            out[..., 0::7, 4::7] = Crop_trans[..., 0::6, 4::6].cuda()
            out[..., 0::7, 5::7] = Crop_trans[..., 0::6, 5::6].cuda()
            out[..., 0::7, 6::7] = Crop_trans[..., 0::6, 5::6].cuda()

            # 第二排的数据
            out[..., 1::7, 0::7] = Crop_trans[..., 1::6, 0::6].cuda()
            out[..., 1::7, 1::7] = Crop_trans[..., 1::6, 1::6].cuda()
            out[..., 1::7, 2::7] = Crop_trans[..., 1::6, 2::6].cuda()
            out[..., 1::7, 3::7] = Crop_trans[..., 1::6, 3::6].cuda()
            out[..., 1::7, 4::7] = Crop_trans[..., 1::6, 4::6].cuda()
            out[..., 1::7, 5::7] = Crop_trans[..., 1::6, 5::6].cuda()
            out[..., 1::7, 6::7] = Crop_trans[..., 1::6, 5::6].cuda()

            # 第三排的数据
            out[..., 2::7, 0::7] = Crop_trans[..., 2::6, 0::6].cuda()
            out[..., 2::7, 1::7] = Crop_trans[..., 2::6, 1::6].cuda()
            out[..., 2::7, 2::7] = Crop_trans[..., 2::6, 2::6].cuda()
            out[..., 2::7, 3::7] = Crop_trans[..., 2::6, 3::6].cuda()
            out[..., 2::7, 4::7] = Crop_trans[..., 2::6, 4::6].cuda()
            out[..., 2::7, 5::7] = Crop_trans[..., 2::6, 5::6].cuda()
            out[..., 2::7, 6::7] = Crop_trans[..., 2::6, 5::6].cuda()

            # 第四排的数据
            out[..., 3::7, 0::7] = Crop_trans[..., 3::6, 0::6].cuda()
            out[..., 3::7, 1::7] = Crop_trans[..., 3::6, 1::6].cuda()
            out[..., 3::7, 2::7] = Crop_trans[..., 3::6, 2::6].cuda()
            out[..., 3::7, 3::7] = Crop_trans[..., 3::6, 3::6].cuda()
            out[..., 3::7, 4::7] = Crop_trans[..., 3::6, 4::6].cuda()
            out[..., 3::7, 5::7] = Crop_trans[..., 3::6, 5::6].cuda()
            out[..., 3::7, 6::7] = Crop_trans[..., 3::6, 5::6].cuda()

            # 第五排的数据
            out[..., 4::7, 0::7] = Crop_trans[..., 4::6, 0::6].cuda()
            out[..., 4::7, 1::7] = Crop_trans[..., 4::6, 1::6].cuda()
            out[..., 4::7, 2::7] = Crop_trans[..., 4::6, 2::6].cuda()
            out[..., 4::7, 3::7] = Crop_trans[..., 4::6, 3::6].cuda()
            out[..., 4::7, 4::7] = Crop_trans[..., 4::6, 4::6].cuda()
            out[..., 4::7, 5::7] = Crop_trans[..., 4::6, 5::6].cuda()
            out[..., 4::7, 6::7] = Crop_trans[..., 4::6, 5::6].cuda()

            # 第六排的数据
            out[..., 5::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
            out[..., 5::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
            out[..., 5::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
            out[..., 5::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
            out[..., 5::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
            out[..., 5::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
            out[..., 5::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()

            # 第七排的数据
            out[..., 6::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
            out[..., 6::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
            out[..., 6::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
            out[..., 6::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
            out[..., 6::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
            out[..., 6::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
            out[..., 6::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()

            # out = out


        #  下面直接对特征图进行裁切
        if np.size(x, 2) == 14:
            # x = x
            # Crop_trans = x[..., 1:-1, 1:-1]      #a 这个操作是切除最外层的一个皮      直接切片索引就ok
            Crop_trans = x[..., 1:-1, 1:-1]      #a 这个操作是切除最外层的一个皮      直接切片索引就ok
            b, c, h, w = Crop_trans.shape
            out = torch.zeros((b, c, h * 7 // 6, w * 7 // 6))
            # 第一排的数据，6个值变成7个值
            out[..., 0::7, 0::7] = Crop_trans[..., 0::6, 0::6].cuda()
            out[..., 0::7, 1::7] = Crop_trans[..., 0::6, 1::6].cuda()
            out[..., 0::7, 2::7] = Crop_trans[..., 0::6, 2::6].cuda()
            out[..., 0::7, 3::7] = Crop_trans[..., 0::6, 3::6].cuda()
            out[..., 0::7, 4::7] = Crop_trans[..., 0::6, 4::6].cuda()
            out[..., 0::7, 5::7] = Crop_trans[..., 0::6, 5::6].cuda()
            out[..., 0::7, 6::7] = Crop_trans[..., 0::6, 5::6].cuda()

            # 第二排的数据
            out[..., 1::7, 0::7] = Crop_trans[..., 1::6, 0::6].cuda()
            out[..., 1::7, 1::7] = Crop_trans[..., 1::6, 1::6].cuda()
            out[..., 1::7, 2::7] = Crop_trans[..., 1::6, 2::6].cuda()
            out[..., 1::7, 3::7] = Crop_trans[..., 1::6, 3::6].cuda()
            out[..., 1::7, 4::7] = Crop_trans[..., 1::6, 4::6].cuda()
            out[..., 1::7, 5::7] = Crop_trans[..., 1::6, 5::6].cuda()
            out[..., 1::7, 6::7] = Crop_trans[..., 1::6, 5::6].cuda()

            # 第三排的数据
            out[..., 2::7, 0::7] = Crop_trans[..., 2::6, 0::6].cuda()
            out[..., 2::7, 1::7] = Crop_trans[..., 2::6, 1::6].cuda()
            out[..., 2::7, 2::7] = Crop_trans[..., 2::6, 2::6].cuda()
            out[..., 2::7, 3::7] = Crop_trans[..., 2::6, 3::6].cuda()
            out[..., 2::7, 4::7] = Crop_trans[..., 2::6, 4::6].cuda()
            out[..., 2::7, 5::7] = Crop_trans[..., 2::6, 5::6].cuda()
            out[..., 2::7, 6::7] = Crop_trans[..., 2::6, 5::6].cuda()

            # 第四排的数据
            out[..., 3::7, 0::7] = Crop_trans[..., 3::6, 0::6].cuda()
            out[..., 3::7, 1::7] = Crop_trans[..., 3::6, 1::6].cuda()
            out[..., 3::7, 2::7] = Crop_trans[..., 3::6, 2::6].cuda()
            out[..., 3::7, 3::7] = Crop_trans[..., 3::6, 3::6].cuda()
            out[..., 3::7, 4::7] = Crop_trans[..., 3::6, 4::6].cuda()
            out[..., 3::7, 5::7] = Crop_trans[..., 3::6, 5::6].cuda()
            out[..., 3::7, 6::7] = Crop_trans[..., 3::6, 5::6].cuda()

            # 第五排的数据
            out[..., 4::7, 0::7] = Crop_trans[..., 4::6, 0::6].cuda()
            out[..., 4::7, 1::7] = Crop_trans[..., 4::6, 1::6].cuda()
            out[..., 4::7, 2::7] = Crop_trans[..., 4::6, 2::6].cuda()
            out[..., 4::7, 3::7] = Crop_trans[..., 4::6, 3::6].cuda()
            out[..., 4::7, 4::7] = Crop_trans[..., 4::6, 4::6].cuda()
            out[..., 4::7, 5::7] = Crop_trans[..., 4::6, 5::6].cuda()
            out[..., 4::7, 6::7] = Crop_trans[..., 4::6, 5::6].cuda()

            # 第六排的数据
            out[..., 5::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
            out[..., 5::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
            out[..., 5::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
            out[..., 5::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
            out[..., 5::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
            out[..., 5::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
            out[..., 5::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()

            # 第七排的数据
            out[..., 6::7, 0::7] = Crop_trans[..., 5::6, 0::6].cuda()
            out[..., 6::7, 1::7] = Crop_trans[..., 5::6, 1::6].cuda()
            out[..., 6::7, 2::7] = Crop_trans[..., 5::6, 2::6].cuda()
            out[..., 6::7, 3::7] = Crop_trans[..., 5::6, 3::6].cuda()
            out[..., 6::7, 4::7] = Crop_trans[..., 5::6, 4::6].cuda()
            out[..., 6::7, 5::7] = Crop_trans[..., 5::6, 5::6].cuda()
            out[..., 6::7, 6::7] = Crop_trans[..., 5::6, 5::6].cuda()

            # out = out

        out = out.cuda()

        out = self.Inception_c(out)
        return out

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()

        layer = _TransitionLayer(num_input_features, num_output_features)
        # layer = generate_inception_module(num_input_features, num_output_features, 1, InceptionC)
        self.add_module('_TransitionLayer', layer)


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




# # https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
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


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
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
        self.trans = MHSA(growth_rate//2, 28, 28)
        self.drop_rate = drop_rate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = nn.ConvTranspose2d(in_channels=growth_rate//2,out_channels=growth_rate//2,kernel_size=3,stride=2,padding=1,output_padding=1),


    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        x_A, x_B = new_features.split([int(new_features.shape[1] / 2) * 1, int(new_features.shape[1] / 2) * 1], dim=1)
        new_features = self.bn3(x_A)
        new_features = self.relu3(new_features)
        x_A = self.conv3(new_features)
        x_B = self.pool(x_B)
        x_B = self.trans(x_B)
        x_B = self.up(x_B)
        new_features = torch.cat([x_A, x_B], 1)
        new_features = channel_shuffle(new_features, 2)
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
        self.trans = MHSA(growth_rate//2, 14, 14)
        self.drop_rate = drop_rate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        x_A, x_B = new_features.split([int(new_features.shape[1] / 2) * 1, int(new_features.shape[1] / 2) * 1], dim=1)
        new_features = self.bn3(x_A)
        new_features = self.relu3(new_features)
        x_A = self.conv3(new_features)
        x_B = self.pool(x_B)
        x_B = self.trans(x_B)
        x_B = self.up(x_B)
        new_features = torch.cat([x_A, x_B], 1)
        new_features = channel_shuffle(new_features, 2)
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
        self.trans = MHSA(growth_rate//2, 7, 7)
        self.drop_rate = drop_rate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        x_A, x_B = new_features.split([int(new_features.shape[1] / 2) * 1, int(new_features.shape[1] / 2) * 1], dim=1)
        new_features = self.bn3(x_A)
        new_features = self.relu3(new_features)
        x_A = self.conv3(new_features)
        x_B = self.pool(x_B)
        x_B = self.trans(x_B)
        x_B = self.up(x_B)
        new_features = torch.cat([x_A, x_B], 1)
        new_features = channel_shuffle(new_features, 2)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseLayerD(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerD, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(growth_rate//2)
        self.relu3 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        # self.trans = MHSA(growth_rate, 7, 7)
        self.conv3 = nn.Conv2d(growth_rate//2, growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.trans = MHSA(growth_rate//2, 7, 7)
        self.drop_rate = drop_rate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        x_A, x_B = new_features.split([int(new_features.shape[1] / 2) * 1, int(new_features.shape[1] / 2) * 1], dim=1)
        new_features = self.bn3(x_A)
        new_features = self.relu3(new_features)
        x_A = self.conv3(new_features)
        x_B = self.pool(x_B)
        x_B = self.trans(x_B)
        x_B = self.up(x_B)
        new_features = torch.cat([x_A, x_B], 1)
        new_features = channel_shuffle(new_features, 2)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)



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
        # x = torch.cat([x, xA, xB, xC, xD, xE, xF, xG, xH, xI, xJ, xK, xL], 1)
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
# #


def generate_inception_module(input_channels, output_channels, block_num, block):
    layers = nn.Sequential()
    for l in range(block_num):
        layers.add_module("{}_{}".format(block.__name__, l), block(input_channels, output_channels))
        # input_channels = output_channels
    return layers


class SEDenseNet(nn.Module):

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels, output_channels))
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
                 num_init_features=64, bn_size=1, drop_rate=0, num_classes=1000):       #经过计算，为了残差连接能实现，把这里的bn_size改为1合适，通道数全部改为32了     r如果需要加大必须要保证参数同步增加倍数

        super(SEDenseNet, self).__init__()

        self.Inception_c = self._generate_inception_module(3, 64, 1, InceptionA_1_1_res)

        self.First = self._generate_inception_module(64, 64, 1, InceptionA_1_4_res)


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
        x = self.Inception_c(x)
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
        X = X
        model = model
    model.eval()
    with torch.no_grad():
        output = model(X)
        print(output.shape)


if __name__ == "__main__":
    test_se_densenet()

