# -*- coding:utf-8 -*-
# @time :2020.03.17
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch
import torchvision
from models_32 import resnet18, resnet34, resnet50, resnet101, densenet121, densenet161, densenet169, densenet201, resnet50, mobilenet_v2
from models_32 import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl
from models_32 import EfficientNet
from models_32 import LOCAL_PRETRAINED, model_urls

# def Resnet18(num_classes, test=False):
def Resnet18(num_classes=2, pretrained=False, test=False):
    num_classes = num_classes
    model = resnet18()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnet18'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet18'])
            print('模型从本地导入成功')
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

# def Resnet18(num_classes, test=False):
def Resnet34(pretrained=False, test=False):
    num_classes = 2
    model = resnet34()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnet34'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet34'])
            print('模型从本地导入成功')
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

# def Resnet50(num_classes, test=False):
def Resnet50(pretrained=False, test=False):
    num_classes = 2
    model = resnet50()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnet50'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
            print('模型从本地导入成功')
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


# def Resnet101(num_classes, test=False):
def Resnet101(pretrained=False, test=False, num_classes=1000):
    num_classes = num_classes
    model = resnet101()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnet101'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet101'])
            print('模型从本地导入成功')
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


import models_32
# def Resnet101(num_classes, test=False):
def Resnet101_SK(pretrained=False, test=False, num_classes=1000):
    num_classes = num_classes
    model = models_32.vision.resnet_SK.resnet101_SK()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnet101'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet101'])
            print('模型从本地导入成功')
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Resnet18_SK(num_classes=2, pretrained=False, test=False):
    num_classes = num_classes
    # model = resnet18()
    model = models_32.vision.resnet_SK.resnet18_SK()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnet18'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet18'])
            print('模型从本地导入成功')
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

# def Resnet101(num_classes, test=False):
def Resnet152(pretrained=False, test=False, num_classes=1000):
    num_classes = 2
    model = resnet101()
    if pretrained:
            if not test:
                if LOCAL_PRETRAINED['resnet152'] == None:
                    state_dict = load_state_dict_from_url(model_urls['resnet152'], progress=True)
                else:
                    state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet152'])
                print('模型从本地导入成功')
                model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


# def Resnext101_32x8d(num_classes, test=False):
def Resnext101_32x8d(pretrained=False, test=False, num_classes=1000):
    num_classes = num_classes
    model = resnext101_32x8d_wsl()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnext101_32x8d'] == None:
                print("模型已经正在从网络服务器上下载")
                state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'], progress=True)
                print("模型已经从网络下载到本地，装载成功")
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnext101_32x8d'])
                print("模型已经从本地装载成功")
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


# def Resnext101_32x16d(num_classes, test=False):
def Resnext101_32x16d(pretrained=False, test=False, num_classes=1000):
    num_classes = 2
    model = resnext101_32x16d_wsl()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnext101_32x16d'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnext101_32x16d'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnext101_32x16d'])
                print("模型已经从本地装载成功")
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


# def Resnext101_32x32d(num_classes, test=False):
def Resnext101_32x32d(pretrained=False, test=False, num_classes=1000):
    num_classes = 2
    model = resnext101_32x32d_wsl()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnext101_32x32d'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnext101_32x32d'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnext101_32x32d'])
                print("模型已经从本地装载成功")
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


# def Resnext101_32x48d(num_classes, test=False):
def Resnext101_32x48d(pretrained=False, test=False, num_classes=1000):
    num_classes = 2
    model = resnext101_32x48d_wsl()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['resnext101_32x48d'] == None:
                state_dict = load_state_dict_from_url(model_urls['resnext101_32x48d'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnext101_32x48d'])
                print("模型已经从本地装载成功")
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


# def Densenet121(num_classes, test=False):
def Densenet121(pretrained=False, test=False):
    num_classes = 2
    model = densenet121()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['densenet121'] == None:
                state_dict = load_state_dict_from_url(model_urls['densenet121'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['densenet121'])

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k,v in state_dict.items():
                # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
                # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
                # 直接利用if语句筛选不一致的键
                ### 修正键值的不对应
                if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
                    k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
                # print(k)
                else:
                    pass
                new_state_dict[k] = v
            print('模型从本地导入成功')
            model.load_state_dict(new_state_dict)
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


# def Densenet121(num_classes, test=False):
def Densenet161(pretrained=False, test=False):
    num_classes = 2
    model = densenet161()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['densenet161'] == None:
                state_dict = load_state_dict_from_url(model_urls['densenet161'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['densenet161'])

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k,v in state_dict.items():
                # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
                # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
                # 直接利用if语句筛选不一致的键
                ### 修正键值的不对应
                if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
                    k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
                # print(k)
                else:
                    pass
                new_state_dict[k] = v
            print('模型从本地导入成功')
            model.load_state_dict(new_state_dict)
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model

def Densenet169(pretrained=False, test=False):
# def Densenet169(num_classes, test=False):
    num_classes = 2
    model = densenet169()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['densenet169'] == None:
                state_dict = load_state_dict_from_url(model_urls['densenet169'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['densenet169'])

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k,v in state_dict.items():
                # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
                # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
                # 直接利用if语句筛选不一致的键
                ### 修正键值的不对应
                if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
                    k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
                # print(k)
                else:
                    pass
                new_state_dict[k] = v
            print('模型从本地导入成功')
            model.load_state_dict(new_state_dict)
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Densenet201(pretrained=False, test=False):
# def Densenet169(num_classes, test=False):
    num_classes = 2
    model = densenet201()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['densenet201'] == None:
                state_dict = load_state_dict_from_url(model_urls['densenet201'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['densenet201'])

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k,v in state_dict.items():
                # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
                # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
                # 直接利用if语句筛选不一致的键
                ### 修正键值的不对应
                if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
                    k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
                # print(k)
                else:
                    pass
                new_state_dict[k] = v
            print('模型从本地导入成功')
            model.load_state_dict(new_state_dict)
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model





# def Mobilenetv2(num_classes, test=False):
def Mobilenetv2(pretrained=False, test=False):
    num_classes = 2
    model = mobilenet_v2()
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED['moblienetv2'] == None:
                state_dict = load_state_dict_from_url(model_urls['moblienetv2'], progress=True)
            else:
                state_dict = state_dict = torch.load(LOCAL_PRETRAINED['moblienetv2'])
            model.load_state_dict(state_dict)
        print('模型从本地加载成功')
    print(model.state_dict().keys())
    fc_features = model.classifier[1].in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


# def Efficientnet(model_name, num_classes, test = False):
def Efficientnet(pretrained=False, model_name='efficientnet-b0', test = False):
    num_classes = 2
    '''
    model_name :'efficientnet-b0', 'efficientnet-b1-7'
    '''
    model = EfficientNet.from_name(model_name)
    if pretrained:
        if not test:
            if LOCAL_PRETRAINED[model_name] == None:
                state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
            else:
                state_dict = torch.load(LOCAL_PRETRAINED[model_name])
            print('模型权重从本地加载成功')
            model.load_state_dict(state_dict)
    fc_features = model._fc.in_features
    model._fc = nn.Linear(fc_features, num_classes)
    return model

