import torch
import torch.nn as nn
from .utils import load_state_dict_from_url


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

#   ####CSDN上的VGG16 https://blog.csdn.net/weixin_43273742/article/details/106869191
# class VGG16(nn.Module):
#     def __init__(self, num_classes):
#         super(VGG16, self).__init__()
#
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         num_classes = 2
#         self.fc_layers = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, num_classes),
#         )
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0.01)
#                 nn.init.constant_(m.bias, 0)



# https://blog.csdn.net/qq_36560894/article/details/104923543       VGG实现猫狗分类的代码
class VGG19(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        # 因为前面可以用预训练模型参数，所以单独把最后一层提取出来
        self.classifier2 = nn.Linear(4096, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # torch.flatten 推平操作
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 查看模型结构
model = VGG19(num_classes=1000, init_weights=True)
# print(model)





#   #####这是一个课程里面的VGG        https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test3_vggnet/model.py
# class VGG(nn.Module):
#     def __init__(self, features, num_classes=1000, init_weights=False):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(512*7*7, 2048),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 2048),
#             nn.ReLU(True),
#             nn.Linear(2048, num_classes)
#         )
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         # N x 3 x 224 x 224
#         x = self.features(x)
#         # N x 512 x 7 x 7
#         x = torch.flatten(x, start_dim=1)
#         # N x 512*7*7
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 # nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#
# def make_features(cfg: list):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == "M":
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             layers += [conv2d, nn.ReLU(True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
#
# cfgs = {
#     'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#
#
# def vgg(model_name="vgg16", **kwargs):
#     try:
#         cfg = cfgs[model_name]
#     except:
#         print("Warning: model number {} not in cfgs dict!".format(model_name))
#         exit(-1)
#     model = VGG(make_features(cfg), **kwargs)
#     return model


# 这是原始的，但是不好使用
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
    # def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()

        # num_classesA = num_classes
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes=num_classes),
            # nn.Linear(4096, 1000),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}




def _vgg(arch, cfg, batch_norm, pretrained, progress, num_classes, **kwargs):

#     #https://blog.csdn.net/kongkongqixi/article/details/88182900  发现没用

    # num_classes = 1000

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        print("从本地导入权重成功")

        # for name, weights in state_dict.items():
        #     print(len(weights.size()))
        #     print(name, '1', weights.squeeze(0).size())       #           查看转化后的模型名字和权重维度
        #     if len(weights.size()) == 999:       # 判断需要修改维度的条件
        #         for i in (998):
        #             print(i)
        #             state_dict[name] = weights.squeeze(i)           # 去掉维度0，把(1, 128)    转为(128)
        #             得到的结果如下：
        #     classifier.3.bias 1 torch.Size([4096])
        # classifier.6.weight 1 torch.Size([1000, 4096])
        # classifier.6.bias 1 torch.Size([1000])



        for name, weights in state_dict.items():
            # print(len(weights.size()))
            # print(name, '1', weights.squeeze(0).size())       #           查看转化后的模型名字和权重维度
        #     if len(weights.size()) == 999:       # 判断需要修改维度的条件
        #         for i in (998):
        #             print(i)
        #             state_dict[name] = weights.squeeze(i)           # 去掉维度0，把(1, 128)    转为(128)

        # for name, weights in state_dict.items():
        #     model.classifier._modules['6'] = nn.Linear(4096, 2)
        #
        # model.classifier._modules['6'] = nn.Linear(4096, 2)
        # model.classifier[6].out_features = 2
        # model.classifier[6].weight = torch.nn.Parameter(torch.randn(1, 4096))
        # model.classifier[6].bias = torch.nn.Parameter(torch.ones(1))
        #     model.features = nn.Sequential(*list(model.children())[:-1])
        #     model.classifier = nn.Sequential(nn.Linear(512, 1))
        #     for p in model.features.parameters():
        #         p.requires_grad = False
        #     model.classifier._modules['7'] = nn.Linear(1000, 2)

        #     model.classifier.out_features = 2
        #     model.classifier[6].weight = torch.nn.Parameter(torch.randn(1, 4096))
        #     model.classifier[6].bias = torch.nn.Parameter(torch.ones(1))

            # print(model.classifier._modules['7'])
            print(name, '1', weights.squeeze(0).size())
            # model.fc = nn.Linear(512, 2)


        # for param in model.parameters():
        #     param.requires_grad = False
        #     # Replace the last fully-connected layer
        #     # Parameters of newly constructed modules have requires_grad=True by default
        #     model.classifier._modules['6'] = nn.Linear(4096, 2)

        # model.fc = nn.Linear(4096, 2)  # assuming that the fc7 layer has 512 neurons, otherwise change it
        # model.cuda()

        model.load_state_dict(state_dict)

    #
    # n_feats = 2  # whatever is your number of output features
    # last_item_index = len(model.classifier) - 1
    # old_fc = model.classifier.__getitem__(last_item_index)
    # new_fc = nn.Linear(in_features=old_fc.in_features, out_features=n_feats, bias=True)
    # model.classifier.__setitem__(last_item_index, new_fc)

    # # creating a mlp model
    # import keras
    # from keras.layers import Dense, Activation
    #
    # model.add(Dense(1000, input_dim=25088, activation='relu', kernel_initializer='uniform'))
    # keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)
    #
    # model.add(Dense(500, input_dim=1000, activation='sigmoid'))
    # keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)
    #
    # model.add(Dense(150, input_dim=500, activation='sigmoid'))
    # keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)
    #
    # model.add(Dense(units=10))
    # model.add(Activation('softmax'))
    #
    # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# 没法这样解决
    # model_ft = model
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # fc_features = model.fc.in_features
    # model.fc = nn.Linear(fc_features, num_classes)
    return model



def vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)
    # return VGG16(num_classes=2, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, num_classes=1000, **kwargs):
# def vgg19(batch_norm, pretrained=False, progress=True, **kwargs ):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _vgg('vgg19', 'E', False, pretrained, progress, num_classes, **kwargs)
    return _vgg('vgg19', 'E', False, pretrained= progress, progress = progress, num_classes = num_classes, **kwargs)


    # if pretrained:
    #     kwargs['init_weights'] = False
    # cfg = 'E'
    # arch = 'vgg19'
    # model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    #
    # if pretrained:
    #     # CPU 或者 GPU
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     # 初始化网络,加载预训练模型
    #     model_dict = model.state_dict()
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    #     model_dict.update(new_state_dict)
    #     model.load_state_dict(new_state_dict)

    # # 查看GPU可用情况
    # if torch.cuda.device_count() > 1:
    #     print('We are using', torch.cuda.device_count(), 'GPUs!')
    #     model = nn.DataParallel(model)
    # model.to(device)
    # return model


    # return VGG19(num_classes=2, init_weights=True)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
