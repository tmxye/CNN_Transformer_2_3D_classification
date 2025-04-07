import torch.nn as nn
from .utils import load_state_dict_from_url

import torch
from models_32 import LOCAL_PRETRAINED, model_urls

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
    # def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

import re
import torch.utils.model_zoo as model_zoo
import os
home = os.path.expanduser('~')

# def alexnet(pretrained=False, progress=True, **kwargs):
def alexnet(pretrained=False, num_classes=1000, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(num_classes=num_classes, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['alexnet'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    #
    #
    # by YXY
    # if not test:
    #     # if LOCAL_PRETRAINED['alexnet'] == None:
    #     #     state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=True)
    #     # else:
    #     state_dict = torch.load(home + '/weights/alexnet.pth')
    #     model.load_state_dict(state_dict)

    # by YXY
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls['alexnet'],
        #                                       progress=progress)
        state_dict = torch.load(home + '/weights/alexnet.pth')
        print("预训练从本地装载成功")
        print(state_dict)
        model.load_state_dict(state_dict)

    return model

