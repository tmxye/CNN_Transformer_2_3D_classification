# coding:utf8
from torch import nn
from .basic_module import BasicModule

import torch
import os
home = os.path.expanduser('~')

# __all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(BasicModule):
    """
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    """

    def __init__(self, num_classes=1000):
    # def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()

        self.model_name = 'alexnet'

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
        x = self.classifier(x)
        return x

    #     self.classifier = nn.Sequential(
    #         nn.Dropout(),
    #         nn.Linear(256 * 6 * 6, 4096),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(inplace=True),
    #         # nn.Linear(4096, num_classes),
    #     )
    #     self.gategory = nn.Linear(4096, num_classes)
    #
    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     x = self.gategory(x)
    #     return x




def alexnet(pretrained=False, num_classes=1000, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(num_classes=num_classes, **kwargs)


    # by YXY
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls['alexnet'],
        #                                       progress=progress)
        # from torchvision.models import alexnet
        # alex = alexnet(pretrained=True)
        # pretrained_dict = alex.state_dict()

        state_dict = torch.load(home + '/weights/alexnet.pth')
        print("预训练从本地加载成功")
        # pretrained_dict = state_dict
        # weight_0 = pretrained_dict['features.3.weight']
        # bias_0 = pretrained_dict['features.3.bias']
        # print(weight_0.shape)
        # print(bias_0.shape)
        num_classes = 2
        # fc_features = model.classifier.in_features
        # model.classifier = nn.Linear(fc_features, num_classes)


        model.load_state_dict(state_dict)

    # num_classes = 2
    # model_ft = model
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model




#
#
# def make_weights_for_balanced_classes(images, nclasses):
#     count = [0] * nclasses
#     for item in images:
#         count[item[1]] += 1
#     weight_per_class = [0.] * nclasses
#     N = float(sum(count))
#     for i in range(nclasses):
#         weight_per_class[i] = N/float(count[i])
#     weight = [0] * len(images)
#     for idx, val in enumerate(images):
#         weight[idx] = weight_per_class[val[1]]
#     return weight
#
#
# root='./data_2/'
# from PIL import Image
# # from openpyxl.drawing.image import Image
# import torch
#
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# num_classes = 2
#
# class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
#     def __init__(self, root, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
#         super(MyDataset, self).__init__()
#         # fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
#         fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
#         imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
#         for line in fh:  # 按行循环txt文本中的内容
#             line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
#             words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
#             imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
#             # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
#
#         # root = './data_1/'
#         # root = './original_dataset/'
#
#         fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
#         # img = Image.open(root + fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
#         img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
#
#         if self.transform is not None:
#             img = self.transform(img)  # 是否进行transform
#         return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
#
#     def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
#         return len(self.imgs)
#
# train_data = MyDataset(datatxt=root + 'data_2_train.txt', transform = transforms.Compose([transforms.Resize((224, 224)),  # 缩放
#                 transforms.RandomCrop(224, padding=4),  # 裁剪
#                 # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
#                 transforms.ToTensor(),  # 转为张量，同时归一化
#                 # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化      #原文是transforms.Normalize(norm_mean, norm_std),
#                 ]), root='./data_1/')
#
# weights = make_weights_for_balanced_classes(train_data.imgs, num_classes)
# weights = torch.DoubleTensor(weights)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
#
#
# def weight_init(m):
# # 使用isinstance来判断m属于什么类型
#     if isinstance(m, nn.Conv2d):
#         import math
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#     elif isinstance(m, nn.BatchNorm2d):
# # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()