import torch

from pathlib import Path
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# by YXY
# torch.cuda.empty_cache()

# torch.no_grad()
# https://zhuanlan.zhihu.com/p/193102483    在测试阶段使用with torch.no_grad()可以对整个网络都停止自动求导，可以大大加快速度，也可以使用大的batch_size来测试# #当然，也可以不使用with torch.no_grad
# model.eval()      # with  torch.no_grad():

# from utils_25_Stop import Trainer
from utils_25_Stop_A import Trainer
from LOSS.a_focal_loss_pytorch.focalloss import FocalLoss
from LOSS.c_Multiclass_LinearSVM_with_SGD.multiClassHingeLossYXY import multiClassHingeLossYXY
from LOSS.LossYXY import LossYXY

# from dataset_YXY import get_dataloaderYXY
from dataset_YXY_N import get_dataloaderYXY
# from dataset_YXY_N1 import get_dataloaderYXY
# from dataset_YXY_Small import get_dataloaderYXY
# from dataset_YXY_56x56 import get_dataloaderYXY
# from dataset_YXY_CT_56x56_224 import get_dataloaderYXY
# from cifar10 import get_dataloader
from torchstat import stat
from thop import profile
from thop import clever_format


# from models_32 import Resnet50, Resnet101, Resnext101_32x8d,Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, Efficientnet, Resnext101_32x32d, Resnext101_32x48d
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized错误提示 https://blog.csdn.net/jialibang/article/details/107392240
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import core

from classification_choice_YXY_2 import get_model
# 这行代码使用之后，程序就不进行优化了。会导致3G显存变为7.5G显存
# torch.backends.cudnn.enabled = False


# 0 AlexNet     155 ZFNet   17 vgg19    21 resnet18  234  resnet101   331 googlenet        943 se_resnet101      4201 se_densenet121     4023    densenet121
#451 densenet121  4000 se_densenet_w_block_a      481 se_densenet_full      491 se_densenet_full_in_loop        411 se_densenet_w_block     471 se_densenet
# dense 451         densenet_dropout=0.2的是 4510    seden 4000         n1_1n 4001     ac 4005  d 4006    e 4007      acegh_a 4012          ce 4013             ce 56 4015         ce 56_3*3 4016
# densenet  ce 56_3*3 4017
#  236   Resnext101_32x8d       80  efficientnet-b0     943 se_resnet101        947 se_resnext101

model_index = 106
# model_index = 4204
# lr = 5e-5
# lr = 5e-3
lr = 1e-2
# model_index = 441     # DenseNet
# model_index = 4510
# model = get_model(model_index = model_index)

import hiddenlayer as h


# 网络可视化显示
from tensorboardX import SummaryWriter
# import torch
# from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
from PIL import Image
# import tensorflow as tf


    # 对pytorch的训练进行优化，代码如下，优化了好几处，但是有个这个位置没进行优化,如下：
    # 全局变量累加导致内存泄漏，如loss变量的处理。这里要注意的是，一定要先和GPU detach了，仅仅使用loss的数值，不然添加的实际上是整个计算图。当然除了loss，其他的变量问题基本上都是出现内存泄露的原因呀。
    #       epoch_loss += loss.detach().item()
    # https: // blog.csdn.net / qq_39388410 / article / details / 108027769
    # 使用inplace操作， 比如relu 可以使用 inplace=True。当设置为True时，我们在通过relu()计算时的得到的新值不会占用新的空间而是直接覆盖原来的值，这也就是为什么当inplace参数设置为True时可以节省一部分内存的缘故。
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

import torch
import torch.backends.cudnn
import torch.nn as nn
import os
# def main(batch_size, baseline, reduction):  #原始的
def main(batch_size, reduction, gpu):
    from utils_tool.create_folder import initialize_folders
    success = initialize_folders()
#     文件内容删除
    with open(r"./visual/state_time.txt", "a+", encoding="utf-8") as txt:
        txt.truncate(0)        #   清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
    with open(r"./visual/state.txt", "a+", encoding="utf-8") as txt:
        txt.truncate(0)        #   清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
    with open(r"./visual/save_epoch_evaluating_indicator.txt", "a+", encoding="utf-8") as txt:
        txt.truncate(0)        #   清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
#     文件内容删除
    with open(r"./visual/visual_val/state_time.txt", "a+", encoding="utf-8") as txt:
        txt.truncate(0)        #   清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
    with open(r"./visual/visual_val/state.txt", "a+", encoding="utf-8") as txt:
        txt.truncate(0)        #   清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
    with open(r"./visual/visual_val/save_epoch_evaluating_indicator.txt", "a+", encoding="utf-8") as txt:
        txt.truncate(0)        #   清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
    # with open("model_visual.txt", "a", encoding="utf-8") as f:
    #     f.truncate()        #   清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
# 删除文件的代码如下：import os   # os.remove(r'test.txt')

# def main(batch_size):
#     train_loader, test_loader = get_dataloader(batch_size)
    train_loader, val_loader, test_loader = get_dataloaderYXY(batch_size)

    # 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销
    torch.backends.cudnn.benchmark = True

    # 如果在python内调用pytorch有可能显存和GPU占用不会被自动释放，此时需要加入如下代码来删除一些不需要的变量
    torch.cuda.empty_cache()

    print('模型开始导入')
    # create modelW
    model = get_model(model_index = model_index, num_classes = 2)
    # model = get_model(model_index = model_index, num_classes = 1000)
    # model = get_model(model_index = model_index)
    print('模型导入成功')


    # # 这也是一个优化语句，注释之后效果可能会更好点
    model.apply(inplace_relu)

    # 之前写的，不知道可不可以运行
    # model = model.cuda()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # model.to(device)
    # # model.to(device)  # 使用序号为1的GPU


    # # 模型可视化，参数显示
    # # 1、查看网络的模型，和参数的信息情况
    # print(model)
    # # 打印参数的信息
    # # for parameters in model.parameters():
    # #     print(parameters)
    # # 打印参数的名字以及参数的数据形状
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())

    # 2、查看参数的数量，以及所占内存的大小
    # # # 方法1：使用stat函数
    # from torchstat import stat
    # stat(model, (3, 224, 224))

    # # # #   方法2、使用summary函数       方法一需要在进gpu之前，不使用
    # from torchsummary import summary
    # # model = model.to('cuda:0')
    # model = model.cuda()
    # # summary(model, input_size=(3, 224, 224), batch_size=1, device="cuda")
    # summary(model, input_size=(3, 224, 224), batch_size=1, device="cuda")

    '''
    下面的代码是计算量，但使用之后没法运行其它代码module._DenseBlockD.DenseLayerP.conv3.softmax.total_params".(yxy21) Y:\05DenseNet_Dense_SA>
    因此，在不需要的时候暂时注销
    '''
    # # # https://github.com/Swall0w/torchstat  借助工具自动计算FLOPs值
    # stat(model, (3, 224, 224))      # 出现因为双卡而导致的问题，RuntimeError: module must have its parameters and buffers on device cuda:0 (device_ids[0]) but found one of them on device: cpu
    # 还是出现问题    RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same。   这个是由于输入类型需要将tensor转换成了CUDA 类型 https://blog.csdn.net/qq_27261889/article/details/86575033
    #  # #为了方便，直接使用下面这个代码对计算量进行计算，input直接改
    # # # 推荐两个工具pytorch-OpCounter和torchstat，调用方法很简单，具体看github项目里面的readme。   https://github.com/Lyken17/pytorch-OpCounter
    model = model.cuda()
    input = torch.randn(1, 3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    macs, params = profile(model, inputs=(input, ))     # 也出现一样的双卡问题，RuntimeError: module must have its parameters and buffers on device cuda:0 (device_ids[0]) but found one of them on device: cpu
    print("Params(M)是\n", params)
    print("MACs(G)是\n", macs)
    macs, params = clever_format([macs, params], "%.3f")
    print("提高输出可读性Params(M)是\n", params)
    print("提高输出可读性MACs(G)是\n", macs)


    # by YXY        多gpu并行
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        gpus = list(range(len(gpu.split(','))))
    else:
        gpus = [0]  # [1,2]
    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()


    # # 这里可视化方案删除了很多，具体可以查看09classfication文件
    #
    # # # 可视化方案4： 通过HiddenLayer可视化网络
    # # vis_graph = h.build_graph(model, torch.zeros([1 ,3, 224, 224]))   # 获取绘制图像的对象
    # # vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
    # # vis_graph.save("./visual/网络结构.png")   # 保存图像的路径
    # # print('模型图像保存成')
    # # # 可以打印出来，但是显示的特别长
    #
    # # # 可视化方案5：通过PyTorchViz可视化网络        这个效果比5的效果好
    # from torchviz import make_dot
    # x = torch.randn(1, 3, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
    # MyConvNet = model
    # y = MyConvNet(x)    # 获取网络的预测值
    # MyConvNetVis = make_dot(y, params=dict(list(MyConvNet.named_parameters()) + [('x', x)]))
    # # MyConvNetVis.format = "png"
    # MyConvNetVis.format = "pdf"
    #  # 指定文件生成的文件夹
    # MyConvNetVis.directory = "D:/01DenseNet_YXY_IncepV4C_1nn1_SN_DYRelu_Residual/visual/model/"
    #  # 生成文件
    # MyConvNetVis.view()
    # print('网络模型文件生成成功')
    #
    # # 使用Visdom进行可视化，暂时先放着   可视化的需求已经完成，有时间再继续编写，主要参考网页：https://blog.csdn.net/weixin_39671631/article/details/110978200


# print_model_summary(network=model) 这句是tf的语句，这个好像用不了

    '''
    这里是预训练模型的加载，可以加载模型的部分参数
    '''
    # #     # http://www.cppcns.com/jiaoben/python/220506.html
    # # '''     直接加载预训练模型       '''
    # 如果我们使用的模型和原模型完全一样，那么我们可以直接加载别人训练好的模型：
    # model.load_state_dict(torch.load("weights/model_epoch_66.pth"))
    # model.load_state_dict(torch.load("model_epoch_4420.pth"))
    # model.load_state_dict(torch.load("weights/model_epoch_5e_4.pth"))
    #       # 还有第二种加载方法：
    # model = torch.load("my_resnet.pth")
    #
    # # '''   加载部分预训练模型     方法一    '''
    # # 其实大多数时候我们需要根据我们的任务调节我们的模型，所以很难保证模型和公开的模型完全一样，但是预训练模型的参数确实有助于提高训练的准确率，为了结合二者的优点，就需要我们加载部分预训练模型。
    # # pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
    # # 将这里的 改写为自己的预训练结果
    # # modelA = torch.load("weights/model_epoch_66.pth")
    # # pretrained_dict = modelA.load_state_dict()      # 出现问题  pretrained_dict = modelA.load_state_dict()  AttributeError: 'collections.OrderedDict' object has no attribute 'load_state_dict'
    # # 这里想到一个思路，就是把原始的网络模型加载进来作为modelA，但是模型的内容没法加载，那只能仔细研究第一行load_url, 看看能不能model_zoo导入本地pth。https://blog.csdn.net/qq_42698422/article/details/100547225
    # pretrained_dict = torch.load("weights/model_epoch_4018.pth")      # 现在就正确了    Y:\05DenseNet_Dense_SA\models_32\build_model.py
    # model_dict = model.state_dict()
    # # 将pretrained_dict里不属于model_dict的键剔除掉
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 更新现有的model_dict
    # model_dict.update(pretrained_dict)
    # # 加载我们真正需要的state_dict
    # model.load_state_dict(model_dict)
    # #
    # # '''   加载部分预训练模型     方法二    '''
    # # # #http://www.zhugaofei.com/index.php/archives/248/   keras 加载预训练模型用于fine-tuning（只加载部分层，前几层）技巧
    # # # model.load_weights('weights/model_epoch_66.pth', by_name=True)
    # # # # 出现问题： 不能并行计算  torch.nn.modules.module.ModuleAttributeError: 'DataParallel' object has no attribute 'load_weights'



    # # 这里是原始的网络模型输入，在上面把网络模型修改了
    # if baseline:
    #     model = densenet121()
    # else:
    #     model = se_densenet121(num_classes=10)

    # optimizer = optim.Adam(params=model.parameters(), lr=1e-2)
    # 这个是原文的densenet，
    # optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)


    # 这个是原文的SEdensenet
    # optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)
    # 这个是原文的SEdensenet，进行改进
    # optimizer = optim.SGD(params=model.parameters(), lr=1e-1, weight_decay=1e-4)
    # optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9, weight_decay=4e-5)
    # optimizer = optim.SGD(params=model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.SGD(params=model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.SGD(params=model.parameters(), lr=6e-2, momentum=0.9, weight_decay=4e-5)
    # optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)       #StepLR(optim,step_size=5 , gamma=0.1)  第五个epoch学习率会先衰减为 lr * (gamma^2)，然后下一个epoch之后才是 lr * gamma.
    #为了解决上面的问题    milestones为一个数组，如 [50,70]. gamma为倍数。如果learning rate开始为0.01 ，则当epoch为50时变为0.001，epoch 为70 时变为0.0001。
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    # optimizer = optim.Adam(params=model.parameters(), lr=1e-3)  # use default
    # # scheduler = optim.lr_scheduler.StepLR(optimizer, 40, 0.1)     #4个负责学习率调整的类：StepLR、ExponentialLR、MultiStepLR和CosineAnnealingLR
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)     #milestones为一个数组，如 [50,70]. gamma为倍数。如果learning rate开始为0.01 ，则当epoch为50时变为0.001，epoch 为70 时变为0.0001。当last_epoch=-1,设定为初始lr。
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], 0.9)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4, 8, 12, 16, 20, 24], 0.7)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], 0.7)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 35, 45, 55], 0.7)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 45], 0.1)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], 0.7)
    # # schedulerA = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], 0.7)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140], 0.9)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140], 0.9)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 45, 50, 55, 60], 0.9)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140], 0.9)
    # # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)        # https://www.cnblogs.com/emperinter/p/14170243.html
    # # schedulerA = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-08)        # https://www.cnblogs.com/emperinter/p/14170243.html
    # schedulerA = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=6, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08)        # https://www.cnblogs.com/emperinter/p/14170243.html
    # # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)        # https://www.cnblogs.com/emperinter/p/14170243.html
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140], 0.9)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 120], 0.1)
    # # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 60], 0.1)


    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 70, 80, 90, 100, 110, 120, 130, 140], 0.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 45, 50, 55, 60], 0.9)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140], 0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140], 0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)        # https://www.cnblogs.com/emperinter/p/14170243.html
    # schedulerA = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-08)        # https://www.cnblogs.com/emperinter/p/14170243.html
    schedulerA = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=6, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08)  # https://www.cnblogs.com/emperinter/p/14170243.html
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)        # https://www.cnblogs.com/emperinter/p/14170243.html
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140], 0.9)

    # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)  # use default




    # 这个是Util25那个网页上的参数
    # optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
    # model = net.train(model, loss_function, optimizer, scheduler, num_epochs = 100)

    # https://www.cnblogs.com/jimchen1218/p/12049746.html   这个网页给的参数，从上到下效果越来越好
    # 这个已经可以满足除了GoogleNet的其他模型了
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)       #step_size：多少个周期后学习率发生改变  gamma:学习率如何改变

    # 现在在学习下面几个之后再进行这个的编写，试试看看效果会不会更好
    # optimizer = torch.optim.Adam(model.parameters(), lr=4*1e-4, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)       #step_size：多少个周期后学习率发生改变  gamma:学习率如何改变



# 5 通过余弦退火的方式来减少学习率     相关论文：SGDR: Stochastic Gradient Descent with Warm Restarts       https://arxiv.org/abs/1608.03983
# epochs = 60
# optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = (epochs // 9) + 1)
# for epoch in range(epochs):
#     scheduler.step(epoch)
# 效果并不是很好
#     epochs = 100
#     optimizer = optim.SGD(model.parameters(),lr = 1e-3, momentum=0.9, weight_decay=1e-4)        #SGD好像不需要weight_decay
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)
# # 针对模型的不同层设置不同的学习率  再对余弦退火进行修改      https://wanghao.blog.csdn.net/article/details/110220655
# #     model = torchvision.models.resnet101(pretrained=True)     #AttributeError: 'AlexNet' object has no attribute 'fc'只有部分模型可以使用下面这个，如221 231，但是也不能使用下面这个
#     model = model   #上面有编写过
#     large_lr_layers = list(map(id,model.fc.parameters()))
#     small_lr_layers = filter(lambda p:id(p) not in large_lr_layers,model.parameters())
#     # optimizer = torch.optim.SGD([{"params":large_lr_layers},{"params":small_lr_layers,"lr":1e-4}],lr = 1e-2, momenum=0.9)
#     optimizer = torch.optim.SGD([{"params":large_lr_layers},{"params":small_lr_layers,"lr":1e-4}],lr = 1e-3)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)
# # TypeError: optimizer can only optimize Tensors, but one of the params is int,这个也还是用不了，resnet也不行
#     optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)        #SGD好像不需要weight_decay，但是对于余弦退火是需要的
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)

# https://zhuanlan.zhihu.com/p/93624972     效果还是不好
#     optimizer_CosineLR = torch.optim.SGD(model.parameters(), lr=0.1)
#     CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_CosineLR, T_max=150, eta_min=0)
    # 参数T_max表示余弦函数周期；eta_min表示学习率的最小值，默认它是0表示学习率至少为正值。  确定一个余弦函数需要知道最值和周期，其中周期就是T_max，最值是初试学习率


    print('模型开始训练')
    trainer = Trainer(model, optimizer, F.cross_entropy, save_dir="weights")            #分类问题常用的损失函数为交叉熵( Cross Entropy Loss)
    # trainer = Trainer(model, optimizer, FocalLoss, save_dir="weights")
    # trainer = Trainer(model, optimizer, multiClassHingeLossYXY, save_dir="weights")
    # trainer = Trainer(model, optimizer, LossYXY, save_dir="weights")
    # trainer.loop(100, train_loader, test_loader, scheduler)
    # trainer.loop(100, train_loader, val_loader, test_loader, scheduler)
    trainer.loop(120, train_loader, val_loader, test_loader, scheduler, schedulerA)
    print('模型训练结束')

    # torch.save(model.state_dict(), "./weights/my_model.pth")

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    # p.add_argument("--batchsize", type=int, default=64)
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument('--gpu', type=str, default='0，1', help='ID of GPUs to use, eg. 1,3')
    # p.add_argument("--baseline", action="store_true")# 原始的注释掉
    args = p.parse_args()
    # main(args.batchsize, args.baseline, args.reduction)#原始的
    # main(args.batchsize)
    main(args.batchsize, args.reduction, args.gpu)
    print(model_index)
#     结果查询发现reduction主要是SElayer里面那个r=4 *4之后为16
#     发现注释掉这句代码以后运行的效果很不错但也只是极个别几个的时候效果好，因为出现了loss为nan的情况，意思是这个参数很需要调整
# [train] loss: 0.6994/accuracy: 0.5342
# >>>[test] loss: 0.6983/accuracy: 0.5000
# >>>[train] loss: 0.6444/accuracy: 0.5667
# >>>[test] loss: 0.3577/accuracy: 0.8533
# >>>[train] loss: nan/accuracy: 0.5900
# >>>[test] loss: nan/accuracy: 0.5000
