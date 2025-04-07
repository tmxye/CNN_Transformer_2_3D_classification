import argparse
import os, sys, time

import torch
import torch.nn as nn
import torchvision.models as models 

# import custom dataset classes
import ChestXray14
from ChestXray14.datasets import XRaysTrainDataset
from ChestXray14.datasets import XRaysTestDataset

# import neccesary libraries for defining the optimizers
import torch.optim as optim

from ChestXray14.trainer import fit
from classification_choice_YXY_2 import get_model
import torch.nn.functional as F

model_index = 231
# model_index = 4023

def q(text = ''): # easy way to exiting the script. useful while debugging
    print('> ', text)
    sys.exit()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

from torch.utils.data import DataLoader
# https://zhuanlan.zhihu.com/p/97190313
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# PyTorch：The "freeze_support()" line can be omitted if the program is not going to be frozen
# 解决办法是将你要运行的代码块放到main函数中运行即可       if __name__ == '__main__':      # 	#your code
if __name__ == '__main__':
    # # 由于我这里准备使用多GPU，因此这里设置换成我自己编写的
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(f'\ndevice: {device}')
    with open(r"./ChestXray14/state.txt", "a+", encoding="utf-8") as txt:
        txt.truncate(0)        #   清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。

    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself ! Cool huh ? :D')
    parser.add_argument('--data_path', type = str, default = 'NIH Chest X-rays', help = 'This is the path of the training data')
    # parser.add_argument('--bs', type = int, default = 128, help = 'batch size')
    # parser.add_argument('--bs', type = int, default = 160, help = 'batch size')
    # parser.add_argument('--bs', type = int, default = 16, help = 'batch size')
    parser.add_argument('--bs', type = int, default = 32, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'Learning Rate for the optimizer')
    parser.add_argument('--stage', type = int, default = 1, help = 'Stage, it decides which layers of the Neural Net to train')
    # parser.add_argument('--loss_func', type = str, default = 'FocalLoss', choices = {'BCE', 'FocalLoss'}, help = 'loss function')
    parser.add_argument('--loss_func', type = str, default = 'BCE', choices = {'BCE', 'FocalLoss', 'Cross'}, help = 'loss function')
    parser.add_argument('-r','--resume', action = 'store_true') # args.resume will return True if -r or --resume is used in the terminal
    parser.add_argument('--ckpt', type = str, help = 'Path of the ckeckpoint that you wnat to load')
    parser.add_argument('-t','--test', action = 'store_true')   # args.test   will return True if -t or --test   is used in the terminal

    # by YXY    添加多GPU代码
    parser.add_argument('--gpu', type=str, default = '0', help='ID of GPUs to use, eg. 1,3')
    parser.add_argument("--reduction", type=int, default=16)

    args = parser.parse_args()

    if args.resume and args.test: # what if --test is not defiend at all ? test case hai ye ek
        q('The flow of this code has been designed either to train the model or to test it.\nPlease choose either --resume or --test')

    stage = args.stage
    if not args.resume:
        print(f'\nOverwriting stage to 1, as the model training is being done from scratch')
        stage = 1

    if args.test:
        print('TESTING THE MODEL')
    else:
        if args.resume:
            print('RESUMING THE MODEL TRAINING')
        else:
            print('TRAINING THE MODEL FROM SCRATCH')

    script_start_time = time.time() # tells the total run time of this script

    # mention the path of the data
    data_dir = os.path.join('ChestXray14',args.data_path) # Data_Entry_2017.csv should be present in the mentioned path

    # define a function to count the total number of trainable parameters
    def count_parameters(model):
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_parameters/1e6 # in terms of millions

    # make the datasets
    print("A 有没有准备进去读训练数据集")
    XRayTrain_dataset = XRaysTrainDataset(data_dir, transform = ChestXray14.config.transform, transform_Label = ChestXray14.config.transform_Label)
    print('num images in XRayTrain_dataset   : {}'.format(len(XRayTrain_dataset)))
    print("D 训练数据集装载成功")
    train_percentage = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])

    XRayTest_dataset = XRaysTestDataset(data_dir, transform = ChestXray14.config.transform)

    print('\n-----Initial Dataset Information-----')
    print('num images in train_dataset   : {}'.format(len(train_dataset)))
    print('num images in val_dataset     : {}'.format(len(val_dataset)))
    print('num images in XRayTest_dataset: {}'.format(len(XRayTest_dataset)))
    print('-------------------------------------')

    # # make the dataloaders
    # batch_size = args.bs # 128 by default
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)
    # test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size = batch_size, shuffle = not True)

    # by YXY, 这里将前面的位置修改为我自己写的那个dataset_YXY
    # make the dataloaders
    batch_size = args.bs # 128 by default
    # by YXY 这里的batchsize如果是64，那么建议numwork是12。。。。    如果这里的batchsize如果是32，那么建议numwork是6，验证集的numwork改为8
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # # test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size = batch_size, shuffle = not True)
    # test_loader = torch.utils.data.DataLoader(dataset=XRayTest_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    # # test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size = batch_size, shuffle = not True)
    # test_loader = torch.utils.data.DataLoader(dataset=XRayTest_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    train_loader = DataLoaderX(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoaderX(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    # test_loader = DataLoaderX(XRayTest_dataset, batch_size = batch_size, shuffle = not True)
    test_loader = DataLoaderX(dataset=XRayTest_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    print('\n-----Initial Batchloaders Information -----')
    print('num batches in train_loader: {}'.format(len(train_loader)))
    print('num batches in val_loader  : {}'.format(len(val_loader)))
    print('num batches in test_loader : {}'.format(len(test_loader)))
    print('-------------------------------------------')

    # sanity check
    if len(XRayTrain_dataset.all_classes) != 15: # 15 is the unique number of diseases in this dataset
        q('\nnumber of classes not equal to 15 !')

    a,b = train_dataset[0]
    print('\nwe are working with \nImages shape: {} and \nTarget shape: {}'.format( a.shape, b.shape))

    # make models directory, where the models and the loss plots will be saved
    if not os.path.exists(ChestXray14.config.models_dir):
        os.mkdir(ChestXray14.config.models_dir)


    # define the learning rate
    lr = args.lr

    # # by YXY        多gpu并行
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #     # gpus = list(range(len(args.gpu.split(','))))
    #     gpus = list(range(len(args.gpu.split(','))))
    #     # gpu1 = args.gpu
    # else:
    #     gpus = [0]  # [1,2]
    # print('gpus是', gpus)
    # model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    # print('gpu1是', gpu1)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # # # define the loss function
    # # if args.loss_func == 'FocalLoss':  # by default
    # #     from ChestXray14.losses import FocalLoss
    # #     loss_fn = FocalLoss(device=device, gamma=2.).to(device)
    # # elif args.loss_func == 'BCE':
    # #     loss_fn = nn.BCEWithLogitsLoss().to(device)
    # # device = torch.cuda.get_device_name(gpus)
    # device_id = torch.cuda.device_count()
    # device = torch.cuda.get_device_name(range(device_id))
    # # 报错误RuntimeError: Invalid device string: 'TITAN V'，需要再修改，测试下cuda:0       device = ('cuda:0')可以使用
    # device = ("cuda:{}".format(gpu1))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cuda")


    # # define the loss function
    # if args.loss_func == 'FocalLoss':  # by default
    #     from ChestXray14.losses import FocalLoss
    #     loss_fn = FocalLoss(device=device, gamma=2.).to(device)
    # elif args.loss_func == 'BCE':
    #     loss_fn = nn.BCEWithLogitsLoss().to(device)

    # define the loss function
    if args.loss_func == 'FocalLoss':  # by default
        from ChestXray14.losses import FocalLoss
        # loss_fn = FocalLoss(device=device, gamma=2.).to(device)
        loss_fn = FocalLoss(device=device, gamma=2.)
    elif args.loss_func == 'BCE':
        # loss_fn = nn.BCEWithLogitsLoss().to(device)
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss_func == 'A':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_func == 'B':
        loss_fn = nn.MSELoss()
    elif args.loss_func == 'C':
        loss_fn = nn.L1Loss()
    #     出现问题是：Expected object of scalar type Long but got scalar type Float for argument #2 'target' in call to _thnn_nll_loss_forward，因此注释掉
    # elif args.loss_func == 'Cross':
    #     # loss_fn = F.cross_entropy().to(device)
    #     # 出现错误      TypeError: cross_entropy() missing 2 required positional arguments: 'input' and 'target'
    #     loss_fn = nn.CrossEntropyLoss().to(device)


    if not args.test: # training
        # initialize the model if not args.resume
        if not args.resume:
            print('\ntraining from scratch')

            # # import pretrained model
            # model = models.resnet50(pretrained=True) # pretrained = False bydefault
            # 这里试着改成我编写的代码，再进行测试
            print('模型开始导入')
            # create modelW
            model = get_model(model_index=model_index)
            print('模型导入成功')
            # 这里的是直接在后面添加了一个linear层，现在测试下，把原始网络的linear去掉，再在这里添加新的
            # change the last linear layer  由于SEdensenet后面不使用fc结构，替换一下
            # 原始的resnet代码是self.fc = nn.Linear(512 * block.expansion, num_classes)   替换的DenseNet为self.classifier = nn.Linear(num_features, num_classes)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(XRayTrain_dataset.all_classes)) # 15 output classes

            # num_ftrs = model.classifier.in_features         # 4023
            # print("15?_______", len(XRayTrain_dataset.all_classes))
            # model.classifier = nn.Linear(num_ftrs, len(XRayTrain_dataset.all_classes)) # 15 output classes

            # model.to(device)
            # # 这里也可以只写一个
            # print('模型开始导入')
            # # create modelW
            # model = get_model(model_index=model_index,num_classes=15)
            # print('模型导入成功')

            # 后面使用上19classification里面的多GPU代码
            # # 这也是一个优化语句，注释之后效果可能会更好点
            model.apply(inplace_relu)

            # by YXY        多gpu并行
            if args.gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
                # gpus = list(range(len(args.gpu.split(','))))
                gpus = list(range(len(args.gpu.split(','))))
                # gpu1 = args.gpu
            else:
                gpus = [0]  # [1,2]
            print('gpus是', gpus)
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

            # # import pretrained model
            # model = models.resnet50(pretrained=True) # pretrained = False bydefault
            # change the last linear layer
            # num_ftrs = model.fc.in_features
            # model.fc = nn.Linear(num_ftrs, len(XRayTrain_dataset.all_classes)) # 15 output classes
            # model.to(device)

            # print('----- STAGE 1 -----') # only training 'layer2', 'layer3', 'layer4' and 'fc'
            # for name, param in model.named_parameters(): # all requires_grad by default, are True initially
            #     # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
            #     if ('layer2' in name) or ('layer3' in name) or ('layer4' in name) or ('fc' in name):
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False

            # since we are not resuming the training of the model
            epochs_till_now = 0

            # making empty lists to collect all the losses
            losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}

        else:
            if args.ckpt == None:
                q('ERROR: Please select a valid checkpoint to resume from')

            print('\nckpt loaded: {}'.format(args.ckpt))
            ckpt = torch.load(os.path.join(ChestXray14.config.models_dir, args.ckpt))

            # since we are resuming the training of the model
            epochs_till_now = ckpt['epochs']
            model = ckpt['model']
            # model.to(device)
            # 这里也需要修改，因此改为上面的那个
            # by YXY        多gpu并行
            if args.gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
                # gpus = list(range(len(args.gpu.split(','))))
                gpus = list(range(len(args.gpu.split(','))))
                # gpu1 = args.gpu
            else:
                gpus = [0]  # [1,2]
            print('gpus是', gpus)
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

            # loading previous loss lists to collect future losses
            losses_dict = ckpt['losses_dict']

        # printing some hyperparameters
        print('\n> loss_fn: {}'.format(loss_fn))
        print('> epochs_till_now: {}'.format(epochs_till_now))
        print('> batch_size: {}'.format(batch_size))
        print('> stage: {}'.format(stage))
        print('> lr: {}'.format(lr))

    else: # testing
        if args.ckpt == None:
            q('ERROR: Please select a checkpoint to load the testing model from')

        print('\ncheckpoint loaded: {}'.format(args.ckpt))
        ckpt = torch.load(os.path.join(ChestXray14.config.models_dir, args.ckpt))

        # since we are resuming the training of the model
        epochs_till_now = ckpt['epochs']
        model = ckpt['model']

        # loading previous loss lists to collect future losses
        losses_dict = ckpt['losses_dict']

    # make changes(freezing/unfreezing the model's layers) in the following, for training the model for different stages
    if (not args.test) and (args.resume):

        if stage == 1:

            print('\n----- STAGE 1 -----') # only training 'layer2', 'layer3', 'layer4' and 'fc'
            for name, param in model.named_parameters(): # all requires_grad by default, are True initially
                # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                if ('layer2' in name) or ('layer3' in name) or ('layer4' in name) or ('fc' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif stage == 2:

            print('\n----- STAGE 2 -----') # only training 'layer3', 'layer4' and 'fc'
            for name, param in model.named_parameters():
                # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                if ('layer3' in name) or ('layer4' in name) or ('fc' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif stage == 3:

            print('\n----- STAGE 3 -----') # only training  'layer4' and 'fc'
            for name, param in model.named_parameters():
                # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                if ('layer4' in name) or ('fc' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif stage == 4:

            print('\n----- STAGE 4 -----') # only training 'fc'
            for name, param in model.named_parameters():
                # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                if ('fc' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


    if not args.test:
        # checking the layers which are going to be trained (irrespective of args.resume)
        trainable_layers = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                layer_name = str.split(name, '.')[0]
                if layer_name not in trainable_layers:
                    trainable_layers.append(layer_name)
        print('\nfollowing are the trainable layers...')
        print(trainable_layers)

        print('\nwe have {} Million trainable parameters here in the {} model'.format(count_parameters(model), model.__class__.__name__))

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    # # make changes in the parameters of the following 'fit' function
    # fit(device, XRayTrain_dataset, train_loader, val_loader,
    #                                         test_loader, model, loss_fn,
    #                                         optimizer, losses_dict,
    #                                         epochs_till_now = epochs_till_now, epochs = 30,
    #                                         log_interval = 25, save_interval = 1,
    #                                         lr = lr, bs = batch_size, stage = stage,
    #                                         test_only = args.test)

    # # by YXY
    # # 除了修改底下东西，还需要修改让代码不只是训练部分层，需要训练全部层。使用我的代码里面的优化器试试
    # # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    # optimizer = optim.SGD(params=model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    # # optimizer = optim.Adam(params=model.parameters(), lr=1e-3)  # use default
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 60], 0.1)
    # # make changes in the parameters of the following 'fit' function
    # fit(device, XRayTrain_dataset, train_loader, val_loader,
    #                                         test_loader, model, loss_fn,
    #                                         optimizer, scheduler, losses_dict,
    #                                         epochs_till_now = epochs_till_now, epochs = 80,
    #                                         log_interval = 25, save_interval = 10,
    #                                         lr = lr, bs = batch_size, stage = stage,
    #                                         test_only = args.test)

    # by YXY    训练的时候发现问题，schedular没法进行更新， 因此更改StepLR的代码，不使用这个，还是使用原始的optimizer实现训练优化器。
    # 但由于要使用stepLR，将optimizer写到train的函数里面。现在进行更改
    # 除了修改底下东西，还需要修改让代码不只是训练部分层，需要训练全部层。使用我的代码里面的优化器试试
    # optimizer = optim.SGD(params=model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 60], 0.1)
    optimizer = None
    scheduler = None
    # make changes in the parameters of the following 'fit' function
    print('A.，开始进入训练')
    # fit(device, XRayTrain_dataset, train_loader, val_loader,
    #                                         test_loader, model, loss_fn,
    #                                         optimizer, scheduler, losses_dict,
    #                                         epochs_till_now = epochs_till_now, epochs = 80,
    #                                         log_interval = 25, save_interval = 10,
    #                                         lr = lr, bs = batch_size, stage = stage,
    #                                         test_only = args.test)
    fit(device, XRayTrain_dataset, train_loader, val_loader,
                                            test_loader, model, loss_fn,
                                            optimizer, scheduler, losses_dict,
                                            epochs_till_now = epochs_till_now, epochs = 80,
                                            log_interval = 5, save_interval = 5,
                                            lr = lr, bs = batch_size, stage = stage,
                                            test_only = args.test)

    script_time = time.time() - script_start_time
    m, s = divmod(script_time, 60)
    h, m = divmod(m, 60)
    print('{} h {}m laga poore script me !'.format(int(h), int(m)))

    # '''
    # This is how the model is trained...
    # ##### STAGE 1 ##### FocalLoss lr = 1e-5
    # training layers = layer2, layer3, layer4, fc
    # epochs = 2
    # ##### STAGE 2 ##### FocalLoss lr = 3e-4
    # training layers = layer3, layer4, fc
    # epochs = 5
    # ##### STAGE 3 ##### FocalLoss lr = 7e-4
    # training layers = layer4, fc
    # epochs = 4
    # ##### STAGE 4 ##### FocalLoss lr = 1e-3
    # training layers = fc
    # epochs = 3
    # '''
