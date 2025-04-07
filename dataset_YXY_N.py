from PIL import Image
# from openpyxl.drawing.image import Image
import torch

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from RandAugment import RandAugment

# https://blog.csdn.net/Teeyohuang/article/details/79587125
# 或者是用这个制作https://blog.csdn.net/hszzjs/article/details/81005636

# https://zhuanlan.zhihu.com/p/97190313
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    #
    def __init__(self, root, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        # fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        # Add RandAugment with N, M(hyperparameter)
        # transform.transforms.insert(0, RandAugment(N, M))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容

        # root = './data_1/'
        # root = './original_dataset/'

        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        # img = Image.open(root + fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


class MyDataset_Rand(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    #
    def __init__(self, root, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset_Rand, self).__init__()
        # fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        # Add RandAugment with N, M(hyperparameter)
        # transform.transforms.insert(0, RandAugment(N, M))
        # transform.transforms.insert(0, RandAugment(2, 10))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容

        # root = './data_1/'
        # root = './original_dataset/'

        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        # img = Image.open(root + fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def get_dataloaderYXY(batch_size, root='./data_4/'):
# def get_dataloaderYXY(batch_size):

    print(root + 'data_2_train.txt')
    # https://blog.csdn.net/haozige888/article/details/108268042分享技术图片预处理 transforms 模块机制
    # 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    # train_data = {'train':transforms.Compose([transforms.RandomHorizontalFlip(),#核心是这里差了一个compose操作
    #                 transforms.Resize(image_size),
    #                 transforms.CenterCrop(image_size),
    #                 transforms.ToTensor()]), root='./data_1/')

# #下面这3个就是参考的上面的：https://blog.csdn.net/haozige888/article/details/108268042分享技术图片预处理 transforms 模块机制
#     train_data = MyDataset(datatxt=root + 'data_4_train.txt', transform = transforms.Compose([transforms.Resize((256, 256)),  # 缩放
#                     transforms.RandomCrop(224, padding=4),  # 裁剪
#                     # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化      #原文是transforms.Normalize(norm_mean, norm_std),
#                     ]), root='./data_1/')
#
#     val_data = MyDataset(datatxt=root + 'data_4_test.txt', transform = transforms.Compose([transforms.Resize((256, 256)),  # 缩放
#                     # transforms.RandomCrop(224, padding=4),  # 裁剪
#                     # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化
#                     ]), root='./data_1/')
#
#     test_data = MyDataset(datatxt=root + 'data_4_test.txt', transform = transforms.Compose([transforms.Resize((256, 256)),  # 缩放
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     ]), root='./data_1/')



# # InceptionV3
# #下面这3个就是参考的上面的：https://blog.csdn.net/haozige888/article/details/108268042分享技术图片预处理 transforms 模块机制
#     train_data = MyDataset(datatxt=root + 'data_4_train.txt', transform = transforms.Compose([transforms.Resize((299, 299)),  # 缩放
#                     transforms.RandomCrop(299, padding=4),  # 裁剪
#                     # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化      #原文是transforms.Normalize(norm_mean, norm_std),
#                     ]), root='./data_4/')
#
#     val_data = MyDataset(datatxt=root + 'data_4_test.txt', transform = transforms.Compose([transforms.Resize((299, 299)),  # 缩放
#                     # transforms.RandomCrop(224, padding=4),  # 裁剪
#                     # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化
#                     ]), root='./data_4/')
#
#     test_data = MyDataset(datatxt=root + 'data_4_test.txt', transform = transforms.Compose([transforms.Resize((299, 299)),  # 缩放
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     ]), root='./data_4/')



# # 其他网络
# #下面这3个就是参考的上面的：https://blog.csdn.net/haozige888/article/details/108268042分享技术图片预处理 transforms 模块机制
#     train_data = MyDataset_Rand(datatxt=root + 'data_2_train.txt', transform = transforms.Compose([transforms.Resize((256, 256)),  # 缩放
#                     transforms.RandomCrop(224, padding=None),  # 裁剪
#                     # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化      #原文是transforms.Normalize(norm_mean, norm_std),
#                     ]), root='./data_4/')
#
#     val_data = MyDataset(datatxt=root + 'data_2_val.txt', transform = transforms.Compose([transforms.Resize((256, 256)),  # 缩放
#                     # transforms.RandomCrop(224, padding=4),  # 裁剪
#                     transforms.RandomCrop(224, padding=None),  # 裁剪
#                     # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化
#                     ]), root='./data_4/')
#
#     test_data = MyDataset(datatxt=root + 'data_2_test.txt', transform = transforms.Compose([transforms.Resize((224, 224)),  # 缩放
#                     # transforms.RandomCrop(224, padding=None),  # 裁剪
#                     transforms.ToTensor(),  # 转为张量，同时归一化
#                     ]), root='./data_4/')




# 数据集变得更大
    train_data = MyDataset_Rand(datatxt=root + 'data_2_train1.txt', transform = transforms.Compose([transforms.Resize((256, 256)),  # 缩放
                    transforms.RandomCrop(224, padding=None),  # 裁剪
                    # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
                    transforms.ToTensor(),  # 转为张量，同时归一化
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化      #原文是transforms.Normalize(norm_mean, norm_std),
                    ]), root='./data_4/')

    val_data = MyDataset(datatxt=root + 'data_2_val1.txt', transform = transforms.Compose([transforms.Resize((256, 256)),  # 缩放
                    # transforms.RandomCrop(224, padding=4),  # 裁剪
                    transforms.RandomCrop(224, padding=None),  # 裁剪
                    # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
                    transforms.ToTensor(),  # 转为张量，同时归一化
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化
                    ]), root='./data_4/')

    test_data = MyDataset(datatxt=root + 'data_2_test1.txt', transform = transforms.Compose([transforms.Resize((224, 224)),  # 缩放
                    # transforms.RandomCrop(224, padding=None),  # 裁剪
                    transforms.ToTensor(),  # 转为张量，同时归一化
                    ]), root='./data_4/')





    # test_data = MyDataset(datatxt=root + 'data_4_test.txt', transform=transforms.Compose（[transforms.RandomSizedCrop(size=224), transforms.RandomHorizontalFlip(), transforms.Resize(size=224),
    #                                                                          transforms.CenterCrop(size=224), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), transforms.ToTensor()]）, root='./data_1/')


    # test_data = MyDataset(datatxt=root + 'data_4_test.txt', transform=transforms.Compose（[transforms.RandomSizedCrop(size=224), transforms.RandomHorizontalFlip(), transforms.Resize(size=224), transforms.Scale((args.image_size, args.image_size)),
    #                                                                          transforms.CenterCrop(size=224), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), transforms.ToTensor()]）, root='./data_1/')


    # train_data = MyDataset(datatxt=root + 'data_4_train.txt', transform=transforms.ToTensor(), root='./data_1/')
    # test_data = MyDataset(datatxt=root + 'data_4_test.txt', transform=transforms.ToTensor(), root='./data_1/')

    #
    # #然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    # train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    # val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=True) #batch_size单次训练用的样本数            shuffle:先对batch内打乱,再按顺序取batch
    # test_loader = DataLoader(dataset=test_data, batch_size=64)


    #然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
# #batch_size变大。这样可以一次性多载入数据到显存中，可以提高它的占用率，并且可以尽量占满GPU的内存。
# Dataloader中的num_workers。这个参数可以多进程的载入数据提高效率，一般可以选用4，8，16等等。但是，这个数量并不是越多越好，因为worker越多，一些进程间的分配和协作+I/O问题反而会拖慢速度。
# pin_memory=True。锁页内存，数据将不在硬盘中存储，省掉了将数据从CPU传入到缓存RAM里面，再给传输到GPU上，利用GPU时就会更快一些。
#     train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True) #batch_size单次训练用的样本数            shuffle:先对batch内打乱,再按顺序取batch
#     test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    train_loader = DataLoaderX(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoaderX(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True) #batch_size单次训练用的样本数            shuffle:先对batch内打乱,再按顺序取batch
    test_loader = DataLoaderX(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)

    # train_loader = DataLoaderX(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoaderX(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=4) #batch_size单次训练用的样本数            shuffle:先对batch内打乱,再按顺序取batch
    # test_loader = DataLoaderX(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)


    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

# 这个网站的也很不错     https://blog.csdn.net/weixin_32759777/article/details/105247567


