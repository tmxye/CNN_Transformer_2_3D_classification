import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import model_YXY_Google.Google_YXY

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
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

# transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




root='../data_2/'

train_data = MyDataset(datatxt=root + 'data_2_train_google.txt',
                       transform=transforms.Compose([transforms.Resize((299, 299)),  # 缩放
                                                     transforms.RandomCrop(299, padding=4),  # 裁剪
                                                     # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
                                                     transforms.ToTensor(),  # 转为张量，同时归一化
                                                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化      #原文是transforms.Normalize(norm_mean, norm_std),
                                                     ]), root='../data_2/')

# trainset = datasets.CIFAR10(root='D:/CIFAR-10', train=True,download=True, transform=transform)
# #
# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)
# # # 对于测试集的操作和训练集一样，我就不赘述了
# # testset = torchvision.datasets.CIFAR10(root='D:/CIFAR-10', train=False,download=True, transform=transform)
# # testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=2)
# #     # 类别信息也是需要我们给定的
# classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

val_data = MyDataset(datatxt=root + 'data_2_test_google.txt',
                     transform=transforms.Compose([transforms.Resize((299, 299)),  # 缩放
                                                   # transforms.RandomCrop(224, padding=4),  # 裁剪
                                                   # transforms.RandomHorizontalFlip(),  #**功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
                                                   transforms.ToTensor(),  # 转为张量，同时归一化
                                                   # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 标准化
                                                   ]), root='../data_2/')

test_data = MyDataset(datatxt=root + 'data_2_test_google.txt',
                      transform=transforms.Compose([transforms.Resize((299, 299)),  # 缩放
                                                    transforms.ToTensor(),  # 转为张量，同时归一化
                                                    ]), root='../data_2/')
batch_size = 64

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                          pin_memory=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=4,
                        pin_memory=True)  # batch_size单次训练用的样本数            shuffle:先对batch内打乱,再按顺序取batch
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

# 训练的网络导入
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model_YXY_Google.Google_YXY.GoogLeNet_YXY(num_classes=1000, aux_logits=True, init_weights=True)
# model =
model.to(device)
print(model)

import torch.nn as nn
import torch.optim as optim
#用到了神经网络工具箱 nn 中的交叉熵损失函数
loss_function = nn.CrossEntropyLoss()
# 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


best_acc = 0.0
# save_path = 'D:/CIFAR-10/model/GoogleNet.pth'
save_path = './GoogleNet.pth'
for epoch in range(5):
    # train
    model.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits, aux_logits2, aux_logits1 = model(images.to(device))
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        # x = model(images.to(device))
        # loss = x
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in test_loader:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / len(test_loader)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')
#
# # 定义2个存储每类中测试正确的个数的 列表，初始化为0
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in test_loader:
#         model.eval()
#         images, labels = data
#         images = Variable(images).cuda()
#         labels = Variable(labels).cuda()
#         outputs = model(images)
#
#         _, predicted = torch.max(outputs.data, 1)
#         # 4组(batch_size)数据中，输出于label相同的，标记为1，否则为0
#         c = (predicted == labels).squeeze()
#         for i in range(16):  # 因为每个batch都有4张图片，所以还需要一个4的小循环
#             label = labels[i]  # 对各个类的进行各自累加
#             class_correct[label] += c[i]
#             class_total[label] += 1
#
# for i in range(10):
#     # print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
#     print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))







