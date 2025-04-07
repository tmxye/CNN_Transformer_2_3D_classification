# https://blog.csdn.net/qq_36560894/article/details/104923543

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model_YXY.VGG_hc import VGG19
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# %matplotlib inline

# 利用torchvision对图像数据预处理
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomAffine(degrees=15,scale=(0.8,1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.ImageFolder(root='../data_2/train/', transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

valset = torchvision.datasets.ImageFolder(root='../data_2/test/', transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=1)

# 展示训练样本和测试样本数
print(len(trainloader))
print(len(valloader))
# CPU 或者 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 初始化网络,加载预训练模型
model = VGG19(num_classes=2, init_weights=False)
model_dict = model.state_dict()

import os
home = os.path.expanduser('~')

state_dict = torch.load(home + '/weights/vgg19.pth')
new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
model_dict.update(new_state_dict)
model.load_state_dict(model_dict)

# 查看GPU可用情况
if torch.cuda.device_count() > 1:
    print('We are using', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
model.to(device)

# 定义loss function和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# 保存每个epoch后的Accuracy Loss Val_Accuracy
Accuracy = []
Loss = []
Val_Accuracy = []
BEST_VAL_ACC = 0.


# 训练
since = time.time()
for epoch in range(10):
    train_loss = 0.
    train_accuracy = 0.
    run_accuracy = 0.
    run_loss = 0.
    total = 0.
    model.train()
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # 经典四步
        optimizer.zero_grad()
        outs = model(images)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        # 输出状态
        total += labels.size(0)
        run_loss += loss.item()
        _, prediction = torch.max(outs, 1)
        run_accuracy += (prediction == labels).sum().item()
        if i % 20 == 19:
            print('epoch {},iter {},train accuracy: {:.4f}%   loss:  {:.4f}'.format(epoch, i + 1, 100 * run_accuracy / (
                        labels.size(0) * 20), run_loss / 20))
            train_accuracy += run_accuracy
            train_loss += run_loss
            run_accuracy, run_loss = 0., 0.
    Loss.append(train_loss / total)
    Accuracy.append(100 * train_accuracy / total)
    # 可视化训练过程
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(0, epoch + 1, 1), Accuracy)
    ax1.set_title("Average trainset accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. train. accuracy")
    plt.savefig('Train_accuracy_vs_epochs.png')
    plt.clf()
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(epoch + 1), Loss)
    ax2.set_title("Average trainset loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig('loss_vs_epochs.png')

    plt.clf()
    plt.close()
    # 验证
    acc = 0.
    model.eval()
    print('waitting for Val...')
    with torch.no_grad():
        accuracy = 0.
        total = 0
        for data in valloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            _, prediction = torch.max(out, 1)
            total += labels.size(0)
            accuracy += (prediction == labels).sum().item()
            acc = 100. * accuracy / total
    print('epoch {}  The ValSet accuracy is {:.4f}% \n'.format(epoch, acc))
    Val_Accuracy.append(acc)
    if acc > BEST_VAL_ACC:
        print('Find Better Model and Saving it...')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/VGG19_Cats_Dogs_hc.pth')
        BEST_VAL_ACC = acc
        print('Saved!')

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch + 1), Val_Accuracy)
    ax3.set_title("Average Val accuracy vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current Val accuracy")

    plt.savefig('val_accuracy_vs_epoch.png')
    plt.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Now the best val Acc is {:.4f}%'.format(BEST_VAL_ACC))






