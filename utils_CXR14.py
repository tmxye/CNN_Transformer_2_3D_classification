from pathlib import Path
import torch
import torch.backends.cudnn
# from numpy.core import sort
# from tensorflow.python.ops.metrics_impl import auc
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
plt.switch_backend('agg')

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_recall_curve, plot_precision_recall_curve, plot_roc_curve
import itertools
from models_32.earlystopping import EarlyStopping

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
test_loss_list = []
test_acc_list = []
time_list = []
# timestart = time.clock()
timestart = time.perf_counter()


def FocalLoss(input, target):
    BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    #         BCE_loss = self.BCE_loss(input, target)
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    F_loss = (1 - pt) ** 2.0 * BCE_loss
    return F_loss.mean()


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=1):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer

        # loss_f = nn.BCEWithLogitsLoss()
        loss_f = nn.CrossEntropyLoss()
        # loss_f = nn.BCELoss()
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq

    # def _iteration(self, data_loader, is_train=True, scheduler=None):
    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        output_list = []
        target_list = []
        # output_list1 = []
        # target_list1 = []
        for data, target in tqdm(data_loader, ncols=120):
            # target = torch.zeros(len(self.all_classes))
            # print("target", target.shape)
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                # data = data.cuda('cuda:0', non_blocking=True)
                # target = target.cuda('cuda:0', non_blocking=True)

                # data, target = data, target.cuda()

            # # 这也是一个优化语句，效果不确定
            # data = Variable(data, volatile=True)

            output = self.model(data)
            # print("output", output.shape)

            loss = self.loss_f(output, target)

            # loss = FocalLoss(output, target)
            # print("out", output.shape)
            # print("target)", target.shape)

            # # data = pd.DataFrame(output)
            # # data1 = pd.DataFrame(target)
            # file = open('visual/probability.txt', 'w')
            # file.write(str(output))
            # file.close()
            # file = open('visual/true.txt', 'w')
            # file.write(str(target))
            # file.close()

            # for module in self.model.modules():
            #     if hasattr(module, 'get_custom_L2'):
            #         # loss += args.weight_decay * 0.5 * module.get_custom_L2()
            #         loss += (1e-4) * 0.5 * module.get_custom_L2()
            #     #if hasattr(module, 'weight_gen'):
            #         #loss += args.weight_decay * 0.5 * ((module.weight_gen()**2).sum())
            #     #if hasattr(module, 'weight'):
            #         #loss += args.weight_decay * 0.5 * ((module.weight**2).sum())


            # # 在InceptionV3中需要修改为outputs,aux2,aux1 = net(inputs)
            # # outputs,aux2,aux1 = self.model(data)      不会修改
            # loss = self.loss_f(output, target)
            # print("loss", loss)


            loop_loss.append(loss.item() / len(data_loader))
            # accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            # output_list.extend(output.data.max(1)[1].cpu().detach().numpy().tolist())
            output_list.append(output.data.cpu().detach().numpy().tolist())
            target_list.append(target.data.cpu().detach().numpy().tolist())
            # output_list1.append(output_list[:])
            output_list1 = output_list
            target_list1 = target_list
            # # output_list1 = np.hstack(output_list[:])
            # # print(output_list1[:])

            # # print(output_list1)
            # # target_list1.append(target_list[:])
            # # print(output_list)
            # # print('目标数据')
            # # print(target.data)
            # # print('网络输出数据')
            # # print(output.data.max(1)[1])
            # # 这里出现问题是TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            #
            # fpr, tpr, threshold = roc_curve(output_list1, target_list1)
            # # print(output_list1)
            # # fpr, tpr, threshold = roc_curve(output_list, target_list)
            # AUC = auc(fpr, tpr)
            # # print(fpr)
            # # print(tpr)

            # # AUC = ('%.5f' % AUC)  # 只比较auc前5位数字
            # print(AUC)

            # scheduler.step()     PyTorch 1.1.0 之前， scheduler.step() 应该在 optimizer.step() 之前调用。现在这么做则会跳过学习率更新的第一个值。
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        mode = "train" if is_train else "test"
        print(">>>[{}] loss:".format(mode, sum(loop_loss)))
        # print('\n')
        train_data_len = 10
        test_data_len = 10
        # if not is_train:
        #     print('\n')           #这个放在时间后面会更加好
            # print(sum(loop_loss))
            # test_loss_list.append(sum(loop_loss))
            # test_acc_list.append(sum(accuracy) / len(data_loader.dataset))
            # test_data_len = len(data_loader.dataset)
        # if is_train: # 之前测试用的，通过修改train已经完成了这个操作，因此注释掉
            # print("yes")
            # train_loss_list.append(sum(loop_loss))
            # train_acc_list.append(sum(accuracy) / len(data_loader.dataset))
            # train_data_len = len(data_loader.dataset)

        # https://blog.csdn.net/qq_37534947/article/details/109726153   这是画图的参考位置
        # print('picture')
        # x = np.linspace(0, train_data_len, test_data_len)
        # plt.plot(x, train_loss_list, label="train_loss", linewidth=1.5)
        # plt.plot(x, test_loss_list, label="test_loss", linewidth=1.5)
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        # plt.legend()
        # # plt.show()
        # plt.savefig('2loss.jpg')
        # plt.clf()
        #
        # x = np.linspace(0, train_data_len, test_data_len)
        # plt.plot(x, train_acc_list, label="train_acc", linewidth=1.5)
        # plt.plot(x, test_acc_list, label="test_acc", linewidth=1.5)
        # plt.xlabel("epoch")
        # plt.ylabel("acc")
        # plt.legend()
        # # plt.show()
        # plt.savefig('2acc.jpg')

        return mode, sum(loop_loss), output_list1, target_list1

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            mode, loss, output_list1, target_list1 = self._iteration(data_loader)
            # return mode, loss, correct
            train_loss_list.append(loss)
            # train_acc_list.append(correct)
            train_data_len = len(data_loader.dataset)
            return mode, loss, train_loss_list, train_acc_list, train_data_len, output_list1, target_list1


    def val(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            mode, loss, output_list1, target_list1 = self._iteration(data_loader, is_train=False)
            # print('测试集的输出列表数据输出')
            # print(output_list1.shape())
            #  by YXY
            val_loss_list.append(loss)
            # val_acc_list.append(correct)
            val_data_len = len(data_loader.dataset)
            # output_list1 = np.hstack(output_list1[:])
            # target_list1 = np.hstack(target_list1[:])
            # print(output_list1)
            # print(output_list1.size)

            return mode, loss, val_loss_list, val_data_len, output_list1, target_list1

    def test(self, data_loader):
        # self.model.test()
        # with torch.no_grad():
        mode, loss, output_list1, target_list1 = self._iteration(data_loader, is_train=False)
        # print('测试集的输出列表数据输出')
        # print(output_list1.shape())
        #  by YXY
        test_loss_list.append(loss)
        # test_acc_list.append(correct)
        test_data_len = len(data_loader.dataset)
        # output_list1 = np.hstack(output_list1[:])
        # target_list1 = np.hstack(target_list1[:])
        # print(output_list1)
        # print(output_list1.size)

        return mode, loss, test_loss_list, test_data_len, output_list1, target_list1



    # UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later,
    # you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.
    # Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.
    # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    #   "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

    # 原本代码：
    # def loop(self, epochs, train_data, test_data, scheduler=None):
    #     for ep in range(1, epochs + 1):
    #         if scheduler is not None:
    #             scheduler.step()
    #         print("epochs: {}".format(ep))
    #         # save statistics into txt file
    #         self.save_statistic(*((ep,) + self.train(train_data)))
    #         self.save_statistic(*((ep,) + self.test(test_data)))
    #         if ep % self.save_freq:
    #             self.save(ep)



    def loop(self, epochs, train_data, val_data, test_data, scheduler=None, schedulerA=None):
        # early_stopping = EarlyStopping(args['patience'], verbose=True, delta=args['delta'])
        early_stopping = EarlyStopping(4000, verbose=True, delta=0)
        for ep in range(1, epochs + 1):
                # scheduler.step(ep)
            # 添加一个根据epoch改变学习率的代码，好像很难实现
            if scheduler is not None:
                scheduler.step()
            epochstart = time.perf_counter()
            print("epochs: {}".format(ep))
            # save statistics into txt file
            self.save_statistic_train(*((ep,) + self.train(train_data)))
            # print(test_data)
            eval_loss = self.save_statistic_val(*((ep,) + self.val(val_data)))
            if schedulerA is not None:
                schedulerA.step(eval_loss)
            with torch.no_grad():
                self.save_statistic_test(*((ep,) + self.test(test_data)))
            early_stopping(eval_loss, self.model)
            # if scheduler is not None:
            #     scheduler.step()
                # scheduler.step(ep)
            if ep % self.save_freq == 0:
                self.save(ep)
                        # 参考网页  https://blog.csdn.net/weixin_38314865/article/details/103937717
                        # 原因是如报错所说，在“optimizer.step()”之前对“lr_scheduler.step()”的进行了调用
                #         可能都需要放，比如train后面和test后面
                # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,但是不绝对，可以根据具体的需求来做。
                # 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。通常我们有
                # optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
                # scheduler = lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
                # model = net.train(model, loss_function, optimizer, scheduler, num_epochs = 100)
                # 在scheduler的step_size表示scheduler.step()每调用step_size次，对应的学习率就会按照策略调整一次。
                # 所以如果scheduler.step()是放在mini-batch里面，那么step_size指的是经过这么多次迭代，学习率改变一次。

                    #  判断训练网络需要时间的代码
            gc.collect()
            elapsed = (time.perf_counter() - epochstart)
            time_list.append(elapsed)
            print('第 %d 周期训练和测试用的Time used %.6f s \n\n' %(ep, elapsed))
            save_time(ep, elapsed)
            # if ep % 1 == 0:


            if ep % 1 == 0:     #画出训练集和测试集的图形       根据之前训练集和测试集的训练loss和准确度进行绘图，在这里通过plt.savefig将其保存到了本地
                x = np.linspace(0, len(train_loss_list), len(test_loss_list))
                plt.plot(x, train_loss_list, label="train_loss", linewidth=1.5)
                plt.plot(x, val_loss_list, label="val_loss", linewidth=1.5)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend()
                # plt.show()
                plt.savefig('./visual/2loss.jpg', dpi=300)
                # print('Picture has Finish,but you can alse use visual/viz.py to read state.txt and write it ,Which is word by YXY')
                plt.clf()
                print('把2acc数据保存到csv文件中')
                data_a = pd.DataFrame(index=range(0, len(train_loss_list)), columns=('x', 'y'))
                data_a['x'] = train_loss_list
                data_a['y'] = test_loss_list
                with open(r"./visual/2loss.csv".format(ep), "a+", encoding="utf-8") as txt:
                    txt.truncate(0)  # 清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
                data_a.to_csv(f'visual/2loss.csv', index=False)

                # x = np.linspace(0, len(train_acc_list), len(test_acc_list))
                # plt.plot(x, train_acc_list, label="train_acc", linewidth=1.5)
                # plt.plot(x, val_acc_list, label="val_acc", linewidth=1.5)
                # plt.xlabel("epoch")
                # plt.ylabel("acc")
                # plt.legend()
                # # plt.show()
                # plt.savefig('./visual/2acc.jpg', dpi=300)
                # plt.clf()
                # print('把2acc数据保存到csv文件中')
                # data_a = pd.DataFrame(index=range(0, len(train_acc_list)), columns=('x', 'y'))
                # data_a['x'] = train_acc_list
                # data_a['y'] = val_acc_list
                # with open(r"./visual/2acc.csv".format(ep), "a+", encoding="utf-8") as txt:
                #     txt.truncate(0)  # 清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
                # data_a.to_csv(f'visual/2acc.csv', index=False)

            if early_stopping.early_stop:
                print("Early stopping.")
                timesum = (time.perf_counter() - timestart)
                print('\nThe Early stopping total time is %.6f  s' % (timesum))
                save_time_Earlystopping_end(2222, timesum)
                break

        timesum = (time.perf_counter() - timestart)
        print('\nThe total time is %.6f  s' %(timesum))
        save_time_end(1111, timesum)



    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            # state = {"epoch": epoch, "net_state_dict": self.model.state_dict()}
            state_YXY = self.model.state_dict()
            if not model_out_path.exists():
                model_out_path.mkdir()
            # torch.save(state, model_out_path / "model_epoch_{}.ckpt".format(epoch))
            torch.save(state_YXY, "./weights/model_epoch_{}.pth".format(epoch))

            # torch.save(model.state_dict(), './checkpoint/VGG19_Cats_Dogs_hc.pth')



    def save_statistic(self, epoch, mode, loss):
        with open("state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss}))
            f.write("\n")
    def save_statistic_train(self, epoch, mode, loss, train_loss_list, train_acc_list, train_data_len, output_list1, target_list1):
        with open("visual/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss}))
            f.write("\n")
        # torch.save(mode.state_dict(), "my_model.pth")  # 只保存模型的参数
        # torch.save(model, "my_model.pth")  # 保存整个模型
    def save_statistic_val(self, epoch, mode, loss,  test_loss_list, test_data_len, output_list1, target_list1):

        # # AUC = ('%.5f' % AUC)  # 只比较auc前5位数字
        # print("accuracy_score:", accuracy_score(target_list1, output_list1))
        # # print("sklearn auc:", roc_auc_score(target_list1, output_list1))
        # # print("recall_score:", recall_score(target_list1, output_list1))
        # # print("f1_score:", f1_score(target_list1, output_list1))
        # # print("precision_score:", precision_score(target_list1, output_list1))
        # # return mode, loss, correct

        with open("./visual/visual_val/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss}))
            f.write("\n")

        eval_loss = loss
        return eval_loss

    def save_statistic_test(self, epoch, mode, loss, test_loss_list, test_data_len, output_list1, target_list1):
        # ##########     ##########      网络的二分类处理上五种评价指标的文件保存     ############          ##########
        # # roc_auc = get_roc_auc_score(target_list1, output_list1)
        # roc_auc = roc_auc_score(target_list1, output_list1)
        # # accuracy_score1 = accuracy_score(target_list1, output_list1)
        # # roc_auc_score1 = roc_auc_score(target_list1, output_list1)
        # # recall_score1 = recall_score(target_list1, output_list1)
        # # f1_score1 = f1_score(target_list1, output_list1)
        # # precision_score1 = precision_score(target_list1, output_list1)
        # save_epoch_test_indicator(roc_auc, roc_auc, roc_auc, roc_auc,  roc_auc)

        with open("./visual/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss}))
            f.write("\n")


def get_roc_auc_score(y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    # print('y_true.shape, y_probs.shape ', y_true.shape, y_probs.shape)

    # data = pd.DataFrame(index=range(0, len(y_true)), columns=('probability', 'The true label'))
    data = pd.DataFrame(y_true)
    data1 = pd.DataFrame(y_probs)
    data.to_csv(f'./visual/true.csv', index=False)
    data1.to_csv(f'./visual/probability.csv', index=False)
    class_roc_auc_list = []
    useful_classes_roc_auc_list = []

    for i in range(y_true.shape[1]):
        class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        class_roc_auc_list.append(class_roc_auc)
    print('\nclass_roc_auc_list: ', class_roc_auc_list)

    return np.mean(np.array(useful_classes_roc_auc_list))


def save_time(epoch, elapsed):
    with open("./visual/state_time.txt", "a", encoding="utf-8") as f:
        f.write(str({"每个epoch的": epoch, "elapsed": elapsed}))
        f.write("\n")
def save_time_end(epoch, elapsed):
    with open("./visual/state_time.txt", "a", encoding="utf-8") as f:
        f.write(str({"最后epoch的": epoch, "elapsed": elapsed}))
        f.write("\n")
def save_time_Earlystopping_end(epoch, elapsed):
    with open("./visual/state_time.txt", "a", encoding="utf-8") as f:
        f.write(str({"save_time_Earlystopping_end最后epoch的": epoch, "elapsed": elapsed}))
        f.write("\n")

def save_epoch_evaluating_indicator(accuracy_score, roc_auc_score, recall_score, f1_score, precision_score):
    with open("./visual/visual_val/save_epoch_evaluating_indicator.txt", "a", encoding="utf-8") as f:
        f.write(str({"accuracy_score:": accuracy_score, "sklearn_auc:": roc_auc_score, "recall_score:": recall_score, "f1_score:": f1_score, "precision_score:": precision_score}))
        f.write("\n")
def save_epoch_test_indicator(accuracy_score, roc_auc_score, recall_score, f1_score, precision_score):
    with open("./visual/save_epoch_evaluating_indicator.txt", "a", encoding="utf-8") as f:
        f.write(str({"accuracy_score:": accuracy_score, "sklearn_auc:": roc_auc_score, "recall_score:": recall_score, "f1_score:": f1_score, "precision_score:": precision_score}))
        f.write("\n")
# def adjust_learning_rate(optimizer, epoch):
#     # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
#     lr = 0.1
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# import torch.optim as optim
#
# epochs = 100
# optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9)        #SGD好像不需要weight_decay
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)

