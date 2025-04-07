from pathlib import Path
import torch
import torch.backends.cudnn
# from numpy.core import sort
# from tensorflow.python.ops.metrics_impl import auc
from torch.autograd import Variable

from tqdm import tqdm
import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
plt.switch_backend('agg')

import scikitplot as skplt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, roc_curve, auc, confusion_matrix
from scikitplot.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve

# from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, roc_curve, auc
# from sklearn.metrics import confusion_matrix, precision_recall_curve, plot_precision_recall_curve, plot_roc_curve
#
# # from sklearn.metrics import confusion_matrix, precision_recall_curve
import itertools
from models_32.earlystopping import EarlyStopping


import os
cudaNum = 0
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaNum)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
test_loss_list = []
test_acc_list = []
time_list = []
# timestart = time.clock()
timestart = time.perf_counter()


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=1):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
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
            for module in self.model.modules():
                if hasattr(module, 'get_custom_L2'):
                    # loss += args.weight_decay * 0.5 * module.get_custom_L2()
                    loss += (1e-4) * 0.5 * module.get_custom_L2()
                #if hasattr(module, 'weight_gen'):
                    #loss += args.weight_decay * 0.5 * ((module.weight_gen()**2).sum())
                #if hasattr(module, 'weight'):
                    #loss += args.weight_decay * 0.5 * ((module.weight**2).sum())


            # # 在InceptionV3中需要修改为outputs,aux2,aux1 = net(inputs)
            # # outputs,aux2,aux1 = self.model(data)      不会修改
            # loss = self.loss_f(output, target)
            # print("loss", loss)


            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            # output_list.extend(output.data.max(1)[1].cpu().detach().numpy().tolist())
            output_list.append(output.data.max(1)[1].cpu().detach().numpy().tolist())
            target_list.append(target.data.cpu().detach().numpy().tolist())
            # output_list1.append(output_list[:])
            output_list1 = np.hstack(output_list[:])
            target_list1 = np.hstack(target_list[:])
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
        print(">>>[{}] loss: {:.4f}/accuracy: {:.4f} ".format(mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset)))
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

        return mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset), output_list1, target_list1

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            mode, loss, correct, output_list1, target_list1 = self._iteration(data_loader)
            # return mode, loss, correct
            train_loss_list.append(loss)
            train_acc_list.append(correct)
            train_data_len = len(data_loader.dataset)
            return mode, loss, correct, train_loss_list, train_acc_list, train_data_len, output_list1, target_list1


    def val(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            mode, loss, correct, output_list1, target_list1 = self._iteration(data_loader, is_train=False)
            # print('测试集的输出列表数据输出')
            # print(output_list1.shape())
            #  by YXY
            val_loss_list.append(loss)
            val_acc_list.append(correct)
            val_data_len = len(data_loader.dataset)
            # output_list1 = np.hstack(output_list1[:])
            # target_list1 = np.hstack(target_list1[:])
            # print(output_list1)
            # print(output_list1.size)

            return mode, loss, correct, val_loss_list, val_acc_list, val_data_len, output_list1, target_list1

    def test(self, data_loader):
        # self.model.test()
        # with torch.no_grad():
        mode, loss, correct, output_list1, target_list1 = self._iteration(data_loader, is_train=False)
        # print('测试集的输出列表数据输出')
        # print(output_list1.shape())
        #  by YXY
        test_loss_list.append(loss)
        test_acc_list.append(correct)
        test_data_len = len(data_loader.dataset)
        # output_list1 = np.hstack(output_list1[:])
        # target_list1 = np.hstack(target_list1[:])
        # print(output_list1)
        # print(output_list1.size)

        return mode, loss, correct, test_loss_list, test_acc_list, test_data_len, output_list1, target_list1



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

                x = np.linspace(0, len(train_acc_list), len(test_acc_list))
                plt.plot(x, train_acc_list, label="train_acc", linewidth=1.5)
                plt.plot(x, val_acc_list, label="val_acc", linewidth=1.5)
                plt.xlabel("epoch")
                plt.ylabel("acc")
                plt.legend()
                # plt.show()
                plt.savefig('./visual/2acc.jpg', dpi=300)
                plt.clf()
                print('把2acc数据保存到csv文件中')
                data_a = pd.DataFrame(index=range(0, len(train_acc_list)), columns=('x', 'y'))
                data_a['x'] = train_acc_list
                data_a['y'] = val_acc_list
                with open(r"./visual/2acc.csv".format(ep), "a+", encoding="utf-8") as txt:
                    txt.truncate(0)  # 清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
                data_a.to_csv(f'visual/2acc.csv', index=False)

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



    def save_statistic(self, epoch, mode, loss, accuracy):
        YXY = 0
        # with open("state.txt", "a", encoding="utf-8") as f:
        #     f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
        #     f.write("\n")
    def save_statistic_train(self, epoch, mode, loss, accuracy, train_loss_list, train_acc_list, train_data_len, output_list1, target_list1):
        with open("visual/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
            f.write("\n")
        # torch.save(mode.state_dict(), "my_model.pth")  # 只保存模型的参数
        # torch.save(model, "my_model.pth")  # 保存整个模型
    def save_statistic_val(self, epoch, mode, loss, accuracy, test_loss_list, test_acc_list, test_data_len, output_list1, target_list1):
        # print('开始测试了，后面有很多可视化的，判断是否每个周期都在运行')
        # ##########     ##在这里再添加一种ROC曲线画法        ##########        这个是python2的写法，最终显示不能使用，具体报错是多了参数，但依旧可以改代码来实现
        # [output_list1, index] = sort(output_list1)        # ValueError: too many values to unpack (expected 2)
        # # 计算出ground_truth中正样本的数目pos_num和负样本的数目neg_num
        # pos_num = sum(target_list1 == 1)
        # neg_num = sum(target_list1 == 0)
        # # 根据该数目可以计算出沿x轴或者y轴的步长
        # x_step = 1.0 / neg_num
        # y_step = 1.0 / pos_num
        # # 首先对predict中的分类器输出值按照从小到大排列
        # # data.sort_values('probability', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        # [output_list1, index] = sort(output_list1)
        # ground_truth = target_list1(index)
        # # 对predict中的每个样本分别判断他们是FP或者是TP        % 遍历ground_truth的元素，        % 若ground_truth[i] = 1, 则TP减少了1，往y轴方向下降y_step        % 若ground_truth[i] = 0, 则FP减少了1，往x轴方向下降x_step
        # x = 0
        # y = 0
        # X = []
        # Y = []
        # for i in range(len(ground_truth)):
        #     if ground_truth(i) == 1:
        #         y = y - y_step
        #     else:
        #         x = x - x_step
        #     X[i] = x
        #     Y[i] = y
        #
        # # 画出图像
        # plt.plot(X, Y, '-ro', 'LineWidth', 2, 'MarkerSize', 3)
        # plt.xlabel('虚报概率')
        # plt.ylabel('击中概率')
        # plt.title('ROC曲线图')
        # plt.savefig('./visual/6新的ROC曲线绘制.jpg')
        # plt.clf()
        # print('绘制图6完成')
        # # % 计算小矩形的面积, 返回auc
        # # auc = -trapz(X, Y)


    ##########     ##在这里再添加一种ROC曲线画法        ##########
        # fpr, tpr, thresholds = roc_curve(output_list1, target_list1, pos_label=1)
        # fpr = []
        # tpr = []
        # thresholds = []
        # for iii in len(output_list1):
        #     fpr[iii], tpr[iii], thresholds[iii] = roc_curve(output_list1[:iii], target_list1[:iii])
        # fpr, tpr, thresholds = roc_curve(output_list1, target_list1[:])
        # print("output_list1", output_list1.shape)         #output_list1 (645,)
        # print("target_list1", target_list1.shape)         #target_list1 (645,)
        fpr, tpr, thresholds = roc_curve(output_list1, target_list1[:])
        # plt.plot(fpr[iii], tpr[iii], linewidth=2, label="ROC")
        plt.plot(fpr, tpr, linewidth=2, label="ROC")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 01.01])
        print(fpr)
        print('开始画图ROc啦')
        plt.savefig('./visual/visual_val/ROC曲线绘制/仅一个点的/5新的ROC曲线绘制_{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/visual_val/5新的ROC曲线绘制.jpg', dpi=300)
        plt.clf()
        # plt.show()
        print('把数据保存到csv文件中')
        data_a = pd.DataFrame(index=range(0, len(fpr)), columns=('x', 'y'))
        data_a['x'] = fpr
        data_a['y'] = tpr
        with open(r"./visual/visual_val/ROC曲线绘制/roc曲线需要的参数fpr_tpr/roc曲线需要的参数fpr_tpr_{}.csv".format(epoch), "a+", encoding="utf-8") as txt:
            txt.truncate(0)  # 清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
        data_a.to_csv("./visual/visual_val/ROC曲线绘制/roc曲线需要的参数fpr_tpr/roc曲线需要的参数fpr_tpr_{}.csv".format(epoch), index=False)
        data_a.to_csv(f'visual/visual_val/roc曲线需要的参数fpr_tpr.csv', index=False)

        ##########      混淆矩阵可视化     #######
        fpr, tpr, threshold = roc_curve(output_list1, target_list1)
        # print(output_list1)
        #  UndefinedMetricWarning: No positive samples in y_true, true positive value should be meaningless
        # print(fpr)
        # print(tpr)
        # 计算混淆矩阵https://zhuanlan.zhihu.com/p/25212301
        cm = np.arange(4).reshape(2, 2)
        a = 0
        b = 0
        c = 0
        d = 0
        for i in range(len(output_list1)):
            if (output_list1[i] == 0) and (target_list1[i] < 0.5):  # TN
                # cm[0,0]=len(data[data['The ture label']==0][data['probability']<0.5])
                a = a + 1
            cm[0, 0] = a
            if (output_list1[i] == 0) and (target_list1[i] >= 0.5):  # FP
                b = b + 1
            cm[1, 0] = b
            if (output_list1[i] == 1) and (target_list1[i] < 0.5):  # FN
                c = c + 1
            cm[0, 1] = c
            if (output_list1[i] == 1) and (target_list1[i] >= 0.5):  # TP
                d = d + 1
            cm[1, 1] = d
        # print(a)
        # print(b)
        # print(a)
        # print(d)
        classes = [0, 1]
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix', fontsize=20)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
        plt.yticks(tick_marks, classes, fontsize=20)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
        # plt.tight_layout()      # 这句好如果注释，好像就不会出现图像的边缘只显示一半的情况了
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.savefig('./visual/visual_val/混淆矩阵/3混淆矩阵_{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/visual_val/3混淆矩阵.jpg', dpi=300)
        # torch.save(state_YXY, "./weights/model_epoch_{}.pth".format(epoch))
        plt.clf()

        # cm[0, 1] = len(data[data['The ture label'] == 0][data['probability'] >= 0.5])  # FP

        ##########      网络的二分类处理上五种评价指标     ############
        AUC = auc(fpr, tpr)
        # AUC = ('%.5f' % AUC)  # 只比较auc前5位数字
        print(AUC)
        print("accuracy_score:", accuracy_score(target_list1, output_list1))
        print("sklearn auc:", roc_auc_score(target_list1, output_list1))
        print("recall_score:", recall_score(target_list1, output_list1))
        print("f1_score:", f1_score(target_list1, output_list1))
        print("precision_score:", precision_score(target_list1, output_list1))
        # return mode, loss, correct

        ##########     ROC曲线绘制          ##########
        # TPRandFPR = pd.DataFrame(index=range(len(target_list1)), columns=('TP', 'FP'))
        # for j in range(len(data)):
        #     data1 = data.head(n=j + 1)
        #     FP = len(data1[data1['The ture label'] == 0][data1['probability'] >= data1.head(len(data1))['probabi lity']]) / float(len(data[data['The ture label'] == 0]))
        #     TP = len(data1[data1['The ture label'] == 1][data1['probability'] >= data1.head(len(data1))['probability']]) / float(len(data[data['The ture label'] == 1]))
        TPRandFPR = pd.DataFrame(index=range(len(target_list1)), columns=('TP', 'FP'))
        ######################################################这里有个问题是，别人是选择的部分值，而我选择的是一个值，应该是一系列值https://zhuanlan.zhihu.com/p/25212301
        #           这里需要是取一个值，一个值的。而我是没法取出一个值，然后一直取的


        data = pd.DataFrame(index=range(0, len(target_list1)), columns=('probability', 'The true label'))
        data['probability'] = output_list1
        data['The true label'] = target_list1
        # print(data)
        # print("YYYYYYYYYYY output_list1YYYYYYYYYYYYYYY",output_list1)

        # 首先，按预测概率从大到小的顺序排序：    #https://zhuanlan.zhihu.com/p/25212301
        print("output_list1", output_list1.shape)
        print("data", data.shape)
        data.sort_values('probability', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        result = data
        data1 = []
        # for j in range(len(data)):        #按类预测概率从大到小排序后的数据集
        for j in range(len(output_list1)):          #计算全部概率值下的FPR和TPR
            data1 = data.head(n=j+1)
            # FP = len(data1[data1['The true label'] == 0][data1['probability'] >= data1.head(len(data1))['probability']]) / float(len(data[data['The true label'] == 0]))
            # TP = len(data1[data1['The true label'] == 1][data1['probability'] >= data1.head(len(data1))['probability']]) / float(len(data[data['The true label'] == 1]))
            # FP = len(data1[(data1['The true label'] == 0)&(data1['probability'] >= data1.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 0]))
            # TP = len(data1[(data1['The true label'] == 1)&(data1['probability'] >= data1.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 1]))
            FP = len(data1[(data1['The true label'] == 0)&(data1['probability'] >= data.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 0]))
            TP = len(data1[(data1['The true label'] == 1)&(data1['probability'] >= data.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 1]))
            TPRandFPR.iloc[j] = [TP, FP]
            TPRandFPR.iloc[0] = [0, 0]

            # pr_data = data1
            # P = len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])])+len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]))
            # R = len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 0]))
            # print("float(len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])])+len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]))", float(len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])])+len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])])))
            # print("len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])])", len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]))
            # print("float(len(result[result['The true label'] == 0]))", float(len(result[result['The true label'] == 0])))
            # print("P", P)
            # print("R", R)

        # # 画出最终的ROC曲线和计算AUC值
        # # TPRandFPR.iloc[j] = [(b/(a+b)), (d/(c+d))]
        # # TPRandFPR.iloc[j] = []
        # print(TPRandFPR)
        # # plt.scatter(x=fpr, y=tpr, label='(FPR,TPR)', color='k')
        AUC = auc(TPRandFPR['FP'], TPRandFPR['TP'])
        # plt.scatter(x=TPRandFPR['FP'], y=TPRandFPR['TP'], label='(FPR,TPR)', color='g')               #原始文章的代码，这个代码是进行描点的，因此显得很粗
        # plt.scatter(x=TPRandFPR['FP'], y=TPRandFPR['TP'], ls='-', lw=1.5, label='(FPR,TPR)', color='g')
        # plt.plot(fprs[0],tprs[0], lw=1.5, label="W-Y, AUC=%.3f)"%aucs[0])
        # plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], 'k', label='AUC = %0.2f' % AUC)
        plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], ls='-', lw=1.5, label='AUC = %0.2f' % AUC, color='r')
        # plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], ls='-', lw=1.5, label='AUC = %0.2f' % AUC, color='r')
        # plt.legend(loc='lower right')

        plt.title('Receiver Operating Characteristic')
        # plt.plot([(0, 0), (1, 1)], 'r--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 01.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.show()
        print('开始画图啦')
        plt.savefig('./visual/visual_val/ROC曲线绘制/4ROC曲线绘制{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/visual_val/4ROC曲线绘制.jpg', dpi=300)
        print('把数据保存到csv文件中')
        with open(r"./visual/visual_val/ROC曲线绘制/ROC曲线所需参数/roc曲线需要的参数_{}.csv".format(epoch), "a+", encoding="utf-8") as txt:
            txt.truncate(0)  # 清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
        # data1.to_csv("./visual/visual_val/ROC曲线绘制/ROC曲线所需参数/roc曲线需要的参数_{}.csv".format(epoch), index=False)
        # data1.to_csv(f'visual/visual_val/roc曲线需要的参数.csv', index=False)
        data.to_csv("./visual/visual_val/ROC曲线绘制/ROC曲线所需参数/roc曲线需要的参数_{}.csv".format(epoch), index=False)
        data.to_csv(f'visual/visual_val/roc曲线需要的参数.csv', index=False)
        plt.clf()


        # # https://blog.csdn.net/yeshang_lady/article/details/103690564
        # con_matrix = confusion_matrix(target_list1, output_list1)
        # P = precision_score(target_list1, output_list1, average='binary')
        # R = recall_score(target_list1, output_list1, average='binary')
        # F1 = f1_score(target_list1, output_list1, average='binary')
        #
        # # # 1 直接用sklearn.metrics提供的函数画P-R曲线和ROC曲线
        # # plot_precision_recall_curve(LR, X, y)  # P-R曲线
        # # plot_roc_curve(LR, X, y)  # ROC曲线

        # 2 求出所需的值,然后用plot画图
        # y_pre_score = LR.predict_proba(X)
        y_pre_score = output_list1
        # precision, recall, _ = precision_recall_curve(target_list1, y_pre_score[:, 1])
        # fpr, tpr, _ = roc_curve(target_list1, y_pre_score[:, 1])
        # auc_num = roc_auc_score(target_list1, y_pre_score[:, 1])
        precision, recall, _ = precision_recall_curve(target_list1, y_pre_score[:])
        fpr, tpr, _ = roc_curve(target_list1, y_pre_score[:])
        auc_num = roc_auc_score(target_list1, y_pre_score[:])

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 1) P-R曲线
        plt.plot(recall, precision, color='darkorange', label='P-R')
        plt.xlabel('Recall')
        plt.ylabel('Persion')
        plt.savefig('./visual/visual_val/P_R曲线绘制/P_R曲线绘制/5P_R曲线绘制{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/visual_val/5P_R曲线绘制.jpg', dpi=300)
        # plt.show()
        plt.legend()
        plt.clf()

        # 2）ROC曲线
        plt.plot(fpr, tpr, label='ROC曲线')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig('./visual/visual_val/ROC曲线绘制/ROC曲线绘制/6ROC曲线绘制{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/visual_val/6ROC曲线绘制.jpg', dpi=300)
        # plt.show()
        plt.legend()
        plt.clf()


        ##########     ##########      网络的二分类处理上五种评价指标的文件保存     ############          ##########
        accuracy_score1 = accuracy_score(target_list1, output_list1)
        roc_auc_score1 = roc_auc_score(target_list1, output_list1)
        recall_score1 = recall_score(target_list1, output_list1)
        f1_score1 = f1_score(target_list1, output_list1)
        precision_score1 = precision_score(target_list1, output_list1)
        save_epoch_evaluating_indicator(accuracy_score1, roc_auc_score1, recall_score1, f1_score1,  precision_score1)




        # ##########     ##在这里再添加一种P_R曲线画法        ##########      # https://blog.csdn.net/yeshang_lady/article/details/103690564
        result = pd.DataFrame(index=range(0, len(target_list1)), columns=('probability', 'The true label'))
        result['probability'] = output_list1
        result['The true label'] = target_list1

        result.sort_values('probability', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        # result.sort_values('The true label', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        pr_data = []

        dict_1 = pd.DataFrame(index=range(len(target_list1)), columns=('P', 'R'))

        # for pro in pro_data:
        for pro in range(len(output_list1)):          #计算全部概率值下的FPR和TPR
            pr_data = result.head(n=pro+1)
            # print("pr_data", pr_data)
            # pr_data['pre_target'] = (pr_data.loc[:, '1'] >= pro).astype("int")
            # pr_data['pre_target'].value_counts()
            # FPR = len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 0]))
            # TPR = len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 1]))
            # P = len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])])+len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]))
            # R = len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 0]))
            P = len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) + len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]))
            R = len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 1]))

            # TP = pr_data[(pr_data['pre_target'] == 1) & (pr_data['target'] == 1)].shape[0]
            # TN = pr_data[(pr_data['pre_target'] == 0) & (pr_data['target'] == 0)].shape[0]
            # FP = pr_data[(pr_data['pre_target'] == 1) & (pr_data['target'] == 0)].shape[0]
            # FN = pr_data[(pr_data['pre_target'] == 0) & (pr_data['target'] == 1)].shape[0]
            # P = TP / (TP + FP)
            # R = TP / (TP + FN)
            # TPR = TP / (TP + FN)
            # FPR = FP / (FP + TN)

            dict_1.iloc[pro] = [P, R]
            dict_1.iloc[0] = [1, 0]
            dict_1.iloc[len(output_list1)-1] = [0, 1]
        plt.plot(dict_1['R'], dict_1['P'], color='darkorange', label='P-R')
        plt.xlabel("Recall")
        plt.ylabel("Persion")
        plt.savefig('./visual/visual_val/P_R曲线绘制/7P_R曲线绘制{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/visual_val/7P_R曲线绘制.jpg', dpi=300)
        # plt.show()
        plt.legend()
        plt.clf()




        # ##########     ##在这里再添加一种P_R曲线画法        ##########      # https://blog.csdn.net/yeshang_lady/article/details/103690564
        # result = pd.DataFrame(index=range(0, len(target_list1)), columns=('pre_target', 'target'))
        # result['pre_target'] = output_list1
        # result['target'] = target_list1
        # # pr_data['target'] = target_list1
        # pr_data = result[['1', 'target']]
        # # pr_data['pre_target'] = output_list1
        # # # result = pd.DataFrame(index=range(0, len(target_list1)), columns=('pre_target', 'target'))
        # # result['pre_target'] = output_list1
        # # # result = pd.DataFrame(LR.predict_proba(X), columns=LR.classes_).join(pd.Series(LR.predict(X), name='pre_target'))
        # # pr_data = result[['1', 'target']]
        # pro_data = sorted(pr_data['1'], reverse=True)
        # dict_1 = {'P': [], 'R': []}
        #
        # # result.sort_values('target', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        # # # result.sort_values('target', reverse=True)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        # # pr_data = []
        # # dict_1 = pd.DataFrame(index=range(len(target_list1)), columns=('P', 'R'))
        #
        # for pro in pro_data:
        # # for pro in range(len(output_list1)):          #计算全部概率值下的FPR和TPR
        #     # print("pr_data", pr_data)
        #     pr_data['pre_target'] = (pr_data.loc[:, '1'] >= pro).astype("int")
        #     pr_data['pre_target'].value_counts()
        #     TP = pr_data[(pr_data['pre_target'] == 1) & (pr_data['target'] == 1)].shape[0]
        #     TN = pr_data[(pr_data['pre_target'] == 0) & (pr_data['target'] == 0)].shape[0]
        #     FP = pr_data[(pr_data['pre_target'] == 1) & (pr_data['target'] == 0)].shape[0]
        #     FN = pr_data[(pr_data['pre_target'] == 0) & (pr_data['target'] == 1)].shape[0]
        #     P = TP / (TP + FP)
        #     R = TP / (TP + FN)
        #     # dict_1.iloc[pro] = [P, R]
        #     # dict_1.iloc[0] = [0, 0]
        #     dict_1['P'].append(P)
        #     dict_1['R'].append(R)
        #     # print("dict_1", dict_1)
        # plt.plot(dict_1['R'], dict_1['P'], color='darkorange', label='P-R')
        # plt.xlabel("Recall")
        # plt.ylabel("Persion")
        # plt.savefig('./visual/visual_val/P_R曲线绘制/7P_R曲线绘制{}.jpg'.format(epoch), dpi=300)
        # plt.savefig('./visual/visual_val/7P_R曲线绘制.jpg', dpi=300)
        # # plt.show()
        # plt.legend()
        # plt.clf()





        with open("./visual/visual_val/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
            f.write("\n")

        eval_loss = loss
        return eval_loss

    def save_statistic_test(self, epoch, mode, loss, accuracy, test_loss_list, test_acc_list, test_data_len, output_list1, target_list1):
        # print('开始测试了，后面有很多可视化的，判断是否每个周期都在运行')
        # ##########     ##在这里再添加一种ROC曲线画法        ##########        这个是python2的写法，最终显示不能使用，具体报错是多了参数，但依旧可以改代码来实现
        # [output_list1, index] = sort(output_list1)        # ValueError: too many values to unpack (expected 2)
        # # 计算出ground_truth中正样本的数目pos_num和负样本的数目neg_num
        # pos_num = sum(target_list1 == 1)
        # neg_num = sum(target_list1 == 0)
        # # 根据该数目可以计算出沿x轴或者y轴的步长
        # x_step = 1.0 / neg_num
        # y_step = 1.0 / pos_num
        # # 首先对predict中的分类器输出值按照从小到大排列
        # # data.sort_values('probability', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        # [output_list1, index] = sort(output_list1)
        # ground_truth = target_list1(index)
        # # 对predict中的每个样本分别判断他们是FP或者是TP        % 遍历ground_truth的元素，        % 若ground_truth[i] = 1, 则TP减少了1，往y轴方向下降y_step        % 若ground_truth[i] = 0, 则FP减少了1，往x轴方向下降x_step
        # x = 0
        # y = 0
        # X = []
        # Y = []
        # for i in range(len(ground_truth)):
        #     if ground_truth(i) == 1:
        #         y = y - y_step
        #     else:
        #         x = x - x_step
        #     X[i] = x
        #     Y[i] = y
        #
        # # 画出图像
        # plt.plot(X, Y, '-ro', 'LineWidth', 2, 'MarkerSize', 3)
        # plt.xlabel('虚报概率')
        # plt.ylabel('击中概率')
        # plt.title('ROC曲线图')
        # plt.savefig('./visual/6新的ROC曲线绘制.jpg')
        # plt.clf()
        # print('绘制图6完成')
        # # % 计算小矩形的面积, 返回auc
        # # auc = -trapz(X, Y)


    ##########     ##在这里再添加一种ROC曲线画法        ##########
        # fpr, tpr, thresholds = roc_curve(output_list1, target_list1, pos_label=1)
        # fpr = []
        # tpr = []
        # thresholds = []
        # for iii in len(output_list1):
        #     fpr[iii], tpr[iii], thresholds[iii] = roc_curve(output_list1[:iii], target_list1[:iii])
        # fpr, tpr, thresholds = roc_curve(output_list1, target_list1[:])
        fpr, tpr, thresholds = roc_curve(output_list1, target_list1[:])
        # plt.plot(fpr[iii], tpr[iii], linewidth=2, label="ROC")
        plt.plot(fpr, tpr, linewidth=2, label="ROC")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 01.01])
        print(fpr)
        print('开始画图ROc啦')
        plt.savefig('./visual/ROC曲线绘制/仅一个点的/5新的ROC曲线绘制_{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/5新的ROC曲线绘制.jpg', dpi=300)
        plt.clf()
        # plt.show()
        print('把数据保存到csv文件中')
        data_a = pd.DataFrame(index=range(0, len(fpr)), columns=('x', 'y'))
        data_a['x'] = fpr
        data_a['y'] = tpr
        with open(r"./visual/ROC曲线绘制/roc曲线需要的参数fpr_tpr/roc曲线需要的参数fpr_tpr_{}.csv".format(epoch), "a+", encoding="utf-8") as txt:
            txt.truncate(0)  # 清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
        data_a.to_csv("./visual/ROC曲线绘制/roc曲线需要的参数fpr_tpr/roc曲线需要的参数fpr_tpr_{}.csv".format(epoch), index=False)
        data_a.to_csv(f'visual/roc曲线需要的参数fpr_tpr.csv', index=False)

        ##########      混淆矩阵可视化     #######
        fpr, tpr, threshold = roc_curve(output_list1, target_list1)
        # print(output_list1)
        #  UndefinedMetricWarning: No positive samples in y_true, true positive value should be meaningless
        # print(fpr)
        # print(tpr)
        # 计算混淆矩阵https://zhuanlan.zhihu.com/p/25212301
        cm = np.arange(4).reshape(2, 2)
        a = 0
        b = 0
        c = 0
        d = 0
        for i in range(len(output_list1)):
            if (output_list1[i] == 0) and (target_list1[i] < 0.5):  # TN
                # cm[0,0]=len(data[data['The ture label']==0][data['probability']<0.5])
                a = a + 1
            cm[0, 0] = a
            if (output_list1[i] == 0) and (target_list1[i] >= 0.5):  # FP
                b = b + 1
            cm[1, 0] = b
            if (output_list1[i] == 1) and (target_list1[i] < 0.5):  # FN
                c = c + 1
            cm[0, 1] = c
            if (output_list1[i] == 1) and (target_list1[i] >= 0.5):  # TP
                d = d + 1
            cm[1, 1] = d

        # classes = [0, 1]
        # plt.figure()
        # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title('Confusion matrix')
        # tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=0)
        # plt.yticks(tick_marks, classes)
        # thresh = cm.max() / 2.
        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #     plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        # # plt.tight_layout()      # 这句好如果注释，好像就不会出现图像的边缘只显示一半的情况了
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.savefig('./visual/混淆矩阵/3混淆矩阵_{}.jpg'.format(epoch), dpi=300)
        # plt.savefig('./visual/3混淆矩阵.jpg', dpi=300)

        # print(d)
        classes = [0, 1]
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix', fontsize=20)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
        plt.yticks(tick_marks, classes, fontsize=20)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
        # plt.tight_layout()      # 这句好如果注释，好像就不会出现图像的边缘只显示一半的情况了
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.savefig('./visual/混淆矩阵/3混淆矩阵_{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/3混淆矩阵.jpg', dpi=300)
        # torch.save(state_YXY, "./weights/model_epoch_{}.pth".format(epoch))
        plt.clf()

        # cm[0, 1] = len(data[data['The ture label'] == 0][data['probability'] >= 0.5])  # FP

        ##########      网络的二分类处理上五种评价指标     ############
        AUC = auc(fpr, tpr)
        # AUC = ('%.5f' % AUC)  # 只比较auc前5位数字
        print(AUC)
        print("accuracy_score:", accuracy_score(target_list1, output_list1))
        print("sklearn auc:", roc_auc_score(target_list1, output_list1))
        print("recall_score:", recall_score(target_list1, output_list1))
        print("f1_score:", f1_score(target_list1, output_list1))
        print("precision_score:", precision_score(target_list1, output_list1))
        # return mode, loss, correct

        ##########     ROC曲线绘制          ##########
        # TPRandFPR = pd.DataFrame(index=range(len(target_list1)), columns=('TP', 'FP'))
        # for j in range(len(data)):
        #     data1 = data.head(n=j + 1)
        #     FP = len(data1[data1['The ture label'] == 0][data1['probability'] >= data1.head(len(data1))['probabi lity']]) / float(len(data[data['The ture label'] == 0]))
        #     TP = len(data1[data1['The ture label'] == 1][data1['probability'] >= data1.head(len(data1))['probability']]) / float(len(data[data['The ture label'] == 1]))
        TPRandFPR = pd.DataFrame(index=range(len(target_list1)), columns=('TP', 'FP'))
        ######################################################这里有个问题是，别人是选择的部分值，而我选择的是一个值，应该是一系列值https://zhuanlan.zhihu.com/p/25212301
        #           这里需要是取一个值，一个值的。而我是没法取出一个值，然后一直取的

        #   下面这个代码画出的ROC曲线不正确
        # for j in range(len(target_list1)):
        #     # # target_list11 = target_list1.head(n=j + 1)
        #     # # if (a+b)!=0 and (c+d)!=NAN
        #     # if np.isnan(a + b) == False and (a + b)!=0 and (c + d)!=0 and np.isnan(c + d) == False:
        #     #     FP = b / (a + b)
        #     #     TP = d / (c + d)
        #     #     TPRandFPR.iloc[j] = [TP, FP]
        #     #     # print('测试这里的TPRandFPR.iloc[j] 里面的TP是否传入')
        #     #     # print(TP)     #测试成功，传入的值是1.0
        #     # data = output_list1.head(n = j + 1)     #   别人代码这里是数组，亦可以说是一个表格       对于list取多少个数据。就是list[:2] # 取前两个            # word[2:] # 除了前两个，其他全部选取            # a[:-1]: 从头一直到最后一个元素a[-1]，但不包含最后一个元素。
        #     output_list1_data = output_list1[: (j + 1)]
        #     target_list1_data = target_list1[: (j + 1)]
        #
        #     # FP = sum(output_list1_data == target_list1[: (j + 1)] and output_list1_data == 0) / float(sum(target_list1_data == 0))      #算出FP的比例，     # 这个思想是合适的 ，先把相同的取出老，再进一步判断正确的有多少个
        #     # TP = sum(output_list1_data == target_list1[: (j + 1)] and output_list1_data == 1) / float(sum(target_list1_data == 1))      #算出TP的比例，
        #
        #     aa = 0
        #     bb = 0
        #     for k in range(len(target_list1_data)):
        #         # print('循环到哪了呀', k)
        #         if (output_list1_data[k] == target_list1_data[k]) and (output_list1_data[k] == 0):
        #             aa = aa + 1
        #         if (output_list1_data[k] == target_list1_data[k]) and (output_list1_data[k] == 1):
        #             bb = bb + 1
        #
        #     # print("aaaaaaaaaaaaaabbbbbbbbbbbbbbbbbb")
        #     # print(aa)
        #     # print(bb)
        #     # if aa != 0 and bb != 0:
        #     FP = aa / float(sum(target_list1 == 0))      #算出FP的比例，     # 这个思想是合适的 ，先把相同的取出老，再进一步判断正确的有多少个
        #     TP = bb / float(sum(target_list1 == 1))      #算出TP的比例，
        #     TPRandFPR.iloc[j] = [TP, FP]
        #     #
        #     # if j == 2599:           #q取出最后一项的值
        #     #     print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        #     #     print(aa)
        #     #     print('\n')
        #     #     print(bb)
        #     #     print(TPRandFPR)
        #     # if j == 20:           #q取出最后一项的值
        #     #     print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
        #     #     print(aa)
        #     #     print('\n')
        #     #     print(bb)
        #     #     print(TPRandFPR)

        data = pd.DataFrame(index=range(0, len(target_list1)), columns=('probability', 'The true label'))
        data['probability'] = output_list1
        data['The true label'] = target_list1
        # print(data)
        # print("YYYYYYYYYYY output_list1YYYYYYYYYYYYYYY",output_list1)

        # 首先，按预测概率从大到小的顺序排序：    #https://zhuanlan.zhihu.com/p/25212301
        data.sort_values('probability', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        data1 = []
        # for j in range(len(data)):        #按类预测概率从大到小排序后的数据集
        for j in range(len(output_list1)):          #计算全部概率值下的FPR和TPR
            data1 = data.head(n=j+1)
            # FP = len(data1[data1['The true label'] == 0][data1['probability'] >= data1.head(len(data1))['probability']]) / float(len(data[data['The true label'] == 0]))
            # TP = len(data1[data1['The true label'] == 1][data1['probability'] >= data1.head(len(data1))['probability']]) / float(len(data[data['The true label'] == 1]))
            # FP = len(data1[(data1['The true label'] == 0)&(data1['probability'] >= data1.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 0]))
            # TP = len(data1[(data1['The true label'] == 1)&(data1['probability'] >= data1.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 1]))
            FP = len(data1[(data1['The true label'] == 0)&(data1['probability'] >= data.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 0]))
            TP = len(data1[(data1['The true label'] == 1)&(data1['probability'] >= data.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 1]))
            TPRandFPR.iloc[j] = [TP, FP]
            TPRandFPR.iloc[0] = [0, 0]

        # # 画出最终的ROC曲线和计算AUC值
        # # TPRandFPR.iloc[j] = [(b/(a+b)), (d/(c+d))]
        # # TPRandFPR.iloc[j] = []
        # print(TPRandFPR)
        # # plt.scatter(x=fpr, y=tpr, label='(FPR,TPR)', color='k')
        AUC = auc(TPRandFPR['FP'], TPRandFPR['TP'])
        # plt.scatter(x=TPRandFPR['FP'], y=TPRandFPR['TP'], label='(FPR,TPR)', color='g')               #原始文章的代码，这个代码是进行描点的，因此显得很粗
        # plt.scatter(x=TPRandFPR['FP'], y=TPRandFPR['TP'], ls='-', lw=1.5, label='(FPR,TPR)', color='g')
        # plt.plot(fprs[0],tprs[0], lw=1.5, label="W-Y, AUC=%.3f)"%aucs[0])
        # plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], 'k', label='AUC = %0.2f' % AUC)
        plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], ls='-', lw=1.5, label='AUC = %0.2f' % AUC, color='r')
        # plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], ls='-', lw=1.5, label='AUC = %0.2f' % AUC, color='r')
        # plt.legend(loc='lower right')

        plt.title('Receiver Operating Characteristic')
        # plt.plot([(0, 0), (1, 1)], 'r--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 01.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.show()
        print('开始画图啦')
        plt.savefig('./visual/ROC曲线绘制/4ROC曲线绘制{}.jpg'.format(epoch))
        plt.savefig('./visual/4ROC曲线绘制.jpg')
        print('把数据保存到csv文件中')
        with open(r"./visual/ROC曲线绘制/ROC曲线所需参数/roc曲线需要的参数_{}.csv".format(epoch), "a+", encoding="utf-8") as txt:
            txt.truncate(0)  # 清空文件内容      注意：仅当以 "r+" "rb+" "w" "wb" "wb+"等以可写模式打开的文件才可以执行该功能。
        data.to_csv("./visual/ROC曲线绘制/ROC曲线所需参数/roc曲线需要的参数_{}.csv".format(epoch), index=False)
        data.to_csv(f'visual/roc曲线需要的参数.csv', index=False)
        plt.clf()


        ##########     ##########      网络的二分类处理上五种评价指标的文件保存     ############          ##########
        accuracy_score1 = accuracy_score(target_list1, output_list1)
        roc_auc_score1 = roc_auc_score(target_list1, output_list1)
        recall_score1 = recall_score(target_list1, output_list1)
        f1_score1 = f1_score(target_list1, output_list1)
        precision_score1 = precision_score(target_list1, output_list1)
        save_epoch_test_indicator(accuracy_score1, roc_auc_score1, recall_score1, f1_score1,  precision_score1)





        # # https://blog.csdn.net/yeshang_lady/article/details/103690564
        # con_matrix = confusion_matrix(target_list1, output_list1)
        # P = precision_score(target_list1, output_list1, average='binary')
        # R = recall_score(target_list1, output_list1, average='binary')
        # F1 = f1_score(target_list1, output_list1, average='binary')
        #
        # # # 1 直接用sklearn.metrics提供的函数画P-R曲线和ROC曲线
        # # plot_precision_recall_curve(LR, X, y)  # P-R曲线
        # # plot_roc_curve(LR, X, y)  # ROC曲线

        # 2 求出所需的值,然后用plot画图
        # y_pre_score = LR.predict_proba(X)
        y_pre_score = output_list1
        # precision, recall, _ = precision_recall_curve(target_list1, y_pre_score[:, 1])
        # fpr, tpr, _ = roc_curve(target_list1, y_pre_score[:, 1])
        # auc_num = roc_auc_score(target_list1, y_pre_score[:, 1])
        precision, recall, _ = precision_recall_curve(target_list1, y_pre_score[:])
        fpr, tpr, _ = roc_curve(target_list1, y_pre_score[:])
        auc_num = roc_auc_score(target_list1, y_pre_score[:])

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 1) P-R曲线
        plt.plot(recall, precision, color='darkorange', label='P-R')
        plt.xlabel('Recall')
        plt.ylabel('Persion')
        plt.savefig('./visual/P_R曲线绘制/P_R曲线绘制/5P_R曲线绘制{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/5P_R曲线绘制.jpg', dpi=300)
        # plt.show()
        plt.legend()
        plt.clf()

        # 2）ROC曲线
        plt.plot(fpr, tpr, label='ROC曲线')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig('./visual/ROC曲线绘制/ROC曲线绘制/6ROC曲线绘制{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/6ROC曲线绘制.jpg', dpi=300)
        # plt.show()
        plt.legend()
        plt.clf()





        # ##########     ##在这里再添加一种P_R曲线画法        ##########      # https://blog.csdn.net/yeshang_lady/article/details/103690564
        result = pd.DataFrame(index=range(0, len(target_list1)), columns=('probability', 'The true label'))
        result['probability'] = output_list1
        result['The true label'] = target_list1

        result.sort_values('probability', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        # result.sort_values('The true label', inplace=True, ascending=False)            # 改了看能不能随机        #g改了以后ROC曲线成反的了
        pr_data = []

        dict_1 = pd.DataFrame(index=range(len(target_list1)), columns=('P', 'R'))

        # for pro in pro_data:
        for pro in range(len(output_list1)):          #计算全部概率值下的FPR和TPR
            pr_data = result.head(n=pro+1)
            # print("pr_data", pr_data)
            # pr_data['pre_target'] = (pr_data.loc[:, '1'] >= pro).astype("int")
            # pr_data['pre_target'].value_counts()
            # FPR = len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 0]))
            # TPR = len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 1]))
            # P = len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])])+len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]))
            # R = len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 0]))
            P = len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(pr_data[(pr_data['The true label'] == 0) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) + len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]))
            R = len(pr_data[(pr_data['The true label'] == 1) & (pr_data['probability'] >= result.head(len(pr_data))['probability'])]) / float(len(result[result['The true label'] == 1]))

            # TP = pr_data[(pr_data['pre_target'] == 1) & (pr_data['target'] == 1)].shape[0]
            # TN = pr_data[(pr_data['pre_target'] == 0) & (pr_data['target'] == 0)].shape[0]
            # FP = pr_data[(pr_data['pre_target'] == 1) & (pr_data['target'] == 0)].shape[0]
            # FN = pr_data[(pr_data['pre_target'] == 0) & (pr_data['target'] == 1)].shape[0]
            # P = TP / (TP + FP)
            # R = TP / (TP + FN)
            # TPR = TP / (TP + FN)
            # FPR = FP / (FP + TN)

            dict_1.iloc[pro] = [P, R]
            dict_1.iloc[0] = [1, 0]
            dict_1.iloc[len(output_list1)-1] = [0, 1]
        plt.plot(dict_1['R'], dict_1['P'], color='darkorange', label='P-R')
        plt.xlabel("Recall")
        plt.ylabel("Persion")
        plt.savefig('./visual/P_R曲线绘制/7P_R曲线绘制{}.jpg'.format(epoch), dpi=300)
        plt.savefig('./visual/7P_R曲线绘制.jpg', dpi=300)
        # plt.show()
        plt.legend()
        plt.clf()



        with open("./visual/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
            f.write("\n")



# def save_time(epoch, elapsed, mode=None, loss=None, accuracy=None, train_loss_list=None, train_acc_list=None, train_data_len=None):
#     with open("state.txt", "a", encoding="utf-8") as f:
#         f.write(str({"每个epoch的": epoch, "elapsed": elapsed, "mode": mode, "loss": loss, "accuracy": accuracy}))
#         f.write("\n")
# def save_time_end(epoch, elapsed, mode=None, loss=None, accuracy=None, train_loss_list=None, train_acc_list=None, train_data_len=None):
#     with open("state.txt", "a", encoding="utf-8") as f:
#         f.write(str({"最后epoch的": epoch, "elapsed": elapsed, "mode": mode, "loss": loss, "accuracy": accuracy}))
#         f.write("\n")
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

