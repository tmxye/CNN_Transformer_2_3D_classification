# from LOSS.a_focal_loss_pytorch.focalloss import FocalLoss
# from LOSS.c_Multiclass_LinearSVM_with_SGD.multiClassHingeLossYXY import multiClassHingeLossYXY
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def multiClassHingeLossYXY(output, y, p=1, margin=1, weight=None, size_average=True): # output: batchsize*n_class
    # print(output.requires_grad)
    # print(y.requires_grad)
    output_y = output[torch.arange(0, y.size()[0]).long().cuda(), y.data.cuda()].view(-1, 1)  # view for transpose
    # margin - output[y] + output[i]
    loss = output - output_y + margin  # contains i=y
    # remove i=y items
    loss[torch.arange(0, y.size()[0]).long().cuda(), y.data.cuda()] = 0
    # max(0,_)
    loss[loss < 0] = 0
    # ^p
    if (p != 1):
        loss = torch.pow(loss, p)
    # add weight
    if (weight is not None):
        loss = loss * weight
    # sum up
    loss = torch.sum(loss)
    if (size_average):
        loss /= output.size()[0]  # output.size()[0]
    return loss


# def __init__(self, gamma=0, alpha=None, size_average=True):
def FocalLoss(input, target, gamma=2, alpha=0.75, size_average=True):       # 计算得出最好的alpha=0.07
    gamma = gamma
    alpha = alpha
    # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])     #原因是Python3.x中没有long类型，只有int类型。
    if isinstance(alpha,(float,int,int)): alpha = torch.Tensor([alpha,1-alpha])
    if isinstance(alpha,list): alpha = torch.Tensor(alpha)
    size_average = size_average

    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    target = target.view(-1,1)

    logpt = F.log_softmax(input)
    logpt = logpt.gather(1,target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())
    # print(pt)

    if alpha is not None:
        if alpha.type()!=input.data.type():
            alpha = alpha.type_as(input.data)
        at = alpha.gather(0,target.data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1-pt)**gamma * logpt
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def LossYXY(input, target):
    lossA = FocalLoss(input, target)
    lossB = multiClassHingeLossYXY(input, target)

    return 0.8 * lossA + 0.2 * lossB
