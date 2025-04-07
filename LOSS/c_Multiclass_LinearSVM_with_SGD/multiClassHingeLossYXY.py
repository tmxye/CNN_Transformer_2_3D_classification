import torch
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