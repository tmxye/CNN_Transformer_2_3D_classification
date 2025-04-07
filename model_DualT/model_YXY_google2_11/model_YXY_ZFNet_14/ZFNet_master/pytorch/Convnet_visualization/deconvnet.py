import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torch.backends import cudnn
import torchvision.transforms as transforms
import matplotlib.pylab as plt
from torch.utils.data import Dataset, DataLoader
from model import ZFNet
import torch.nn.functional as F

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True


def Deconvnet(param, x, gpu=True):
  if gpu:
  	x = x.cuda()
  x = F.conv2d(input=x,weight=param['features.0.weight'].cuda(),bias=param['features.0.bias'].cuda(),stride=(4,4),padding=(2,2))
  x = torch.relu(x)
  x, indices1 = F.max_pool2d(input=x,return_indices=True,kernel_size=3,stride=2)
  x = F.conv2d(input=x,weight=param['features.3.weight'].cuda(),bias=param['features.3.bias'].cuda(),stride=(1,1),padding=(2,2))
  x = torch.relu(x)
  x, indices2 = F.max_pool2d(input=x,return_indices=True,kernel_size=3,stride=2)
  x = F.conv2d(input=x,weight=param['features.6.weight'].cuda(),bias=param['features.6.bias'].cuda(),stride=(1,1),padding=(1,1))
  x = torch.relu(x)
  x = F.conv2d(input=x,weight=param['features.8.weight'].cuda(),bias=param['features.8.bias'].cuda(),stride=(1,1),padding=(1,1))
  x = torch.relu(x)
  x = F.conv2d(input=x,weight=param['features.10.weight'].cuda(),bias=param['features.10.bias'].cuda(),stride=(1,1),padding=(1,1))
  x = torch.relu(x)
  x, indices3 = F.max_pool2d(input=x,return_indices=True,kernel_size=3,stride=2)

  x = F.max_unpool2d(input=x,indices=indices3,kernel_size=3,stride=2)
  x = torch.relu(x)
  k = param['features.10.bias'].cuda()
  k = k.view(1, k.shape[0], 1, 1)
  k = k.repeat(1,1,13,13)
  x = x-k
  x = F.conv_transpose2d(input=x,weight=param['features.10.weight'].cuda(),bias=None,stride=(1,1),padding=(1,1))
  x = torch.relu(x)
  k = param['features.8.bias'].cuda()
  k = k.view(1, k.shape[0], 1, 1)
  k = k.repeat(1,1,13,13)
  x = x-k
  x = F.conv_transpose2d(input=x,weight=param['features.8.weight'].cuda(),bias=None,stride=(1,1),padding=(1,1))
  x = torch.relu(x)
  k = param['features.6.bias'].cuda()
  k = k.view(1, k.shape[0], 1, 1)
  k = k.repeat(1,1,13,13)
  x = x-k
  x = F.conv_transpose2d(input=x,weight=param['features.6.weight'].cuda(),bias=None,stride=(1,1),padding=(1,1))
  x = F.max_unpool2d(input=x,indices=indices2,kernel_size=3,stride=2)
  x = torch.relu(x)
  k = param['features.3.bias'].cuda()
  k = k.view(1, k.shape[0], 1, 1)
  k = k.repeat(1,1,27,27)
  x = x-k
  x = F.conv_transpose2d(input=x,weight=param['features.3.weight'].cuda(),bias=None,stride=(1,1),padding=(2,2))
  x = F.max_unpool2d(input=x,indices=indices1,kernel_size=3,stride=2)
  x = torch.relu(x)
  k = param['features.0.bias'].cuda()
  k = k.view(1, k.shape[0], 1, 1)
  k = k.repeat(1,1,55,55)
  x = x-k
  x = F.conv_transpose2d(input=x,weight=param['features.0.weight'].cuda(),bias=None,stride=(4,4),padding=(2,2))
  x = x.cpu()
  return x


model1 = torchvision.models.alexnet(pretrained=True, progress=True)

img = Image.open("PATH_TO_IMAGE")
model1.eval()
img = torchvision.transforms.Resize((224,224))(img)
ip = torchvision.transforms.ToTensor()(img)
ip = ip.view(1,ip.shape[0],ip.shape[1],ip.shape[2])
if use_cuda:
	ip = ip.cuda()

x = Deconvnet(model1.state_dict(), ip, gpu=use_cuda)

x = x.cpu()
x = x.view(x.shape[3],x.shape[1],x.shape[2])
x = x.numpy()

plt.imshow(x)
plt.show()
