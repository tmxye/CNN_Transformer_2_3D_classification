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


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

#Model train
def train_model(model,train_loader,validation_loader,optimizer,N_test,criterion,n_epochs=4,gpu=False): 
    accuracy_list=[]
    loss_list=[]
    for epoch in range(n_epochs):
        for x, y in train_loader:
            if gpu:
                # Transfer to GPU
                x, y = x.to(device), y.to(device)
            
            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data)

        correct=0
        for x_test, y_test in validation_loader:
            if gpu:
                # Transfer to GPU
                x_test, y_test = x_test.to(device), y_test.to(device)
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        print(epoch+1, accuracy)

    return accuracy_list, loss_list

#Retraining module on CIFAR100
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

training_set = torchvision.datasets.CIFAR100('.', train=True, download=True, transform=transform)
training_generator = DataLoader(training_set, batch_size = 128)

validation_set = torchvision.datasets.CIFAR100('.', train=False, download=True, transform=transform)
validation_generator = DataLoader(validation_set, batch_size = 128)
n = len(validation_set)



model1 = torchvision.models.alexnet(pretrained=True, progress=True)
model2 = ZFNet(number_of_classes = 100)

param1 = model1.state_dict()
param2 = model2.state_dict()
if use_cuda:
	param2['cnn2.weight'] = param1['features.3.weight'].cuda()
	param2['cnn2.bias'] = param1['features.3.bias'].cuda()
	param2['cnn3.weight'] = param1['features.6.weight'].cuda()
	param2['cnn3.bias'] = param1['features.6.bias'].cuda()
	param2['cnn4.weight'] = param1['features.8.weight'].cuda()
	param2['cnn4.bias'] = param1['features.8.bias'].cuda()
	param2['cnn5.weight'] = param1['features.10.weight'].cuda()
	param2['cnn5.bias'] = param1['features.10.bias'].cuda()
	param2['fc1.weight'] = param1['classifier.1.weight'].cuda()
	param2['fc1.bias'] = param1['classifier.1.bias'].cuda()
	param2['fc2.weight'] = param1['classifier.4.weight'].cuda()
	param2['fc2.bias'] = param1['classifier.4.bias'].cuda()

else:
	param2['cnn2.weight'] = param1['features.3.weight']
	param2['cnn2.bias'] = param1['features.3.bias']
	param2['cnn3.weight'] = param1['features.6.weight']
	param2['cnn3.bias'] = param1['features.6.bias']
	param2['cnn4.weight'] = param1['features.8.weight']
	param2['cnn4.bias'] = param1['features.8.bias']
	param2['cnn5.weight'] = param1['features.10.weight']
	param2['cnn5.bias'] = param1['features.10.bias']
	param2['fc1.weight'] = param1['classifier.1.weight']
	param2['fc1.bias'] = param1['classifier.1.bias']
	param2['fc2.weight'] = param1['classifier.4.weight']
	param2['fc2.bias'] = param1['classifier.4.bias']


model2.load_state_dict(param2)

model2.fc2.weight.requires_grad = False
model2.fc2.bias.requires_grad = False
model2.fc1.weight.requires_grad = False
model2.fc1.bias.requires_grad = False
model2.cnn5.weight.requires_grad = False
model2.cnn5.bias.requires_grad = False
model2.cnn4.weight.requires_grad = False
model2.cnn4.bias.requires_grad = False
model2.cnn3.weight.requires_grad = False
model2.cnn3.bias.requires_grad = False
model2.cnn2.weight.requires_grad = False
model2.cnn2.bias.requires_grad = False

model2.to(device)

#Loss and hyperparameters set as per the authors of ZFNet
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
momentum = 0.9
optimizer = torch.optim.SGD([parameters  for parameters in model2.parameters() if parameters.requires_grad], lr = learning_rate, momentum = momentum)	

accuracy_list, loss_list = train_model(model=model2,n_epochs=35,train_loader=training_generator,validation_loader=validation_generator,optimizer=optimizer,gpu=use_cuda,N_test=n,criterion=criterion)

plt.plot(np.arange(len(accuracy_list)),accuracy_list)
plt.show()

plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()