import torch
import torch.nn as nn



#ZFNet - Modified Alexnet module
class ZFNet(nn.Module):
    def __init__(self, number_of_classes=1000):
        super(ZFNet, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2))
        #self.cnn1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        #Authors originally suggested a kernel_size of 7X7 and stride of 2, instead of 11X11 and stride 2 as in original Alexnet.
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        
        self.cnn2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        
        self.cnn3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.cnn4 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.cnn5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        
        self.fc1 = nn.Linear(in_features=9216, out_features=4096, bias=True)

        self.drop1 = nn.Dropout(0.5, inplace=False)

        self.drop2 = nn.Dropout(0.5, inplace=False)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        
        # self.fc3 = nn.Linear(in_features=4096, out_features=100, bias=True)
        # by YXY    改为2分类
        self.fc3 = nn.Linear(in_features=4096, out_features=2, bias=True)

        self.indices1 = None
        self.indices2 = None
        self.indices3 = None
        self.feature = None

        #self.cnn1.weight = nn.Parameter(self.cnn1.weight.new_full(size=(64,3,7,7),fill_value=0.01))
        #self.cnn1.bias = nn.Parameter(self.cnn1.bias.new_full(size=(64,),fill_value=0))
        #Since, partially using a pretrained model, using the default initializer for remaining layers
    
    def get_indices(self):
        return self.indices1, self.indices2, self.indices3
    
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x, self.indices1 = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = torch.relu(x)
        x, self.indices2 = self.maxpool2(x)
        
        x = self.cnn3(x)
        x = torch.relu(x)
        
        x = self.cnn4(x)
        x = torch.relu(x)
        
        x = self.cnn5(x)
        x = torch.relu(x)
        x, self.indices3 = self.maxpool3(x)
        self.feature = x

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        
        return x
      
    def get_feature_map(self):
        return self.feature


model = ZFNet()
# print(model)

def ZFNet2(number_of_classes = 2):
    return ZFNet(number_of_classes = number_of_classes)