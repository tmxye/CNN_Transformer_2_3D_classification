import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model_YXY.VGG_hc import VGG19
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
model = VGG19(num_classes=2, init_weights=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)
model.load_state_dict(torch.load('./checkpoint/VGG19_Cats_Dogs_hc.pth'))

# 测试
id_list = []
pred_list = []

test_path = '../data_2/test/CT1/'
test_files = os.listdir(test_path)
model.eval()
with torch.no_grad():
    for file in tqdm(test_files):
        img = Image.open(test_path + file)
        _id = int(file.split('.')[0])
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        out = model(img)
        #         print(out)
        prediction = F.softmax(out, dim=1)[:, 1].tolist()
        _predict = np.array(prediction)
        _predict = np.where(_predict > 0.5, 1, 0)
        print(_id, _predict[0])
        id_list.append(_id)
        pred_list.append(_predict)

res = pd.DataFrame({
    'id': id_list,
    'label': pred_list
})

res.sort_values(by='id', inplace=True)
res.reset_index(drop=True, inplace=True)
res.to_csv('submission.csv', index=False)

res.head(10)

import random

# class_dict = {0:'cat', 1:'dog'}
class_dict = {0:'CT1', 1:'CT2'}
fig, axes = plt.subplots(2, 5, figsize=(20,12), facecolor='w')

for ax in axes.ravel():
    i = random.choice(res['id'].values)
    label = res.loc[res['id']==i, 'label'].values[0]
    img = Image.open('../data_2/test/CT1/'+str(i)+'.jpg')
    ax.set_title(class_dict[label[0]])
    ax.imshow(img)




