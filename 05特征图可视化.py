# import matplotlib; matplotlib.use('TkAgg')
import torch
from torchvision import models, transforms
from PIL import Image
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
plt.rcParams['font.sans-serif']=['STSong']
# mpl.use('TkAgg')
# plt.switch_backend('agg')
# import torchvision.models as models
# model = models.alexnet(pretrained=True)
from classification_choice_YXY_2 import get_model
import imageio
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# #1.模型查看
# print(model)#可以看出网络一共有3层，两个Sequential()+avgpool
# model_features = list(model.children())
# 出现问题就把它暂时注释了    print(model_features[0][3])#取第0层Sequential()中的第四层# TypeError: 'Conv2d' object does not support indexing
# print(model_features[0][3])#取第0层Sequential()中的第四层


# for index,layer in enumerate(model_features[0]):
#     print(layer)

#2. 导入数据
# 以RGB格式打开图像
# Pytorch DataLoader就是使用PIL所读取的图像格式
# 建议就用这种方法读取图像，当读入灰度图像时convert('')
def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')#是一幅图片
    # 数据预处理方法
    image_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.Resize((224, 224)),
        transforms.Resize((1120, 1120)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)#torch.Size([3, 224, 224])
    image_info = image_info.unsqueeze(0)#torch.Size([1, 3, 224, 224])因为model的输入要求是4维，所以变成4维
    return image_info#变成tensor数据




def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:  # jpg为-3/py为-2
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist


#2. 获取第k层的特征图
'''
args:
k:定义提取第几层的feature map
x:图片的tensor
model_layer：是一个Sequential()特征层
'''
def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):     #model的第一个Sequential()是有多层，所以遍历
            x = layer(x)    #torch.Size([1, 64, 55, 55])生成了64个通道
            if k == index:
                return x


#  可视化特征图
def show_feature_map(feature_map, imgname, inum, k=0):#feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)    #压缩成torch.Size([64, 55, 55])
    feature_map_num = feature_map.shape[0]  #返回通道数
    # row_num = np.ceil(np.sqrt(feature_map_num)) #8
    row_num = int(np.ceil(np.sqrt(feature_map_num))) #8

    # if not os.path.exists('feature_map_save/'+ str(imgname) + '/'):
    #     os.mkdir('feature_map_save/'+ str(imgname) + '/')
    if not os.path.exists('feature_map_save/'+ str(imgname)+ '_'+ str(inum) + '/'):
        os.mkdir('feature_map_save/'+ str(imgname) + '_'+ str(inum) + '/')
    # print("feature_map[index - 1]", feature_map[0].shape)
    endsize = feature_map[0].shape[0]
    # if endsize < 35:
    if endsize < 70:
        return endsize
    plt.figure()
    for index in range(1, feature_map_num + 1):#通过遍历的方式，将64个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        # UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.        #   % get_backend())
        # plt.imshow(feature_map[index - 1], cmap='gray')#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        # scipy.misc.imsave( 'feature_map_save//'+str(index) + ".png", feature_map[index - 1])
        # # 在show_feature_map函数中加上一句，tensor数据变成Img的操作
        # image_PIL = transforms.ToPILImage()(feature_map[index - 1])
        # scipy.misc.imsave('feature_map_save/'+ str(k) + '_' + str(index) + ".png", feature_map[index - 1])
        # scipy.misc.imsave( 'feature_map_save/'+str(index) + ".png", image_PIL)
        # imageio.imsave('feature_map_save/'+ str(imgname) + '/' + str(k) + '_' + str(index) + ".png", feature_map[index - 1])
        imageio.imsave('feature_map_save/'+ str(imgname) + '_'+ str(inum) + '/' + str(k) + '_' + str(index) + ".jpg", feature_map[index - 1])
    # plt.show()

    return endsize



if __name__ ==  '__main__':
    # image_dir = r"car_logol.png"
    # image_dir = r"./data_4/test1/CT3/20.jpg"
    # input_dir = './data_4/test1/CT3'
    input_dir = './data_9_COVID2/feature/'

    for inum in range(69):
        if inum <= 20:
            if inum % 8 != 2:
                continue
        elif 20 < inum <= 50:
            if inum % 12 != 2:
                continue
        elif 50 < inum <= 100:
            if inum % 18 != 2:
                continue
        elif 100 < inum <= 180:
            if inum % 24 != 2:
                continue
        else:
            if inum % 24 != 2:
                continue
        inum = int(inum) + 1

        imglist = getFileList(input_dir, [], 'jpg')
        print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')

        for imgpath in imglist:
            imgname = os.path.splitext(os.path.basename(imgpath))[0]

            # model_index = 155
            model_index = 4023
            model = get_model(model_index=model_index)
            # model = models.alexnet(pretrained=True)

            weights = torch.load("weights/model_epoch_" + str(inum) + ".pth")

            """PyTorch 多 GPU 训练保存模型权重 key 多了 ‘module.‘  https://blog.csdn.net/Fly2Leo/article/details/122352956"""
            i = 0
            pretrained_dict = {}
            for k, v in weights.items():
                i = i + 1
                if 10 < i <= 20:
                    print("旧的参数名", k)
                new_k = k.replace('module.', '') if 'module' in k else k
                if 10 < i <= 20:
                    print("新的参数名", new_k)
                pretrained_dict[new_k] = v
            # model.load_state_dict(pretrained_dict)
            """#  还是报了错误，多出2个参数，Unexpected key(s) in state_dict: "total_ops", "total_params",
            # 解决办法是使用上面那个部分装载代码，对多出来的对这个进行更新"""
            model_dict = model.state_dict()
            # 将pretrained_dict里不属于model_dict的键剔除掉
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print("-----------------------pretrained_dict下面开始loop，pretrained_dict---------------------------------")
            if pretrained_dict == {}:
                print("--------------------\n--------------------\n--------------------\n--------------------\n")
                print("--------------------\n--------------------\n--------------------\n--------------------\n")
                print(
                    "-------------------------pretrained_dict为空，没有装载成功，使用的是ImageNet的权重-----------------------------------------------------\n")
                print("--------------------\n--------------------\n--------------------\n--------------------\n")
                print("--------------------\n--------------------\n--------------------\n--------------------\n")
            else:
                for k, v in pretrained_dict.items():
                    # if 10 < i <= 20:
                    print("pretrained_dict不为空", k)
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            # 加载我们真正需要的state_dict
            model.load_state_dict(model_dict)

            # 定义提取第几层的feature map
            for k1 in range(100):
                k1 = k1
                # image_info = get_image_info(image_dir)
                image_info = get_image_info(imgpath)

                model_layer= list(model.children())
                model_layer=model_layer[0]  #这里选择model的第一个Sequential()
                # model_layer=model_layer[:-1]  #这里选择model的第一个Sequential()
                # model_layer=model_layer[:-14]  #这里选择model的第一个Sequential()

                feature_map = get_k_layer_feature_map(model_layer, k1, image_info)
                # print(feature_map)
                endsize = show_feature_map(feature_map, imgname, inum, k=k1)
                if endsize < 70:
                    break









