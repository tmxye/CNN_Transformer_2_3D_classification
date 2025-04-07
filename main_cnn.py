import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
import torch.backends.cudnn
from torchvision import models
from torchvision import transforms
from utilsA import GradCAM, show_cam_on_image, show_cam_on_imageL
from Evison import Display, show_network
from classification_choice_YXY_2 import get_model
from time import sleep


import cv2

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# input_dir = './data_4/test/CT2'
input_dir = './data_4/test1/CT2'
# input_dir = './data_9_COVID2/test'
# input_dir = './data_14C/orgin'
# input_dir = './data_11_CT/CAM'


img_path = "1.png"


class ResizeTransform:
    def __init__(self, height=7, width=7):
        self.height = height
        self.width = width

    def __call__(self, x):
        # print("进来变化之前的x", x.shape)
        if isinstance(x, tuple):
            self.height = x[1]
            self.width = x[2]
            x = x[0]
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        # print("变化之后的result", result.shape)

        return result

class ResizeTransformNAT:
    def __init__(self, height=7, width=7):
        self.height = height
        self.width = width

    def __call__(self, x):
        # print("进来变化之前的x", x.shape)
        if isinstance(x, tuple):
            self.height = x[1]
            self.width = x[2]
            x = x[0]
        # result = x.reshape(x.size(0),
        #                    self.height,
        #                    self.width,
        #                    x.size(2))

        result = x

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        # print("变化之后的result", result.shape)

        return result


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    print("layer_name", layer_name)
    return layer_name



def main(gpu, layer_name):
    if not os.path.exists('./Grad_CAM/i_CT/'):
        os.mkdir('./Grad_CAM/i_CT/')
    #     https://blog.csdn.net/m0_64829783/article/details/123849755
    # break退出for循环          continue退出for循
    # # CT
    # for inum in range(0, 249):
    #     if inum <= 20:
    #         if inum % 3 != 0:
    #             continue
    #     elif 20 < inum <= 50:
    #         if inum % 5 != 0:
    #             continue
    #     elif 50 < inum <= 100:
    #         if inum % 5 != 0:
    #             continue
    #     elif 100 < inum <= 180:
    #         if inum % 6 != 0:
    #             continue
    #     else:
    #         if inum % 7 != 0:
    #             continue
    # PETCT
    for inum in range(0, 250):
    # for inum in range(0, 120):
        if inum <= 20:
            if inum % 3 != 0:
                continue
        elif 20 < inum <= 50:
            if inum % 5 != 0:
                continue
        elif 50 < inum <= 100:
            if inum % 7 != 0:
                continue
        elif 100 < inum <= 180:
            if inum % 7 != 0:
                continue
        else:
            if inum % 10 != 0:
                continue
    # # CXR
    # for inum in range(0, 249):
    #     if inum <= 20:
    #         if inum % 2 != 0:
    #             continue
    #     elif 20 < inum <= 50:
    #         if inum % 3 != 0:
    #             continue
    #     elif 50 < inum <= 100:
    #         if inum % 5 != 0:
    #             continue
    #     elif 100 < inum <= 180:
    #         if inum % 7 != 0:
    #             continue
    #     else:
    #         if inum % 10 != 0:
    #             continue
        inum = int(inum) + 1
        model_index = 106
        weights = torch.load("weights/model_epoch_" + str(inum) + ".pth")

        # if not os.path.exists('./Grad_CAM/i_CT/CSV/'):
        #     os.mkdir('./Grad_CAM/i_CT/CSV/')
        if not os.path.exists('./Grad_CAM/i_CT/CAM' + str(inum) + '/'):
            os.mkdir('./Grad_CAM/i_CT/CAM' + str(inum) + '/')
        if not os.path.exists('./Grad_CAM/i_CT/JPG' + str(inum) + '/'):
            os.mkdir('./Grad_CAM/i_CT/JPG' + str(inum) + '/')
        if not os.path.exists('./Grad_CAM/i_CT/JPGL' + str(inum) + '/'):
            os.mkdir('./Grad_CAM/i_CT/JPGL' + str(inum) + '/')


        torch.backends.cudnn.benchmark = True
        model = get_model(model_index = model_index, num_classes = 2)
        # model = get_model(model_index = model_index, num_classes = 1000)
        # model = get_model(model_index = model_index, num_classes = 15)
        # model.apply(inplace_relu)
        model = model.cuda()
        show_network(model)
        # print(model)


        layer_name = '_DenseBlockD.DenseLayerP.conv3'      #    4449 dense_Trans  4455 dense_Trans 4453




        # layer_name = 'stages.3.0.conv_1x1.conv'      #    131 RepVGG
        # layer_name = '_blocks.15._project_conv'      #    80 Eff b0
        # layer_name = '_blocks.25._swish'      #             83 Eff  b3
        # layer_name = '_blocks.31._swish'      #             84 Eff  b4
        # layer_name = 'blocks.9.0.conv'      # 102     GhostNet
        # layer_name = 'stages.3.2.convs.2'      # 1003     edgenext_small
        # layer_name = 'layer4.2.bottleneck.7'      #  2510 ResNet50_dy
        # layer_name = 'layer4.1.conv2'      #  2512 resnetv1sn18     #  231 resnet
        # layer_name = 'NAT.level2res_O.blocks.0.mlp.drop'      #  9805 NAT.      # 使用不了
        # layer_name = 'features.denseblock4.denselayer16.conv2'      #  4023 densenet
        # layer_name = 'layer4.1.conv1'      #   2522 resnetv2sn18
        # layer_name = 'stage4.1.residual.5'      #  261  preactresnet18
        # layer_name = 'stages.4.0.conv3_1x1.conv'      #  275  gernet_s
        # layer_name = 's4.b2.conv3.conv'      #  9801     RegNetx032
        # layer_name = 'stages.3.conv_transition.conv'      #  281  cspresnet50      287 cspresnext50   288
        # layer_name = 'features.denseblock_5.denselayer_8.conv_2.conv'      #  951  condensenet
        # layer_name = 'stages.4.blocks.3.conv2.conv'      #  296  darknet53
        # layer_name = 'base.14.layers.3.conv'      #  4100  hardnet68
        # layer_name = 'base.14.layers.3.layer1.conv'      #  4104  hardnet68ds
        # layer_name = 'features.17.conv.3'      #  62  mobilenet_v2
        # layer_name = 'conv4.1.pointwise.2'      #  66  mobilenet
        # layer_name = 'features.11.conv.8'      #  661  mobilenetv3
        # layer_name = 'stage4.3.branch2.7'      #  72  shufflenet_v2_x1_0
        # layer_name = 'stage4.1.residual.5'      #  91  seresnet18
        # layer_name = 'level5.root.conv'      #  915  dla34          911 dla60_res2net
        # layer_name = 'layer4.2.conv3'      #  945  se_resnext50
        # layer_name = 'stages.3.2.conv3'      #  975  nfnet_f0       # 981 nf_regnet_b0      #  979  dm_nfnet_f0
        # layer_name = 'stages.3.1.conv3'      #  985  nf_seresnet26
        # layer_name = 'block5.8.1'      #  101  EffNet
        # layer_name = 'layer4.2.conv3'      #  109  res2net101
        layer_name = 'transitions.2.1.conv'      #  106  create_RepLKNet31B
        hierarchy = layer_name.split('.')
        print("hierarchy", hierarchy)       # hierarchy ['', 'blocks', '15', '', 'project', 'conv']
        target_layer = model._modules[hierarchy[0]]
        print("yolo_target_layer", target_layer)
        for h in hierarchy[1:]:
            target_layer = target_layer._modules[h]
            tname_layers = target_layer
        tname_layers = [tname_layers]
        print("find_yolo_target_layer", tname_layers)


        """
        0 ALex
        """
        # tname_layers = [model.features][-1]      #    0 Alex
        # tname_layers = [model._blocks.15][-1]      #    80 Eff b0
        # tname_layers = [model.layer4][-1]      #    236 ResNext
        # tname_layers = [model.features.denseblock_5.denselayer_8.conv_2.conv]         # 951     CondenseNet
        # tname_layers = [model.s4.b2.conv3.conv]               # 9801     RegNetx032



        """
        200 DPN 
        """
        # tname_layers = [model.features.conv5_3.c3x3_b.conv]      # 299 DPN
        # tname_layers = [model.features.conv5_3.c3x3_b.conv]      # 4023 DPN68
        # tname_layers = [model.features.conv5_3.c1x1_c.conv]      # 4023 DPN68
        # 下面2个都是最后层，会很艳         不好
        # tname_layers = [model.features.conv5_bn_ac.bn.act]      # 4023 DPN68
        # tname_layers = [model.features.conv5_bn_ac.bn]      # 4023 DPN68






        """
        234 ResNet 
        """
        # # tname_layers = [model.layer4][-1]      # 234 ResNet           # 这样才能运行，但是没效果
        # # yolov5        model_23_cv3_act
        # # layer_name = 'layer4_2_conv3'      # 231 ResNet
        # layer_name = 'layer4_2_conv3'      # 234 ResNet
        # # layer_name = 'layer4_2_relu'      # 234 ResNet
        # hierarchy = layer_name.split('_')
        # target_layer = model._modules[hierarchy[0]]._modules[hierarchy[1]]
        # print("yolo_target_layer", target_layer)
        # for h in hierarchy[2:]:
        #     target_layer = target_layer._modules[h]
        #     tname_layers = target_layer
        # tname_layers = [tname_layers]
        # print("find_yolo_target_layer", tname_layers)







        """
        4740     Rep_NAT_DenseNet
        """
        # # tname_layers = [model._DenseBlockD.DenseLayerP.convB]
        # # layer_name = 'NAT.level2res_O.blocks.0.mlp.fc2'      #
        # # layer_name = 'NAT.norm'      #
        # layer_name = 'NAT.level2res_O.blocks.0.mlp.drop'      #
        # # layer_name = 'NAT.level2res_O.blocks.0.drop_path'      #
        # # layer_name = 'NAT.level2ress'      #
        # # NAT.level2res_O.blocks.0.norm2
        # hierarchy = layer_name.split('.')
        # print("hierarchy", hierarchy)       # hierarchy ['', 'blocks', '15', '', 'project', 'conv']
        # target_layer = model._modules[hierarchy[0]]
        # print("yolo_target_layer", target_layer)
        # for h in hierarchy[1:]:
        #     target_layer = target_layer._modules[h]
        #     tname_layers = target_layer
        # tname_layers = [tname_layers]
        # print("find_yolo_target_layer", tname_layers)


        """
        1040     Swin
        """
        # # tname_layers = [model.layers[-1]]     # 等价于layer_name = 'layers.3'
        # # tname_layers = [model.norm]
        # # print("tname_layers", tname_layers)
        # # # layers.3.blocks.1.mlp.fc2
        #
        # # layer_name = 'layers.3'      #
        # # layer_name = 'layers.3.blocks.1'      #
        # # layer_name = 'layers.3.blocks.1.norm1'      #  norm1 norm2 效果不行
        # # layer_name = 'layers.3.blocks.1.drop_path'      # 效果不错
        # layer_name = 'layers.3.blocks.1.mlp.drop'      #效果最好
        # # NAT.level2res_O.blocks.0.norm2
        # hierarchy = layer_name.split('.')
        # print("hierarchy", hierarchy)       # hierarchy ['', 'blocks', '15', '', 'project', 'conv']
        # target_layer = model._modules[hierarchy[0]]
        # print("yolo_target_layer", target_layer)
        # for h in hierarchy[1:]:
        #     target_layer = target_layer._modules[h]
        #     tname_layers = target_layer
        # tname_layers = [tname_layers]
        # print("find_yolo_target_layer", tname_layers)


        """
        1084     NAT_base
        """
        # layer_name = 'stages.3.2.dwconv'      #    107     ConX
        # layer_name = 'levels.3.blocks.4.mlp.drop'      #    1084     NAT_base
        # # layer_name = 'levels.3.blocks.4.drop_path'      #
        # hierarchy = layer_name.split('.')
        # print("hierarchy", hierarchy)       # hierarchy ['', 'blocks', '15', '', 'project', 'conv']
        # target_layer = model._modules[hierarchy[0]]
        # print("yolo_target_layer", target_layer)
        # for h in hierarchy[1:]:
        #     target_layer = target_layer._modules[h]
        #     tname_layers = target_layer
        # tname_layers = [tname_layers]
        # print("find_yolo_target_layer", tname_layers)



        # # by YXY        多gpu并行
        # if gpu:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        #     gpus = list(range(len(gpu.split(','))))
        #     print("yes")
        # else:
        #     gpus = [0]  # [1,2]
        # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        # gpus = [0, 1]  # [1,2]
        # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        # model.load_state_dict(torch.load("weight/model_epoch_200.pth"))
        # model.load_state_dict(torch.load("weight/model_epoch_200.pth"), False)        # 这个方案可以使用，但是是没加载的意思


        """这个是根据blog， 'DataParallel' object has no attribute 'copy' 解决方案     https://blog.csdn.net/qq_33768643/article/details/105553743"""
        # model.load_state_dict(torch.load("weight/model_epoch_200.pth", map_location=lambda storage, loc: storage))
        # model.load_state_dict(torch.load("weight/model_epoch_200.pth", map_location=lambda storage, loc: storage).module.state_dict())


        """这个也是可以使用，也是没加载的意思"""
        # model_dict = model.state_dict()
        # # 将pretrained_dict里不属于model_dict的键剔除掉
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 更新现有的model_dict
        # model_dict.update(pretrained_dict)
        # # 加载我们真正需要的state_dict
        # model.load_state_dict(model_dict)

        """PyTorch 多 GPU 训练保存模型权重 key 多了 ‘module.‘  https://blog.csdn.net/Fly2Leo/article/details/122352956"""
        i = 0
        pretrained_dict = {}
        for k, v in weights.items():
            i = i + 1
            if 10<i<=20:
                print("旧的参数名", k)
            new_k = k.replace('module.', '') if 'module' in k else k
            if 10<i<=20:
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
            print("-------------------------pretrained_dict为空，没有装载成功，使用的是ImageNet的权重-----------------------------------------------------\n")
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


        '''
        Resnet18 and 50: model.layer4[-1]
        VGG and densenet161: model.features[-1]
        mnasnet1_0: model.layers[-1]
        ViT: model.blocks[-1].norm1
        '''
        # target_layers = [model.features]      # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1039360 and 1024x1000)
        # target_layers = [model.features.denseblock4.denselayer16]     # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1039360 and 1024x1000)
        target_layers = get_last_conv_name(model) if layer_name is None else layer_name

        data_transform = transforms.Compose(
            [  # transforms.Resize((224, 224)),  # 缩放     # 不要， 因为这样没法执行下面一条语句 img = np.array(img, dtype=np.uint8)
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        # load image
        # img_path = "both.png"

        print("-----------------------下面开始loop，将文件夹的热力图数据全部取出---------------------------------")
        org_img_folder = input_dir

        # 检索文件
        # img_folder = org_img_folder
        # imglist = [os.path.join(nm) for nm in os.listdir(img_folder) if nm[-3:] in ['jpg', 'png', 'gif']]
        ## print(img_list) 将所有图像遍历并存入一个列表
        imglist = getFileList(org_img_folder, [], 'jpg')
        print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')

        # 下面通过loop，将文件夹的数据全部取出
        # visualization = show_cam_on_image(img.astype(dtype=npz.float32) / 255., grayscale_cam, use_rgb=True)
        gray_cam = []
        visual = []
        visualL = []
        cam_name = []
        vis_name = []
        visL_name = []
        for i in range(len(imglist)):
            gray_cam.append([])
            visual.append([])
            visualL.append([])
            cam_name.append([])
            vis_name.append([])
            visL_name.append([])

        i = -1
        for imgpath in imglist:
            i = i + 1
            imgname = os.path.splitext(os.path.basename(imgpath))[0]
            print("---------------------------imgname---------------------------------", imgname)
            # img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

            # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img_path = input_dir + '/' + imgname + '.jpg'
            img = Image.open(img_path).convert('RGB')
            # img = Image.open("./data_4/test/CT1/1.jpg").convert('RGB')
            img = img.resize((224, 224), Image.ANTIALIAS)  # # 第二个参数：# Image.NEAREST ：低质量# Image.BILINEAR：双线性# Image.BICUBIC ：三次样条插值# Image.ANTIALIAS：高质量
            img = np.array(img, dtype=np.uint8)

            # [N, C, H, W]
            img_tensor = data_transform(img)
            # img_tensorh和input_tensor都改进为GPU版本
            print("img_tensor", img_tensor.shape)
            img_tensor = img_tensor.cuda()
            # expand batch dimension
            input_tensor = torch.unsqueeze(img_tensor, dim=0)
            # input_tensor = torch.cat([input_tensor, input_tensor, input_tensor, input_tensor], dim=0)
            input_tensor = input_tensor.cuda()
            print("input_tensor", input_tensor.shape)

            # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)           # 这个是原始的layer层，自己需要手动给的那种
            # cam = GradCAM(model=model, target_layers=[model.features][-1], use_cuda=True)         # https://blog.csdn.net/rensweet/article/details/123263812
            # cam = GradCAM(model=model, target_layers=[model.features.denseblock4.denselayer16.conv2], use_cuda=True)         # 改了上面的参数，这个也可以运行了，现在重新恢复原先的代码
            end_name = '[model.{}]'.format(target_layers)
            print("end_name", end_name)  # 虽然得到一样的结果，但是没法使用，这个可能是代码的限制

            # cam = GradCAM(model=model, target_layers=[model.features.denseblock4.denselayer16][-1], use_cuda=True)

            cam = GradCAM(model=model, target_layers=tname_layers, use_cuda=True)
            # if Transformer:
            # cam = GradCAM(model=model, target_layers=tname_layers, use_cuda=True, reshape_transform=ResizeTransform())
            # cam = GradCAM(model=model, target_layers=tname_layers, use_cuda=True, reshape_transform=ResizeTransformNAT())

            # target_category = 281  # tabby, tabby cat
            # target_category = 1  # tabby, tabby cat
            target_category = None  # tabby, tabby cat            # 选定目标类别，如果不设置，则默认为分数最高的那一类
            # target_category = 254  # pug, pug-dog

            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            print("grayscale_camA", grayscale_cam.shape)

            grayscale_cam = grayscale_cam[0, :]
            # print("grayscale_camB", grayscale_cam.shape)
            # print(grayscale_cam.shape)
            # txt_name = './Grad_CAM/i_CT/CSV/' + str(imgname) + '.csv'
            # ### grayscale_cam.to_csv("热力图需要的参数.csv", index=False)       # AttributeError: 'numpy.ndarray' object has no attribute 'to_csv'
            # ### https://blog.csdn.net/weixin_39731682/article/details/110681877
            # np.savetxt(txt_name, grayscale_cam, delimiter=',')

            # plt.imshow(grayscale_cam)
            # txt_name = './Grad_CAM/i_CT/CAM/' + str(imgname) + '.jpg'
            # plt.savefig(txt_name, dpi=300)
            txt_name = './Grad_CAM/i_CT/CAM' + str(inum) + '/' + str(imgname)
            gray_cam[i].append(grayscale_cam)
            cam_name[i].append(txt_name)

            image = (Image.fromarray(np.squeeze((np.array(gray_cam[i])), axis=0)*255)).convert('RGB')
            image.save(str(np.squeeze(cam_name[i], axis=0)) + ".jpg")


            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
            # visualization = show_cam_on_image(img.astype(dtype=np.float32), grayscale_cam * 255, use_rgb=True)
            # if Transformer:
            # visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)

            #### https://blog.csdn.net/weixin_27183235/article/details/112016159
            # YXY_cam = np.loadtxt("热力图需要的参数.csv", dtype=float, delimiter=',')
            # YXY_cam = np.loadtxt("1_40.csv", dtype=float, delimiter=',')
            # visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., YXY_cam, use_rgb=True)
            # plt.imshow(visualization)
            # txt_name = './Grad_CAM/i_CT/JPG/' + str(imgname) + '.jpg'
            # plt.savefig(txt_name, dpi=300)
            txt_name = './Grad_CAM/i_CT/JPG' + str(inum) + '/' + str(imgname)
            visual[i].append(visualization)
            vis_name[i].append(txt_name)

            image = (Image.fromarray(np.squeeze((np.array(visual[i])), axis=0)))
            image.save(str(np.squeeze(vis_name[i], axis=0)) + ".jpg")


            visualization = show_cam_on_imageL(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
            txtL_name = './Grad_CAM/i_CT/JPGL' + str(inum) + '/' + str(imgname)
            visualL[i].append(visualization)
            visL_name[i].append(txtL_name)
            image = (Image.fromarray(np.squeeze((np.array(visualL[i])), axis=0)))
            image.save(str(np.squeeze(visL_name[i], axis=0)) + ".jpg")

            # print("grayscale_cam", grayscale_cam.shape)
            # x = np.squeeze((np.array(gray_cam[i])), axis=0)
            # print("gray_cam[i]", x.shape)
            # plt.imshow(x)
            # plt.savefig(txt_name, dpi=300)

            # print("visualization", visualization.shape)
            # x = np.squeeze((np.array(visual[i])), axis=0)
            # print("visual[i]", x.shape)
            # plt.imshow(x)
            # plt.savefig(str(np.squeeze(vis_name[i], axis=0)), dpi=300)

            # plt.show()

        print("运行结束，开始批量导出图像")

        """
        # # for key in globals().keys():
        # #     if not key.startswith("__"):  # 排除系统内建函数
        # #         globals().pop(key)
        # import re
        # # for x in dir():
        # #     # if not re.match('^__', x) and x != "re":
        # #     exec(" ".join(("del", x)))
        #
        # for key in list(globals().keys()):
        #     if (not key.startswith("__")) and (key != "key"):
        #         # print("keykey", key)
        #         if key == "imglist" or "gray_cam[i]" or "torch.cuda.empty_cache()" or "cam_name[i]" or "vis_name[i]" or "visual[i]" or "np" or "plt" or "torch":
        #         # if key == "":
        #             print("这个key参数是需要的", key)
        #         else:
        #             globals().pop(key)
        #             print(key)
        # del key
    
        # torch.cuda.empty_cache()
        # # from time import sleep
        # # sleep(300)  # 强制等待300秒再执行下一步
    
        # # 创建数组（3维）
        # # a = np.arange(100).reshape((10, 5, 2))
        # a = gray_cam
        # np.save(file="./Grad_CAM/i_CT/JPG.npy", arr=a)
        # b = np.load(file="./Grad_CAM/i_CT/JPG.npy")
        # # sleep(300)  # 强制等待300秒再执行下一步
        #
        # # http://www.icodebang.com/article/297520
        # # print("img", (np.array(img)).shape)     # img (224, 224, 3)
        # print("img", (np.array(visual[i])).shape)       # img (1, 224, 224, 3)
        # image = np.squeeze((np.array(visual[i])), axis=0)
        # # image = np.expand_dims(image, axis=2)     # https://blog.csdn.net/Magician0619/article/details/107516116
        # # image = np.concatenate((image, image, image), axis=-1)  # -1则是最后一个维度
        # print("image", image.shape)
        # # imgA = Image.fromarray(np.uint8(image*255))
        # imgA = Image.fromarray(image)
        # print("imgA", (np.array(imgA)).shape)
        # imgA.save("./Grad_CAM/i_CT/JPG/1.jpg")  # 将图片保存为1.jpg
    
    
        # for i in range(len(imglist)):
        #     print("gray_cam[i]", gray_cam[i])
        #     plt.imshow(np.squeeze((np.array(gray_cam[i])), axis=0))
        #     plt.savefig(str(np.squeeze(cam_name[i], axis=0)), dpi=300)
        #     plt.imshow(np.squeeze((np.array(visual[i])), axis=0))
        #     plt.savefig(str(np.squeeze(vis_name[i], axis=0)), dpi=300)
    
            # plt.imshow(np.squeeze((np.array(gray_cam[i])), axis=0))
            # plt.savefig(str(np.squeeze(cam_name[i], axis=0)) + ".jpg", dpi=300)
            # plt.imshow(np.squeeze((np.array(visual[i])), axis=0))
            # plt.savefig(str(np.squeeze(vis_name[i], axis=0)) + ".jpg", dpi=300)
    
        for i in range(len(imglist)):
            print("gray_cam[i]", gray_cam[i])
            image = (Image.fromarray(np.squeeze((np.array(visual[i])), axis=0)))
            image.save(str(np.squeeze(vis_name[i], axis=0)) + ".jpg")
    
            # image = (Image.fromarray(np.uint8(np.squeeze((np.array(gray_cam[i])), axis=0)*255)))
            image = (Image.fromarray(np.squeeze((np.array(gray_cam[i])), axis=0)*255)).convert('RGB')
            image.save(str(np.squeeze(cam_name[i], axis=0)) + ".jpg")
    
    
            # plt.imshow(np.squeeze((np.array(gray_cam[i])), axis=0))
            # plt.savefig(str(np.squeeze(cam_name[i], axis=0)) + ".jpg", dpi=300)
            # plt.imshow(np.squeeze((np.array(visual[i])), axis=0))
            # plt.savefig(str(np.squeeze(vis_name[i], axis=0)) + ".jpg", dpi=300)
        """



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



if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0', help='ID of GPUs to use, eg. 1,3')
    p.add_argument('--layer-name', type=str, default=None, help='last convolutional layer name')
    args = p.parse_args()

    main(args.gpu, args.layer_name)
