import torch
import torch.nn as nn

from core2.CA_Net.layers.modules import conv_block, conv_blockYXY, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3
from core2.CA_Net.layers.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock, MultiAttentionBlockYXY
from core2.CA_Net.layers.channel_attention_layer import SE_Conv_Block
from core2.CA_Net.layers.scale_attention_layer import scale_atten_convblock
from core2.CA_Net.layers.nonlocal_layer import NONLocalBlock2D


class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
    # def __init__(self, model, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
    #              nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Comprehensive_Atten_Unet, self).__init__()
        # # by YXY
        # # Define model
        # if args.data == 'Fetus':
        #     args.num_input = 1
        #     args.num_classes = 3
        #     args.out_size = (256, 256)
        # elif args.data == 'ISIC2018':
        #     args.num_input = 3
        #     args.num_classes = 2
        #     args.out_size = (224, 300)
        # self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        # self.out_size = args.out_size
        # self.out_size = (256, 256)
        # self.out_size = (1024, 1024)
        self.out_size = (224, 300)

        filters = [64, 128, 256, 512, 1024]
        filtersA = [64, 128, 256, 512, 1024, 2048]
        filters = [int(x / self.feature_scale) for x in filters]
        filtersA = [int(x / self.feature_scale) for x in filtersA]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])

        # # by YXY        修改的和原始文章代码一样
        # self.attentionblock1 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
        #                                            nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        # self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
        #                                            nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        # self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
        #                                            nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        # self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)
        # # self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4])
        # # by YXY还是报了相同错误RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'; but device 1 does not equal 0 (while checking arguments for cudnn_convolution)
        #


        # by YXY        修改的和原始文章代码一样        然后在这个基础上添加密集连接的思想，多尺度
        self.attentionblock1 = MultiAttentionBlockYXY(in_size=filters[1], gate_size=filters[1], inter_size=filters[0],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        # self.attentionblock1B = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[0],
        #                                            nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock2 = MultiAttentionBlockYXY(in_size=filters[2], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlockYXY(in_size=filters[3], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)
        # self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4])
        # by YXY还是报了相同错误RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'; but device 1 does not equal 0 (while checking arguments for cudnn_convolution)


        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    # #     by YXY
    # 问题是 RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'; but device 1 does not equal 0 (while checking arguments for cudnn_convolution)
    # #     self.l1 = nn.Sequential(*list(model.children())[:-1]).to('cuda:0')
    #     self.l1 = nn.Sequential(*list(self.children())[:-1]).to('cuda:0')
    #     self.last = list(self.children())[-1]

    # #############################by YXY ,实现募集连接  ################################
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.Skip1 = conv_blockYXY(self.in_channels, filters[0])
        self.Skip2 = conv_blockYXY(self.in_channels, filters[1])
        self.Skip3 = conv_blockYXY(self.in_channels, filters[2])


    def forward(self, inputs):
        # Feature Extraction
        # # by YXY
        # inputs = self.l1(inputs)
        # inputs = inputs.view(inputs.size()[0], -1)
        # inputs = self.last(inputs)
        # inputs = inputs.to('cuda:0')

        conv1 = self.conv1(inputs)
        # conv1 = conv1.to('cuda:0')
        maxpool1 = self.maxpool1(conv1)
        # maxpool1 = maxpool1.to('cuda:0')

        conv2 = self.conv2(maxpool1)
        # conv2 = conv2.to('cuda:0')
        maxpool2 = self.maxpool2(conv2)
        # maxpool2 = maxpool2.to('cuda:0')

        conv3 = self.conv3(maxpool2)
        # conv3 = conv3.to('cuda:0')
        maxpool3 = self.maxpool3(conv3)
        # maxpool3 = maxpool3.to('cuda:0')

        conv4 = self.conv4(maxpool3)
        # conv4 = conv4.to('cuda:0')
        maxpool4 = self.maxpool4(conv4)
        # maxpool4 = maxpool4.to('cuda:0')

        # Gating Signal Generation
        center = self.center(maxpool4)
        # center = center.to('cuda:0')

    # #############################by YXY ,实现募集连接  ################################
        inputsSkip = self.Skip1(inputs)
        # inputsSkip = self.maxpool(inputsSkip)
        inputsSkip1 = torch.cat([inputsSkip, conv1], 1)
        inputsSkip2 = self.maxpool(inputsSkip1)
        inputsSkip2 = torch.cat([inputsSkip2, conv2], 1)
        inputsSkip3 = self.maxpool(inputsSkip2)
        inputsSkip3 = torch.cat([inputsSkip3, conv3], 1)


        # Attention Mechanism
        # Upscaling Part (Decoder)
        # by YXY
        # 问题是  File "D:\YXY\TestEca\Models\networks\network.py", line 116, in forward               #     g_conv4 = self.nonlocal4_2(up4)
        up4 = self.up_concat4(conv4, center)
        up4A = up4.to('cuda:0')
        # up4 = up4.to('cuda:0')
        # g_conv4 = self.nonlocal4_2(up4)
        g_conv4 = self.nonlocal4_2(up4A)
        # g_conv4 = g_conv4.to('cuda:0')
        # by YXY 现在错误变换了，说明起效果了
        # 原本改为          #     g_conv4 = self.nonlocal4_2(up4).cuda()           错误还是有，错误变成了up4, att_weight4 = self.up4(g_conv4)
        # 继续修改to('cuda:0')，还是不行

        g_conv4A = g_conv4.cuda()
        up4, att_weight4 = self.up4(g_conv4A)
        # g_conv3, att3 = self.attentionblock3(conv3, up4)
        # conv3A = conv3.to('cuda:0')
        conv3A = inputsSkip3.to('cuda:0')
        up4A = up4.to('cuda:0')
        g_conv3, att3 = self.attentionblock3(conv3A, up4A)

        # atten3_map = att3.cpu().detach().numpy().astype(np.float)
        # atten3_map = ndimage.interpolation.zoom(atten3_map, [1.0, 1.0, 224 / atten3_map.shape[2],
        #                                                      300 / atten3_map.shape[3]], order=0)

        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)
        # g_conv2, att2 = self.attentionblock2(conv2, up3)
        # conv2A = conv2.to('cuda:0')
        conv2A = inputsSkip2.to('cuda:0')
        up3A = up3.to('cuda:0')
        g_conv2, att2 = self.attentionblock2(conv2A, up3A)

        # atten2_map = att2.cpu().detach().numpy().astype(np.float)
        # atten2_map = ndimage.interpolation.zoom(atten2_map, [1.0, 1.0, 224 / atten2_map.shape[2],
        #                                                      300 / atten2_map.shape[3]], order=0)


        # # #############最后一个模块
        # up2 = self.up_concat2(g_conv2, up3)
        # up2, att_weight2 = self.up2(up2)
        # # g_conv1, att1 = self.attentionblock1(conv1, up2)
        #
        # # atten1_map = att1.cpu().detach().numpy().astype(np.float)
        # # atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
        # #                                                      300 / atten1_map.shape[3]], order=0)
        # up1 = self.up_concat1(conv1, up2)
        # up1, att_weight1 = self.up1(up1)

        # #############修改后的最后一个模块 by  YXY
        up2 = self.up_concat2(g_conv2, up3)
        up2, att_weight2 = self.up2(up2)
        # g_conv1, att1 = self.attentionblock1(conv1, up2)
        # 原文里面添加了这个代码，但是在代码里面没有写出来， 先在下面写出这个模块
        # by YXY            有需要再打开这个最后一个，对应文章的SA4模块
        # conv1B = conv1.to('cuda:0')
        # conv1A = conv1
        conv1A = inputsSkip1.to('cuda:0')
        up2A = up2.to('cuda:0')
        g_conv1, att1 = self.attentionblock1(conv1A, up2A)
        # g_conv1B, att1B = self.attentionblock1B(conv1B, up2A)
        # print('g_conv1',g_conv1.size())
        # print(g_conv1B.size())

        # atten1_map = att1.cpu().detach().numpy().astype(np.float)
        # atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
        #                                                      300 / atten1_map.shape[3]], order=0)
        up1 = self.up_concat1(g_conv1, up2)
        up1, att_weight1 = self.up1(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)

        out = self.final(out)

        return out


