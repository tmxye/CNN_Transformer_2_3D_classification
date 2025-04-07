import torch.nn as nn

from core5.basic import ConvBN
from core5.blocks import DBB, OREPA_1x1, OREPA, OREPA_LargeConvBase, OREPA_LargeConv
from core5.blocks_repvgg import RepVGGBlock, RepVGGBlock_OREPA

CONV_BN_IMPL = 'base'

DEPLOY_FLAG = False

def choose_blk(kernel_size):
    if CONV_BN_IMPL == 'OREPA':
        # print("使用的OREPA")
        if kernel_size == 1:
            blk_type = OREPA_1x1
        elif kernel_size >= 7:
            blk_type = OREPA_LargeConv
        else:
            blk_type = OREPA
    elif CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        blk_type = ConvBN
    elif CONV_BN_IMPL == 'DBB':
        blk_type = DBB
    elif CONV_BN_IMPL == 'RepVGG':
        blk_type = RepVGGBlock
    elif CONV_BN_IMPL == 'OREPA_VGG':
        blk_type = RepVGGBlock_OREPA
    else:
        raise NotImplementedError
    
    return blk_type

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None):
    if assign_type is not None:
        blk_type = assign_type
    else:
        blk_type = choose_blk(kernel_size)
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None):
    if assign_type is not None:
        blk_type = assign_type
    else:
        blk_type = choose_blk(kernel_size)
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    # padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU())
                    # padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)
                    padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU(inplace=True))
                    # padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU(inplace=False))

def switch_conv_bn_impl(block_type):
    global CONV_BN_IMPL
    CONV_BN_IMPL = block_type

def switch_deploy_flag(deploy):
    global DEPLOY_FLAG
    DEPLOY_FLAG = deploy
    print('deploy flag: ', DEPLOY_FLAG)

def build_model(arch):
    if arch == 'ResNet-18':
        from core5.repmodels.resnet import create_Res18
        model = create_Res18()
    elif arch == 'ResNet-34':
        from core5.repmodels.resnet import create_Res34
        model = create_Res34()
    elif arch == 'ResNet-50':
        from core5.repmodels.resnet import create_Res50
        model = create_Res50()
    elif arch == 'ResNet-101':
        from core5.repmodels.resnet import create_Res101
        model = create_Res101()
    elif arch == 'RepVGG-A0':
        from core5.repmodels.repvgg import create_RepVGG_A0
        model = create_RepVGG_A0()
    elif arch == 'RepVGG-A1':
        from core5.repmodels.repvgg import create_RepVGG_A1
        model = create_RepVGG_A1()
    elif arch == 'RepVGG-A2':
        from core5.repmodels.repvgg import create_RepVGG_A2
        model = create_RepVGG_A2()
    elif arch == 'RepVGG-B1':
        from core5.repmodels.repvgg import create_RepVGG_B1
        model = create_RepVGG_B1()
    elif arch == 'ResNet-18-1.5x':
        from core5.repmodels.resnet import create_Res18_1d5x
        model = create_Res18_1d5x()
    elif arch == 'ResNet-18-2x':
        from core5.repmodels.resnet import create_Res18_2x
        model = create_Res18_2x()
    elif arch == 'ResNeXt-50':
        from core5.repmodels.resnext import create_Res50_32x4d
        model = create_Res50_32x4d()
    #elif arch == 'RegNet-800MF':
        #from core5.repmodels.regnet import create_Reg800MF
        #model = create_Reg800MF()
    #elif arch == 'ConvNext-T-0.5x':
        #from core5.repmodels.convnext import convnext_tiny_0d5x
        #model = convnext_tiny_0d5x()
    elif arch == 'DenseNet121':
        from core5.repmodels.densenet import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNet121_A':
        from core5.repmodels.densenet_A import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNAT121':
        from core5.repmodels.denseNAT import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNAT_2res':
        from core5.repmodels.denseNAT_2res import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNAT_2res1':
        from core5.repmodels.denseNAT_2res1 import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNAT_2resden':
        from core5.repmodels.denseNAT_2resden import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNAT_2resOCA':
        from core5.repmodels.denseNAT_2resOCA import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNAT_2SOCA':
        from core5.repmodels.denseNAT_2SOCA import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNAT_res2OCA':
        from core5.repmodels.denseNAT_res2OCA import se_densenet121
        model = se_densenet121()
    elif arch == 'DenseNAT_2OCA':
        from core5.repmodels.denseNAT_2OCA import se_densenet121
        model = se_densenet121()
    elif arch == 'regnet_base':
        from core5.repmodels.regnet_base import regnetx_032
        model = regnetx_032()
    elif arch == 'regnet_2resOCA':
        from core5.repmodels.regnet_2resOCA import regnetx_032
        model = regnetx_032()
    elif arch == 'regNAT':
        from core5.repmodels.regnet_NAT import regnetx_032
        model = regnetx_032()
    elif arch == 'regENAT':
        from core5.repmodels.regnet_ENAT import regnetx_032
        model = regnetx_032()
    elif arch == 'LDenseNAT_2resOCA':
        from core5.repmodels.LdenseNAT_2resOCA import se_densenet121
        model = se_densenet121()
    elif arch == 'LDenseNAT_2SOCA':
        from core5.repmodels.LdenseNAT_2SOCA import se_densenet121
        model = se_densenet121()
    elif arch == 'LDenseNAT_res2OCA':
        from core5.repmodels.LdenseNAT_res2OCA import se_densenet121
        model = se_densenet121()
    elif arch == 'LDenseNAT_2OCA':
        from core5.repmodels.LdenseNAT_2OCA import se_densenet121
        model = se_densenet121()
    elif arch == 'FDenseNAT_2resOCA':
        from core5.repmodels.FdenseNAT_2resOCA import se_densenet121
        model = se_densenet121()
    elif arch == 'FDenseNAT_2SOCA':
        from core5.repmodels.FdenseNAT_2SOCA import se_densenet121
        model = se_densenet121()
    elif arch == 'FDenseNAT_res2OCA':
        from core5.repmodels.FdenseNAT_res2OCA import se_densenet121
        model = se_densenet121()
    elif arch == 'FDenseNAT_2OCA':
        from core5.repmodels.FdenseNAT_2OCA import se_densenet121
        model = se_densenet121()
    elif arch == 'FDenseNAT_2SOCA_MLP':
        from core5.repmodels.FdenseNAT_2SOCA_MLP import se_densenet121
        model = se_densenet121()
    elif arch == 'FDenseNAT_2SOCA_MLPL':
        from core5.repmodels.FdenseNAT_2SOCA_MLPL import se_densenet121
        model = se_densenet121()
    else:
        raise ValueError('这个主干没有，输入的网络错误')
    return model