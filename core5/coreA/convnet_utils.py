import torch.nn as nn

from coreA.basic import ConvBN
from coreA.blocks import DBB, OREPA_1x1, OREPA, OREPA_LargeConvBase, OREPA_LargeConv
from coreA.blocks_repvgg import RepVGGBlock, RepVGGBlock_OREPA

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
        from coreA.repmodels.resnet import create_Res18
        model = create_Res18()
    elif arch == 'ResNet-34':
        from coreA.repmodels.resnet import create_Res34
        model = create_Res34()
    elif arch == 'ResNet-50':
        from coreA.repmodels.resnet import create_Res50
        model = create_Res50()
    elif arch == 'regnet_base':
        from coreA.repmodels.regnet_base import regnetx_032
        model = regnetx_032()
    elif arch == 'regnet_2resOCA':
        from coreA.repmodels.regnet_2resOCA import regnetx_032
        model = regnetx_032()
    elif arch == 'regnet_res2OCA':
        from coreA.repmodels.regnet_res2OCA import regnetx_032
        model = regnetx_032()
    else:
        raise ValueError('这个主干没有，输入的网络错误')
    return model