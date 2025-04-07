def repchoice(model):
    from core5.convnet_utils import switch_deploy_flag, switch_conv_bn_impl, build_model
    # 这一项尽量选择False          True好像有问题
    # switch_deploy_flag(False)


    # parser.add_argument('-t', '--blocktype', metavar='BLK', default='base')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet-18')
    # switch_conv_bn_impl(args.blocktype)
    # model = build_model(args.arch)

    if model == 'OREPA_den':
        switch_deploy_flag(False)
        # switch_deploy_flag(True)
        # DBB       # base      # OREPA     # RepVGG        # OREPA_VGG
        switch_conv_bn_impl(block_type = 'OREPA')
        # ResNet-18      # RepVGG-A0        # ResNeXt-50
        model = build_model(arch = 'DenseNet121')
        print("输出的是orep_densenet")
        return model

    if model == 'DBB_den_A':
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'DBB')
        model = build_model(arch = 'DenseNet121_A')
        print("输出的是orep_dense_A")
        return model

    if model == 'OREPA_den_A':
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'DenseNet121_A')
        print("输出的是orep_dense_A")
        return model

    elif model == 'Rep_den':
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'RepVGG')
        model = build_model(arch = 'DenseNet121')
        return model

    elif model == 'OREPA_Res18':
        switch_deploy_flag(True)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'ResNet-18')
        return model



    else:
        print("没有输入正确的模型，因此输出原始的ResNet-18")
        switch_conv_bn_impl(block_type = 'base')
        model = build_model(arch = 'ResNet-18')
        return model

