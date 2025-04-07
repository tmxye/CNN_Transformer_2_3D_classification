def repchoice(model):
    from core5.convnet_utils import switch_deploy_flag, switch_conv_bn_impl, build_model
    # 这一项尽量选择False          True好像有问题
    # switch_deploy_flag(False)


    # parser.add_argument('-t', '--blocktype', metavar='BLK', default='base')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet-18')
    # switch_conv_bn_impl(args.blocktype)
    # model = build_model(args.arch)

    if model == 'OREPA_denNAT':         # Params size (MB): 59.19
        switch_deploy_flag(False)
        # switch_deploy_flag(True)
        # DBB       # base      # OREPA     # RepVGG        # OREPA_VGG
        switch_conv_bn_impl(block_type = 'OREPA')
        # ResNet-18      # RepVGG-A0        # ResNeXt-50
        model = build_model(arch = 'DenseNAT121')
        print("输出的是orep_denNAT")
        return model

    elif model == 'OREPA_den2res':      #  Params size (MB): 21.69
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'DenseNAT_2res')
        return model

    elif model == 'OREPA_den2res1':      #  Params size (MB): 21.69
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'DenseNAT_2res1')
        return model

    elif model == 'OREPA_den2den':      #  Params size (MB): 2.27
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'DenseNAT_2resden')
        return model

    elif model == 'OREPA_den2OCA':      #  Params size (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'DenseNAT_2resOCA')
        return model

    elif model == 'OREPA_den2SOCA':      # (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'DenseNAT_2SOCA')
        return model

    elif model == 'OREPA_denre2OCA':      #  Params size (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'DenseNAT_res2OCA')
        return model

    elif model == 'OREPA_2denOCA':      #
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'DenseNAT_2OCA')
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



    elif model == 'LOREPA_den2OCA':      #  Params size (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'LDenseNAT_2resOCA')
        return model

    elif model == 'LOREPA_den2SOCA':      # (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'LDenseNAT_2SOCA')
        return model

    elif model == 'LOREPA_denre2OCA':      #  Params size (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'LDenseNAT_res2OCA')
        return model

    elif model == 'LOREPA_2denOCA':      #
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'LDenseNAT_2OCA')
        return model



    elif model == 'FOREPA_den2OCA':      #  Params size (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'FDenseNAT_2resOCA')
        return model

    elif model == 'FOREPA_den2SOCA':      # (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'FDenseNAT_2SOCA')
        return model

    elif model == 'FOREPA_denre2OCA':      #  Params size (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'FDenseNAT_res2OCA')
        return model

    elif model == 'FOREPA_2denOCA':      #
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'FDenseNAT_2OCA')
        return model




    elif model == 'FOREPA_den2SOCA_MLP':      # (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'FDenseNAT_2SOCA_MLP')
        return model
    elif model == 'FOREPA_den2SOCA_MLPL':      # (MB): 2.12
        switch_deploy_flag(False)
        switch_conv_bn_impl(block_type = 'OREPA')
        model = build_model(arch = 'FDenseNAT_2SOCA_MLPL')
        return model


    else:
        print("没有输入正确的模型，因此输出原始的ResNet-18")
        switch_conv_bn_impl(block_type = 'base')
        model = build_model(arch = 'ResNet-18')
        return model

