#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import os
from typing import Optional

from model_DualT.two_d.cvnets.utils import logger


def load_pretrained_model(
    model: torch.nn.Module,
    wt_loc: str,
    is_master_node: Optional[bool] = False,
    *args,
    **kwargs
) -> torch.nn.Module:
    """
    Helper function to load pre-trained weights
    """
    if not os.path.isfile(wt_loc):
        logger.error("Pretrained file is not found here: {}".format(wt_loc))

    wts = torch.load(wt_loc, map_location="cpu")
    try:
        if hasattr(model, "module"):
            model.module.load_state_dict(wts)
        else:
            model.load_state_dict(wts)

        if is_master_node:
            logger.log("Pretrained weights are loaded from {}".format(wt_loc))
    except Exception as e:
        if is_master_node:
            logger.error(
                "Unable to load pretrained weights from {}. Error: {}".format(wt_loc, e)
            )

    return model


def parameter_list(
    named_parameters,
    weight_decay: Optional[float] = 0.0,
    no_decay_bn_filter_bias: Optional[bool] = False,
    *args,
    **kwargs
):
    with_decay = []
    without_decay = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (
                    param.requires_grad
                    and len(param.shape) == 1
                    and no_decay_bn_filter_bias
                ):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                elif param.requires_grad:
                    with_decay.append(param)
    else:
        for p_name, param in named_parameters():
            if (
                param.requires_grad
                and len(param.shape) == 1
                and no_decay_bn_filter_bias
            ):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
            elif param.requires_grad:
                with_decay.append(param)
    param_list = [{"params": with_decay, "weight_decay": weight_decay}]
    if len(without_decay) > 0:
        param_list.append({"params": without_decay, "weight_decay": 0.0})
    return param_list
