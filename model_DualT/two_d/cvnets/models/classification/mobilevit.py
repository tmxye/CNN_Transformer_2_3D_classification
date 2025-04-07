#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn
import argparse
from typing import Dict, Tuple, Optional

from model_DualT.two_d.cvnets.utils import logger

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.mobilevit import get_configuration
from ...layers import ConvLayer, LinearLayer, GlobalPool, Dropout, SeparableConv
from ...modules import InvertedResidual, MobileViTBlock


from model_DualT.two_d.cvnets.data.sampler import arguments_sampler
from model_DualT.two_d.cvnets.data.collate_fns import arguments_collate_fn
from model_DualT.two_d.cvnets.options.utils import load_config_file
from model_DualT.two_d.cvnets.data.datasets import arguments_dataset
# from model_DualT.two_d.cvnets import arguments_model_YXY, arguments_nn_layers, arguments_ema
from model_DualT.two_d.cvnets import arguments_nn_layers
from model_DualT.two_d.cvnets.misc.averaging_utils import arguments_ema
from model_DualT.two_d.cvnets.anchor_generator import arguments_anchor_gen
# from model_DualT.two_d.cvnets.loss_fn import arguments_loss_fn
# from model_DualT.two_d.cvnets.optim import arguments_optimizer
from model_DualT.two_d.cvnets.optim.scheduler import arguments_scheduler
from model_DualT.two_d.cvnets.common import SUPPORTED_MODALITIES
from model_DualT.two_d.cvnets.data.transforms import arguments_augmentation
from model_DualT.two_d.cvnets.metrics import arguments_stats
from model_DualT.two_d.cvnets.data.video_reader import arguments_video_reader
from model_DualT.two_d.cvnets.matcher_det import arguments_box_matcher
from model_DualT.two_d.cvnets.utils import logger


LOSS_REGISTRY = {}

# from .base_criteria import BaseCriteria
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#



def general_optim_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("optimizer", "Optimizer related arguments")
    group.add_argument("--optim.name", default="sgd", help="Which optimizer")
    group.add_argument("--optim.eps", type=float, default=1e-8, help="Optimizer eps")
    group.add_argument(
        "--optim.weight-decay", default=4e-5, type=float, help="Weight decay"
    )
    group.add_argument(
        "--optim.no-decay-bn-filter-bias",
        action="store_true",
        help="No weight decay in normalization layers and bias",
    )
    return parser

OPTIM_REGISTRY = {}
def arguments_optimizer(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = general_optim_args(parser=parser)

    # add optim specific arguments
    for k, v in OPTIM_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser



import torch
from torch import nn, Tensor
import argparse
from typing import Any


class BaseCriteria(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseCriteria, self).__init__()
        self.eps = 1e-7

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(
        self, input_sample: Any, prediction: Any, target: Any, *args, **kwargs
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _class_weights(target: Tensor, n_classes: int, norm_val: float = 1.1) -> Tensor:
        class_hist: Tensor = torch.histc(
            target.float(), bins=n_classes, min=0, max=n_classes - 1
        )
        mask_indices = class_hist == 0

        # normalize between 0 and 1 by dividing by the sum
        norm_hist = torch.div(class_hist, class_hist.sum())
        norm_hist = torch.add(norm_hist, norm_val)

        # compute class weights..
        # samples with more frequency will have less weight and vice-versa
        class_wts = torch.div(torch.ones_like(class_hist), torch.log(norm_hist))

        # mask the classes which do not have samples in the current batch
        class_wts[mask_indices] = 0.0

        return class_wts.to(device=target.device)

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)



def register_loss_fn(name):
    def register_loss_fn_class(cls):
        if name in LOSS_REGISTRY:
            raise ValueError(
                "Cannot register duplicate loss function ({})".format(name)
            )

        if not issubclass(cls, BaseCriteria):
            raise ValueError(
                "Criteria ({}: {}) must extend BaseCriteria".format(name, cls.__name__)
            )

        LOSS_REGISTRY[name] = cls
        return cls

    return register_loss_fn_class



def general_loss_fn_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Loss function arguments", description="Loss function arguments"
    )

    group.add_argument(
        "--loss.category",
        type=str,
        default="classification",
        help="Loss function category (classification,segmentation)",
    )
    group.add_argument(
        "--loss.ignore-idx", type=int, default=-1, help="Ignore idx in loss function"
    )

    return parser

def arguments_loss_fn(parser: argparse.ArgumentParser):
    parser = general_loss_fn_args(parser=parser)

    # add loss function specific arguments
    for k, v in LOSS_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


from model_DualT.two_d.cvnets.models.segmentation import arguments_segmentation, build_segmentation_model
from model_DualT.two_d.cvnets.models.classification import arguments_classification, build_classification_model
from model_DualT.two_d.cvnets.models.detection import arguments_detection, build_detection_model
from model_DualT.two_d.cvnets.models.video_classification import (
    build_video_classification_model,
    arguments_video_classification,
)
def arguments_model_YXY(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # classification network
    parser = arguments_classification(parser=parser)

    # detection network
    parser = arguments_detection(parser=parser)

    # segmentation network
    parser = arguments_segmentation(parser=parser)

    # video classification network
    parser = arguments_video_classification(parser=parser)

    return parser

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace_dict = vars(namespace)

        if len(values) > 0:
            override_dict = {}
            # values are list of key-value pairs
            for value in values:
                key = None
                try:
                    key, value = value.split("=")
                except ValueError as e:
                    logger.error(
                        "For override arguments, a key-value pair of the form key=value is expected"
                    )

                if key in namespace_dict:
                    value_namespace = namespace_dict[key]
                    if value_namespace is None and value is None:
                        value = None
                    elif value_namespace is None and value is not None:
                        # possibly a string or list of strings or list of integers

                        # check if string is a list or not
                        value = value.split(",")
                        if len(value) == 1:
                            # its a string
                            value = str(value[0])

                            # check if its empty string or not
                            if value == "" or value.lower() == "none":
                                value = None
                        else:
                            # its a list of integers or strings
                            try:
                                # convert to int
                                value = [int(v) for v in value]
                            except:
                                # pass because its a string
                                pass
                    else:
                        try:
                            if value.lower() == "true":  # check for boolean
                                value = True
                            elif value.lower() == "false":
                                value = False
                            else:
                                desired_type = type(value_namespace)
                                value = desired_type(value)
                        except ValueError as e:
                            logger.warning(
                                "Type mismatch while over-riding. Skipping key: {}".format(
                                    key
                                )
                            )
                            continue

                    override_dict[key] = value
            setattr(namespace, "override_args", override_dict)
        else:
            setattr(namespace, "override_args", None)



def arguments_ddp(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="DDP arguments", description="DDP arguments"
    )
    group.add_argument("--ddp.disable", action="store_true", help="Don't use DDP")
    group.add_argument(
        "--ddp.rank", type=int, default=0, help="Node rank for distributed training"
    )
    group.add_argument(
        "--ddp.world-size", type=int, default=-1, help="World size for DDP"
    )
    group.add_argument("--ddp.dist-url", type=str, default=None, help="DDP URL")
    group.add_argument(
        "--ddp.dist-port",
        type=int,
        default=30786,
        help="DDP Port. Only used when --ddp.dist-url is not specified",
    )
    group.add_argument("--ddp.device-id", type=int, default=None, help="Device ID")
    group.add_argument(
        "--ddp.no-spawn", action="store_true", help="Don't use DDP with spawn"
    )
    group.add_argument(
        "--ddp.backend", type=str, default="nccl", help="DDP backend. Default is nccl"
    )
    group.add_argument(
        "--ddp.find-unused-params",
        action="store_true",
        help="Find unused params in model. useful for debugging with DDP",
    )

    return parser



def arguments_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="Common arguments", description="Common arguments"
    )

    group.add_argument("--common.seed", type=int, default=0, help="Random seed")
    group.add_argument(
        "--common.config-file", type=str, default=None, help="Configuration file"
    )
    group.add_argument(
        "--common.results-loc",
        type=str,
        default="results",
        help="Directory where results will be stored",
    )
    group.add_argument(
        "--common.run-label",
        type=str,
        default="run_1",
        help="Label id for the current run",
    )

    group.add_argument(
        "--common.resume", type=str, default=None, help="Resume location"
    )
    group.add_argument(
        "--common.finetune_imagenet1k",
        type=str,
        default=None,
        help="Checkpoint location to be used for finetuning",
    )
    group.add_argument(
        "--common.finetune_imagenet1k-ema",
        type=str,
        default=None,
        help="EMA Checkpoint location to be used for finetuning",
    )

    group.add_argument(
        "--common.mixed-precision", action="store_true", help="Mixed precision training"
    )
    group.add_argument(
        "--common.accum-freq",
        type=int,
        default=1,
        help="Accumulate gradients for this number of iterations",
    )
    group.add_argument(
        "--common.accum-after-epoch",
        type=int,
        default=0,
        help="Start accumulation after this many epochs",
    )
    group.add_argument(
        "--common.log-freq",
        type=int,
        default=100,
        help="Display after these many iterations",
    )
    group.add_argument(
        "--common.auto-resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    group.add_argument(
        "--common.grad-clip", type=float, default=None, help="Gradient clipping value"
    )
    group.add_argument(
        "--common.k-best-checkpoints",
        type=int,
        default=5,
        help="Keep k-best checkpoints",
    )

    group.add_argument(
        "--common.inference-modality",
        type=str,
        default="image",
        choices=SUPPORTED_MODALITIES,
        help="Inference modality. Image or videos",
    )

    group.add_argument(
        "--common.channels-last",
        action="store_true",
        default=False,
        help="Use channel last format during training. "
        "Note 1: that some models may not support it, so we recommend to use it with caution"
        "Note 2: Channel last format does not work with 1-, 2-, and 3- tensors. "
        "Therefore, we support it via custom collate functions",
    )

    group.add_argument(
        "--common.tensorboard-logging",
        action="store_true",
        help="Enable tensorboard logging",
    )
    group.add_argument(
        "--common.bolt-logging", action="store_true", help="Enable bolt logging"
    )

    group.add_argument(
        "--common.override-kwargs",
        nargs="*",
        action=ParseKwargs,
        help="Override arguments. Example. To override the value of --sampler.vbs.crop-size-width, "
        "we can pass override argument as "
        "--common.override-kwargs sampler.vbs.crop_size_width=512 \n "
        "Note that keys in override arguments do not contain -- or -",
    )

    group.add_argument(
        "--common.enable-coreml-compatible-module",
        action="store_true",
        help="Use coreml compatible modules (if applicable) during inference",
    )

    group.add_argument(
        "--common.debug-mode",
        action="store_true",
        help="You can use this flag for debugging purposes.",
    )

    return parser


def get_training_arguments(parse_args: Optional[bool] = True):
    parser = argparse.ArgumentParser(description="Training arguments", add_help=True)

    # sampler related arguments
    parser = arguments_sampler(parser=parser)

    # dataset related arguments
    parser = arguments_dataset(parser=parser)

    # anchor generator arguments
    parser = arguments_anchor_gen(parser=parser)

    # arguments related to box matcher
    parser = arguments_box_matcher(parser=parser)

    # Video reader related arguments
    parser = arguments_video_reader(parser=parser)

    # collate fn  related arguments
    parser = arguments_collate_fn(parser=parser)

    # transform related arguments
    parser = arguments_augmentation(parser=parser)

    # model related arguments
    parser = arguments_nn_layers(parser=parser)
    parser = arguments_model_YXY(parser=parser)
    parser = arguments_ema(parser=parser)

    # loss function arguments
    parser = arguments_loss_fn(parser=parser)

    # optimizer arguments
    parser = arguments_optimizer(parser=parser)
    parser = arguments_scheduler(parser=parser)

    # DDP arguments
    parser = arguments_ddp(parser=parser)

    # stats arguments
    parser = arguments_stats(parser=parser)

    # common
    parser = arguments_common(parser=parser)

    if parse_args:
        # parse args
        opts = parser.parse_args()
        opts = load_config_file(opts)
        return opts
    else:
        return parser


@register_cls_models("mobilevit")
class MobileViT(BaseEncoder):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        opts = get_training_arguments()
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(
            opts, "model.classification.classifier_dropout", 0.0
        )

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration(opts=opts)

        super().__init__(*args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=exp_channels,
            kernel_size=1,
            stride=1,
            use_act=True,
            use_norm=True,
        )

        self.model_conf_dict["exp_before_cls"] = {
            "in": in_channels,
            "out": exp_channels,
        }

        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(
                name="dropout", module=Dropout(p=classifier_dropout, inplace=True)
            )
        self.classifier.add_module(
            name="fc",
            module=LinearLayer(
                in_features=exp_channels, out_features=num_classes, bias=True
            ),
        )

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.classification.mit.mode",
            type=str,
            default="small",
            choices=["xx_small", "x_small", "small"],
            help="MobileViT mode. Defaults to small",
        )
        group.add_argument(
            "--model.classification.mit.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mit.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN layers. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mit.dropout",
            type=float,
            default=0.0,
            help="Dropout in Transformer layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mit.transformer-norm-layer",
            type=str,
            default="layer_norm",
            help="Normalization layer in transformer. Defaults to LayerNorm",
        )
        group.add_argument(
            "--model.classification.mit.no-fuse-local-global-features",
            action="store_true",
            help="Do not combine local and global features in MobileViT block",
        )
        group.add_argument(
            "--model.classification.mit.conv-kernel-size",
            type=int,
            default=3,
            help="Kernel size of Conv layers in MobileViT block",
        )

        group.add_argument(
            "--model.classification.mit.head-dim",
            type=int,
            default=None,
            help="Head dimension in transformer",
        )
        group.add_argument(
            "--model.classification.mit.number-heads",
            type=int,
            default=None,
            help="Number of heads in transformer",
        )
        return parser

    def _make_layer(
        self,
        opts,
        input_channel,
        cfg: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
        opts, input_channel: int, cfg: Dict, *args, **kwargs
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
        self,
        opts,
        input_channel,
        cfg: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error(
                "Transformer input dimension should be divisible by head dimension. "
                "Got {} and {}.".format(transformer_dim, head_dim)
            )

        block.append(
            MobileViTBlock(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(
                    opts, "model.classification.mit.attn_dropout", 0.1
                ),
                head_dim=head_dim,
                no_fusion=getattr(
                    opts,
                    "model.classification.mit.no_fuse_local_global_features",
                    False,
                ),
                conv_ksize=getattr(
                    opts, "model.classification.mit.conv_kernel_size", 3
                ),
            )
        )

        return nn.Sequential(*block), input_channel
