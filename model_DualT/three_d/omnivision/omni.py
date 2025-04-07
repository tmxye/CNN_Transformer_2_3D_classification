
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from .models.swin_transformer import SwinTransformer3D



def get_all_heads(dim_in: int = 1024) -> nn.Module:
    heads = nn.ModuleDict(
        {
            "image": get_imagenet_head(dim_in),
            "rgbd": get_sunrgbd_head(dim_in),
            "video": get_kinetics_head(dim_in),
        }
    )
    return heads


def get_imagenet_head(dim_in: int = 1024) -> nn.Module:
    head = nn.Linear(in_features=dim_in, out_features=1000, bias=True)
    return head


def get_sunrgbd_head(dim_in: int = 1024) -> nn.Module:
    head = nn.Linear(in_features=dim_in, out_features=19, bias=True)
    return head


def get_kinetics_head(dim_in: int = 1024, num_classes: int = 400) -> nn.Module:
    head = nn.Linear(in_features=dim_in, out_features=num_classes, bias=True)
    return nn.Sequential(nn.Dropout(p=0.5), head)


class OmnivoreModel(nn.Module):
    def __init__(self, trunk: nn.Module, heads: Union[nn.ModuleDict, nn.Module]):
        super().__init__()
        self.trunk = trunk
        self.heads = heads
        self.types = ["image", "video", "rgbd"]
        self.multimodal_model = False
        if isinstance(heads, nn.ModuleDict):
            self.multimodal_model = True
            # self.multimodal_model = False       # by YXY
            assert all([n in heads for n in self.types]), "All heads must be provided"

    # def forward(self, x: torch.Tensor, input_type: Optional[str] = None):
    def forward(self, x: torch.Tensor, input_type: Optional[str] = "video"):
        """
        Args:
            x: input to the model of shape 1 x C x T x H x W
            input_type: Optional[str] one of ["image", "video", "rgbd"]
                if self.multimodal_model is True
        Returns:
            preds: tensor of shape (1, num_classes)
        """
        assert x.ndim == 5
        features = self.trunk(x)
        head = self.heads
        if self.multimodal_model:
            assert input_type in self.types, "unsupported input type"
            head = head[input_type]
        return head(features)


CHECKPOINT_PATHS = {
    "omnivore_swinT": "https://dl.fbaipublicfiles.com/omnivore/models/swinT_checkpoint.torch",
    "omnivore_swinS": "https://dl.fbaipublicfiles.com/omnivore/models/swinS_checkpoint.torch",
    "omnivore_swinB": "https://dl.fbaipublicfiles.com/omnivore/models/swinB_checkpoint.torch",
    "omnivore_swinB_in21k": "https://dl.fbaipublicfiles.com/omnivore/models/swinB_In21k_checkpoint.torch",
    "omnivore_swinL_in21k": "https://dl.fbaipublicfiles.com/omnivore/models/swinL_In21k_checkpoint.torch",
    "omnivore_swinB_epic": "https://dl.fbaipublicfiles.com/omnivore/models/swinB_epic_checkpoint.torch",
}


def _omnivore_base(
    trunk: nn.Module,
    heads: Optional[Union[nn.Module, nn.ModuleDict]] = None,
    head_dim_in: int = 1024,
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    checkpoint_name: str = "omnivore_swinB",
) -> nn.Module:
    """
    Load and initialize the specified Omnivore
    model trunk (and optionally heads).

    Args:
        trunk: nn.Module of the SwinTransformer3D trunk
        heads: Provide the heads module if using a custom
            model. If not provided image/video/rgbd heads are
            added corresponding to the omnivore base model.
        head_dim_in: Only needs to be set if heads = None.
            The dim is used for the default base model heads.
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.

    Returns:
        model: nn.Module of the full Omnivore model
    """
    if load_heads and heads is None:
        # Get heads
        heads = get_all_heads(dim_in=head_dim_in)

    if pretrained:
        path = CHECKPOINT_PATHS[checkpoint_name]

        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            path, progress=progress, map_location="cpu"
        )
        trunk.load_state_dict(checkpoint["trunk"])

        if load_heads:
            heads.load_state_dict(checkpoint["heads"])

    if load_heads:
        model = OmnivoreModel(trunk=trunk, heads=heads)
    else:
        model = trunk

    return model




def omnivore_swinT(
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    Omnivore model trunk: Swin T patch (2,4,4) window (8,7,7)

    Args:
        pretrained: if True loads weights from model trained on
            Imagenet 1k, Kinetics 400, SUN RGBD.
        progress: print progress of loading checkpoint
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.

    Returns:
        model: nn.Module of the omnivore model
    """

    # Only specify the non default values
    trunk = SwinTransformer3D(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        drop_path_rate=0.2,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
        **kwargs,
    )

    return _omnivore_base(
        trunk=trunk,
        head_dim_in=768,  # 96*8
        progress=progress,
        pretrained=pretrained,
        load_heads=load_heads,
        checkpoint_name="omnivore_swinT",
    )

from ..resnet import resnet18

def omnivore_swinT(
    pretrained: bool = False,
    progress: bool = True,
    load_heads: bool = True,
    **kwargs: Any,
) -> nn.Module:

    # Only specify the non default values
    # trunk = resnet18(sample_size=224, sample_duration=16,  num_classes=2)
    trunk = resnet18(sample_size=224, sample_duration=16,  num_classes=768)

    return _omnivore_base(
        trunk=trunk,
        head_dim_in=768,  # 96*8
        progress=progress,
        pretrained=pretrained,
        load_heads=load_heads,
        checkpoint_name="omnivore_swinT",
    )












