"""Mobile-Former V1

A PyTorch impl of MobileFromer-V1.
 
Paper: Mobile-Former: Bridging MobileNet and Transformer (CVPR 2022)
       https://arxiv.org/abs/2108.05895

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from timm.models import DropPath

from timm.models.hub import has_hf_hub, download_cached_file, load_state_dict_from_hf, load_state_dict_from_hf, \
    load_state_dict_from_hf
from torch.hub import load_state_dict_from_url

# from models_32.model_YXY_fullgood_13.models import Linear
from core5.data.layers import Linear, DropPath
from models_32.model_YXY_fullgood_13.models.features import FeatureListNet, FeatureDictNet, FeatureDictNet, \
    FeatureHookNet
from models_32.model_YXY_fullgood_13.models.layers import Conv2dSame
# from models_32.oldtwo.Conformer.mmdetection.mmdet.models.backbones.Conformer import DropPath
# from .helpers import build_model_with_cfg, default_cfg_for_features
# from .registry import register_model
# from .dna_blocks import DnaBlock, DnaBlock3, _make_divisible, MergeClassifier, Local2Global
# Linear = nn.Linear()
# DropPath = nn.Dropout()

""" Model creation / weight loading / state_dict helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple


_logger = logging.getLogger(__name__)


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_custom_pretrained(model, default_cfg=None, load_fn=None, progress=False, check_hash=False):
    r"""Loads a custom (read non .pth) weight file

    Downloads checkpoint file into cache-dir like torch.hub based loaders, but calls
    a passed in custom load fun, or the `load_pretrained` model member fn.

    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        model: The instantiated model to load weights into
        default_cfg (dict): Default pretrained model cfg
        load_fn: An external stand alone fn that loads weights into provided model, otherwise a fn named
            'laod_pretrained' on the model will be called if it exists
        progress (bool, optional): whether or not to display a progress bar to stderr. Default: False
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file. Default: False
    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)
    if not pretrained_url:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    cached_file = download_cached_file(default_cfg['url'], check_hash=check_hash, progress=progress)

    if load_fn is not None:
        load_fn(model, cached_file)
    elif hasattr(model, 'load_pretrained'):
        model.load_pretrained(cached_file)
    else:
        _logger.warning("Valid function to load pretrained weights is not available, using random initialization.")


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def load_pretrained(model, default_cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, progress=False):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        default_cfg (Optional[Dict]): default configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
        progress (bool): enable progress bar for weight download

    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)
    hf_hub_id = default_cfg.get('hf_hub', None)
    if not pretrained_url and not hf_hub_id:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if hf_hub_id and has_hf_hub(necessary=not pretrained_url):
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({hf_hub_id})')
        state_dict = load_state_dict_from_hf(hf_hub_id)
    else:
        _logger.info(f'Loading pretrained weights from url ({pretrained_url})')
        state_dict = load_state_dict_from_url(pretrained_url, progress=progress, map_location='cpu')
    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = default_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = default_cfg.get('classifier', None)
    label_offset = default_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != default_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                del state_dict[classifier_name + '.weight']
                del state_dict[classifier_name + '.bias']
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_string(parent_module, model_string):
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            # FIXME extra checks to ensure this is actually the FC classifier layer and not a diff Linear layer?
            num_features = state_dict[n + '.weight'][1]
            new_fc = Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module


def adapt_model_from_file(parent_module, model_variant):
    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
    with open(adapt_file, 'r') as f:
        return adapt_model_from_string(parent_module, f.read().strip())


def default_cfg_for_features(default_cfg):
    default_cfg = deepcopy(default_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'crop_pct', 'classifier', 'global_pool')  # add default final pool size?
    for tr in to_remove:
        default_cfg.pop(tr, None)
    return default_cfg


def overlay_external_default_cfg(default_cfg, kwargs):
    """ Overlay 'external_default_cfg' in kwargs on top of default_cfg arg.
    """
    external_default_cfg = kwargs.pop('external_default_cfg', None)
    if external_default_cfg:
        default_cfg.pop('url', None)  # url should come from external cfg
        default_cfg.pop('hf_hub', None)  # hf hub id should come from external cfg
        default_cfg.update(external_default_cfg)


def set_default_kwargs(kwargs, names, default_cfg):
    for n in names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # default_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = default_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, default_cfg[n])


def filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    FIXME this sequence of overlay default_cfg, set default kwargs, filter kwargs
    could/should be replaced by an improved configuration mechanism

    Args:
        default_cfg: input default_cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Overlay default cfg values from `external_default_cfg` if it exists in kwargs
    overlay_external_default_cfg(default_cfg, kwargs)
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if default_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)
    set_default_kwargs(kwargs, names=default_kwarg_names, default_cfg=default_cfg)
    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    filter_kwargs(kwargs, names=kwargs_filter)


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        pretrained_custom_load: bool = False,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        default_cfg (dict): model's default pretrained/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
        pretrained_custom_load (bool): use custom load fn, to load numpy or other non PyTorch weights
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}
    default_cfg = deepcopy(default_cfg) if default_cfg else {}
    update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter)
    default_cfg.setdefault('architecture', variant)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    # Build the model
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = default_cfg

    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model)
        else:
            load_pretrained(
                model,
                num_classes=num_classes_pretrained,
                in_chans=kwargs.get('in_chans', 3),
                filter_fn=pretrained_filter_fn,
                strict=pretrained_strict)

    # Wrap the model in a feature extraction module if enabled
    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.default_cfg = default_cfg_for_features(default_cfg)  # add back default_cfg

    return model


def model_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params based on ordering
        return [p for p in model.parameters()][:-2]
    else:
        return model.parameters()


__all__ = ['MobileFormer']

""" Dna blocks used for Mobile-Former

A PyTorch impl of Dna blocks

Paper: Mobile-Former: Bridging MobileNet and Transformer (CVPR 2022)
       https://arxiv.org/abs/2108.05895 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .layers import DropPath


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        out = x.view(b, -1, h, w)
        return out


class DyReLU(nn.Module):
    def __init__(self, num_func=2, use_bias=False, scale=2., serelu=False):
        """
        num_func: -1: none
                   0: relu
                   1: SE
                   2: dy-relu
        """
        super(DyReLU, self).__init__()

        assert (num_func >= -1 and num_func <= 2)
        self.num_func = num_func
        self.scale = scale

        serelu = serelu and num_func == 1
        self.act = nn.ReLU6(inplace=True) if num_func == 0 or serelu else nn.Sequential()

    def forward(self, x):
        if isinstance(x, tuple):
            out, a = x
        else:
            out = x

        out = self.act(out)

        if self.num_func == 1:  # SE
            a = a * self.scale
            out = out * a
        elif self.num_func == 2:  # DY-ReLU
            _, C, _, _ = a.shape
            a1, a2 = torch.split(a, [C // 2, C // 2], dim=1)
            a1 = (a1 - 0.5) * self.scale + 1.0  # 0.0 -- 2.0
            a2 = (a2 - 0.5) * self.scale  # -1.0 -- 1.0
            out = torch.max(out * a1, out * a2)

        return out


class HyperFunc(nn.Module):
    def __init__(self, token_dim, oup, sel_token_id=0, reduction_ratio=4):
        super(HyperFunc, self).__init__()

        self.sel_token_id = sel_token_id
        squeeze_dim = token_dim // reduction_ratio
        self.hyper = nn.Sequential(
            nn.Linear(token_dim, squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, oup),
            h_sigmoid()
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x, attn = x

        if self.sel_token_id == -1:
            hp = self.hyper(x).permute(1, 2, 0)  # bs x hyper_dim x T

            bs, T, H, W = attn.shape
            attn = attn.view(bs, T, H * W)
            hp = torch.matmul(hp, attn)  # bs x hyper_dim x HW
            h = hp.view(bs, -1, H, W)
        else:
            t = x[self.sel_token_id]
            h = self.hyper(t)
            h = torch.unsqueeze(torch.unsqueeze(h, 2), 3)
        return h


class MaxDepthConv(nn.Module):
    def __init__(self, inp, oup, stride):
        super(MaxDepthConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, oup, (3, 1), stride, (1, 0), bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inp, oup, (1, 3), stride, (0, 1), bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        out = torch.max(y1, y2)
        return out


class Local2GlobalAttn(nn.Module):
    def __init__(
            self,
            inp,
            token_dim=128,
            token_num=6,
            inp_res=0,
            norm_pos='post',
            drop_path_rate=0.
    ):
        super(Local2GlobalAttn, self).__init__()

        num_heads = 2
        self.scale = (inp // num_heads) ** -0.5

        self.q = nn.Linear(token_dim, inp)
        self.proj = nn.Linear(inp, token_dim)

        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        features, tokens = x
        bs, C, _, _ = features.shape

        t = self.q(tokens).permute(1, 0, 2)  # from T x bs x Ct to bs x T x Ct
        k = features.view(bs, C, -1)  # bs x C x HW
        attn = (t @ k) * self.scale

        attn_out = attn.softmax(dim=-1)  # bs x T x HW
        attn_out = (attn_out @ k.permute(0, 2, 1))  # bs x T x C
        # note here: k=v without transform
        t = self.proj(attn_out.permute(1, 0, 2))  # T x bs x C

        tokens = tokens + self.drop_path(t)
        tokens = self.layer_norm(tokens)

        return tokens


class Local2Global(nn.Module):
    def __init__(
            self,
            inp,
            block_type='mlp',
            token_dim=128,
            token_num=6,
            inp_res=0,
            attn_num_heads=2,
            use_dynamic=False,
            norm_pos='post',
            drop_path_rate=0.,
            remove_proj_local=True,
    ):
        super(Local2Global, self).__init__()
        print(f'L2G: {attn_num_heads} heads, inp: {inp}, token: {token_dim}')

        self.num_heads = attn_num_heads
        self.token_num = token_num
        self.norm_pos = norm_pos
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, inp),
                h_sigmoid(),
            )

        if 'mlp' in block_type:
            self.mlp = nn.Linear(inp_res, token_num)

        if 'attn' in block_type:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.q = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(inp, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)

        self.remove_proj_local = remove_proj_local
        if self.remove_proj_local == False:
            self.k = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.v = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)

    def forward(self, x):
        features, tokens = x  # features: bs x C x H x W
        #   tokens: T x bs x Ct

        bs, C, H, W = features.shape
        T, _, _ = tokens.shape
        attn = None

        if 'mlp' in self.block:
            t_sum = self.mlp(features.view(bs, C, -1)).permute(2, 0, 1)  # T x bs x C

        if 'attn' in self.block:
            t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0,
                                                                       3)  # from T x bs x Ct to bs x N x T x Ct/N
            if self.remove_proj_local:
                k = features.view(bs, self.num_heads, -1, H * W)  # bs x N x C/N x HW
                attn = (t @ k) * self.scale  # bs x N x T x HW

                attn_out = attn.softmax(dim=-1)  # bs x N x T x HW
                attn_out = (attn_out @ k.transpose(-1, -2))  # bs x N x T x C/N (k: bs x N x C/N x HW)
                # note here: k=v without transform
            else:
                k = self.k(features).view(bs, self.num_heads, -1, H * W)  # bs x N x C/N x HW
                v = self.v(features).view(bs, self.num_heads, -1, H * W)  # bs x N x C/N x HW
                attn = (t @ k) * self.scale  # bs x N x T x HW

                attn_out = attn.softmax(dim=-1)  # bs x N x T x HW
                attn_out = (attn_out @ v.transpose(-1, -2))  # bs x N x T x C/N (k: bs x N x C/N x HW)
                # note here: k=v without transform

            t_a = attn_out.permute(2, 0, 1, 3)  # T x bs x N x C/N
            t_a = t_a.reshape(T, bs, -1)

            if 'mlp' in self.block:
                t_sum = t_sum + t_a
            else:
                t_sum = t_a

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            t_sum = t_sum * alp

        t_sum = self.proj(t_sum)
        tokens = tokens + self.drop_path(t_sum)
        tokens = self.layer_norm(tokens)

        if attn is not None:
            bs, Nh, Ca, HW = attn.shape
            attn = attn.view(bs, Nh, Ca, H, W)

        return tokens, attn


class GlobalBlock(nn.Module):
    def __init__(
            self,
            block_type='mlp',
            token_dim=128,
            token_num=6,
            mlp_token_exp=4,
            attn_num_heads=4,
            use_dynamic=False,
            use_ffn=False,
            norm_pos='post',
            drop_path_rate=0.
    ):
        super(GlobalBlock, self).__init__()

        print(f'G2G: {attn_num_heads} heads')

        self.block = block_type
        self.num_heads = attn_num_heads
        self.token_num = token_num
        self.norm_pos = norm_pos
        self.use_dynamic = use_dynamic
        self.use_ffn = use_ffn
        self.ffn_exp = 2

        if self.use_ffn:
            print('use ffn')
            self.ffn = nn.Sequential(
                nn.Linear(token_dim, token_dim * self.ffn_exp),
                nn.GELU(),
                nn.Linear(token_dim * self.ffn_exp, token_dim)
            )
            self.ffn_norm = nn.LayerNorm(token_dim)

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, token_dim),
                h_sigmoid(),
            )

        if 'mlp' in self.block:
            self.token_mlp = nn.Sequential(
                nn.Linear(token_num, token_num * mlp_token_exp),
                nn.GELU(),
                nn.Linear(token_num * mlp_token_exp, token_num),
            )

        if 'attn' in self.block:
            self.scale = (token_dim // attn_num_heads) ** -0.5
            self.q = nn.Linear(token_dim, token_dim)

        self.channel_mlp = nn.Linear(token_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        tokens = x

        T, bs, C = tokens.shape

        if 'mlp' in self.block:
            # use post norm, token.shape: token_num x bs x channel
            t = self.token_mlp(tokens.permute(1, 2, 0))  # bs x channel x token_num
            t_sum = t.permute(2, 0, 1)  # token_num x bs x channel

        if 'attn' in self.block:
            t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0,
                                                                       3)  # from T x bs x Ct to bs x N x T x Ct/N
            k = tokens.permute(1, 2, 0).view(bs, self.num_heads, -1,
                                             T)  # from T x bs x Ct -> bs x Ct x T -> bs x N x Ct/N x T
            attn = (t @ k) * self.scale  # bs x N x T x T

            attn_out = attn.softmax(dim=-1)  # bs x N x T x T
            attn_out = (attn_out @ k.transpose(-1, -2))  # bs x N x T x C/N (k: bs x N x Ct/N x T)
            # note here: k=v without transform
            t_a = attn_out.permute(2, 0, 1, 3)  # T x bs x N x C/N
            t_a = t_a.reshape(T, bs, -1)

            t_sum = t_sum + t_a if 'mlp' in self.block else t_a

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            t_sum = t_sum * alp

        t_sum = self.channel_mlp(t_sum)  # token_num x bs x channel
        tokens = tokens + self.drop_path(t_sum)
        tokens = self.layer_norm(tokens)

        if self.use_ffn:
            t_ffn = self.ffn(tokens)
            tokens = tokens + t_ffn
            tokens = self.ffn_norm(tokens)

        return tokens


class Global2Local(nn.Module):
    def __init__(
            self,
            inp,
            inp_res=0,
            block_type='mlp',
            token_dim=128,
            token_num=6,
            attn_num_heads=2,
            use_dynamic=False,
            drop_path_rate=0.,
            remove_proj_local=True,
    ):
        super(Global2Local, self).__init__()
        print(f'G2L: {attn_num_heads} heads, inp: {inp}, token: {token_dim}')

        self.token_num = token_num
        self.num_heads = attn_num_heads
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, inp),
                h_sigmoid(),
            )

        if 'mlp' in self.block:
            self.mlp = nn.Linear(token_num, inp_res)

        if 'attn' in self.block:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.k = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(token_dim, inp)
        self.drop_path = DropPath(drop_path_rate)

        self.remove_proj_local = remove_proj_local
        if self.remove_proj_local == False:
            self.q = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.fuse = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)

    def forward(self, x):
        out, tokens = x

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            v = self.proj(tokens)
            v = (v * alp).permute(1, 2, 0)
        else:
            v = self.proj(tokens).permute(1, 2, 0)  # from T x bs x Ct -> T x bs x C -> bs x C x T

        bs, C, H, W = out.shape
        if 'mlp' in self.block:
            g_sum = self.mlp(v).view(bs, C, H, W)  # bs x C x T -> bs x C x H x W

        if 'attn' in self.block:
            if self.remove_proj_local:
                q = out.view(bs, self.num_heads, -1, H * W).transpose(-1, -2)  # bs x N x HW x C/N
            else:
                q = self.q(out).view(bs, self.num_heads, -1, H * W).transpose(-1, -2)  # bs x N x HW x C/N

            k = self.k(tokens).permute(1, 2, 0).view(bs, self.num_heads, -1,
                                                     self.token_num)  # from T x bs x Ct -> bs x C x T -> bs x N x C/N x T
            attn = (q @ k) * self.scale  # bs x N x HW x T

            attn_out = attn.softmax(dim=-1)  # bs x N x HW x T

            vh = v.view(bs, self.num_heads, -1, self.token_num)  # bs x N x C/N x T
            attn_out = (attn_out @ vh.transpose(-1, -2))  # bs x N x HW x C/N
            # note here k != v
            g_a = attn_out.transpose(-1, -2).reshape(bs, C, H, W)  # bs x C x HW

            if self.remove_proj_local == False:
                g_a = self.fuse(g_a)

            g_sum = g_sum + g_a if 'mlp' in self.block else g_a

        out = out + self.drop_path(g_sum)

        return out


##########################################################################################################
# Dna Blocks
##########################################################################################################
class DnaBlock3(nn.Module):
    def __init__(
            self,
            inp,
            oup,
            stride,
            exp_ratios,  # (e1, e2)
            kernel_size=(3, 3),
            dw_conv='dw',
            group_num=1,
            se_flag=[2, 0, 2, 0],
            hyper_token_id=0,
            hyper_reduction_ratio=4,
            token_dim=128,
            token_num=6,
            inp_res=49,
            gbr_type='mlp',
            gbr_dynamic=[False, False, False],
            gbr_ffn=False,
            gbr_before_skip=False,
            mlp_token_exp=4,
            norm_pos='post',
            drop_path_rate=0.,
            cnn_drop_path_rate=0.,
            attn_num_heads=2,
            remove_proj_local=True,
    ):
        super(DnaBlock3, self).__init__()

        print(f'block: {inp_res}, cnn-drop {cnn_drop_path_rate:.4f}, mlp-drop {drop_path_rate:.4f}')
        if isinstance(exp_ratios, tuple):
            e1, e2 = exp_ratios
        else:
            e1, e2 = exp_ratios, 4
        k1, k2 = kernel_size

        self.stride = stride
        self.hyper_token_id = hyper_token_id

        self.identity = stride == 1 and inp == oup
        self.use_conv_alone = False
        if e1 == 1 or e2 == 0:
            self.use_conv_alone = True
            if dw_conv == 'dw':
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(inp, inp * e1, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp * e1),
                    nn.ReLU6(inplace=True),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                    # pw-linear
                    nn.Conv2d(inp * e1, oup, 1, 1, 0, groups=group_num, bias=False),
                    nn.BatchNorm2d(oup),
                )
            elif dw_conv == 'sepdw':
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(inp, inp * e1 // 2, (3, 1), (stride, 1), (1, 0), groups=inp, bias=False),
                    nn.BatchNorm2d(inp * e1 // 2),
                    nn.Conv2d(inp * e1 // 2, inp * e1, (1, 3), (1, stride), (0, 1), groups=inp * e1 // 2, bias=False),
                    nn.BatchNorm2d(inp * e1),
                    nn.ReLU6(inplace=True),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                    # pw-linear
                    nn.Conv2d(inp * e1, oup, 1, 1, 0, groups=group_num, bias=False),
                    nn.BatchNorm2d(oup),
                )

        else:
            # conv (dw->pw->dw->pw)
            self.se_flag = se_flag
            hidden_dim1 = round(inp * e1)
            hidden_dim2 = round(oup * e2)

            if dw_conv == 'dw':
                self.conv1 = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim1, k1, stride, k1 // 2, groups=inp, bias=False),
                    nn.BatchNorm2d(hidden_dim1),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'maxdw':
                self.conv1 = nn.Sequential(
                    MaxDepthConv(inp, hidden_dim1, stride),
                    ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'sepdw':
                self.conv1 = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim1 // 2, (3, 1), (stride, 1), (1, 0), groups=inp, bias=False),
                    nn.BatchNorm2d(hidden_dim1 // 2),
                    nn.Conv2d(hidden_dim1 // 2, hidden_dim1, (1, 3), (1, stride), (0, 1), groups=hidden_dim1 // 2,
                              bias=False),
                    nn.BatchNorm2d(hidden_dim1),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential()
                )

            num_func = se_flag[0]
            self.act1 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper1 = HyperFunc(
                token_dim,
                hidden_dim1 * num_func,
                sel_token_id=hyper_token_id,
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[0] > 0 else nn.Sequential()

            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim1, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
            num_func = -1
            # num_func = 1 if se_flag[1] == 1 else -1
            self.act2 = DyReLU(num_func=num_func, scale=2.)

            if dw_conv == 'dw':
                self.conv3 = nn.Sequential(
                    nn.Conv2d(oup, hidden_dim2, k2, 1, k2 // 2, groups=oup, bias=False),
                    nn.BatchNorm2d(hidden_dim2),
                    ChannelShuffle(oup) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'maxdw':
                self.conv3 = nn.Sequential(
                    MaxDepthConv(oup, hidden_dim2, 1),
                )
            elif dw_conv == 'sepdw':
                self.conv3 = nn.Sequential(
                    nn.Conv2d(oup, hidden_dim2 // 2, (3, 1), (1, 1), (1, 0), groups=oup, bias=False),
                    nn.BatchNorm2d(hidden_dim2 // 2),
                    nn.Conv2d(hidden_dim2 // 2, hidden_dim2, (1, 3), (1, 1), (0, 1), groups=hidden_dim2 // 2,
                              bias=False),
                    nn.BatchNorm2d(hidden_dim2),
                    ChannelShuffle(oup) if group_num > 1 else nn.Sequential()
                )

            num_func = se_flag[2]
            self.act3 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper3 = HyperFunc(
                token_dim,
                hidden_dim2 * num_func,
                sel_token_id=hyper_token_id,
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[2] > 0 else nn.Sequential()

            self.conv4 = nn.Sequential(
                nn.Conv2d(hidden_dim2, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup)
            )
            num_func = 1 if se_flag[3] == 1 else -1
            self.act4 = DyReLU(num_func=num_func, scale=2.)
            self.hyper4 = HyperFunc(
                token_dim,
                oup * num_func,
                sel_token_id=hyper_token_id,
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[3] > 0 else nn.Sequential()

            self.drop_path = DropPath(cnn_drop_path_rate)

            # l2g, gb, g2l
            self.local_global = Local2Global(
                inp,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                inp_res=inp_res,
                use_dynamic=gbr_dynamic[0],
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

            self.global_block = GlobalBlock(
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                mlp_token_exp=mlp_token_exp,
                use_dynamic=gbr_dynamic[1],
                use_ffn=gbr_ffn,
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate
            )

            oup_res = inp_res // (stride * stride)

            self.global_local = Global2Local(
                oup,
                oup_res,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                use_dynamic=gbr_dynamic[2],
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

    def forward(self, x):
        features, tokens = x
        if self.use_conv_alone:
            out = self.conv(features)
        else:
            # step 1: local to global
            tokens, attn = self.local_global((features, tokens))
            tokens = self.global_block(tokens)

            # step 2: conv1 + conv2
            out = self.conv1(features)

            # process attn: mean, downsample if stride > 1, and softmax
            if self.hyper_token_id == -1:
                attn = attn.mean(dim=1)  # bs x T x H x W
                if self.stride > 1:
                    _, _, H, W = out.shape
                    attn = F.adaptive_avg_pool2d(attn, (H, W))
                attn = torch.softmax(attn, dim=1)

            if self.se_flag[0] > 0:
                hp = self.hyper1((tokens, attn))
                out = self.act1((out, hp))
            else:
                out = self.act1(out)

            out = self.conv2(out)
            out = self.act2(out)

            # step 4: conv3 + conv 4
            out_cp = out
            out = self.conv3(out)
            if self.se_flag[2] > 0:
                hp = self.hyper3((tokens, attn))
                out = self.act3((out, hp))
            else:
                out = self.act3(out)

            out = self.conv4(out)
            if self.se_flag[3] > 0:
                hp = self.hyper4((tokens, attn))
                out = self.act4((out, hp))
            else:
                out = self.act4(out)

            out = self.drop_path(out) + out_cp

            # step 3: global to local
            out = self.global_local((out, tokens))

        if self.identity:
            out = out + features

        return (out, tokens)


class DnaBlock(nn.Module):
    def __init__(
            self,
            inp,
            oup,
            stride,
            exp_ratios,  # (e1, e2)
            kernel_size=(3, 3),
            dw_conv='dw',
            group_num=1,
            se_flag=[2, 0, 2, 0],
            hyper_token_id=0,
            hyper_reduction_ratio=4,
            token_dim=128,
            token_num=6,
            inp_res=49,
            gbr_type='mlp',
            gbr_dynamic=[False, False, False],
            gbr_ffn=False,
            gbr_before_skip=False,
            mlp_token_exp=4,
            norm_pos='post',
            drop_path_rate=0.,
            cnn_drop_path_rate=0.,
            attn_num_heads=2,
            remove_proj_local=True,
    ):
        super(DnaBlock, self).__init__()

        print(f'block: {inp_res}, cnn-drop {cnn_drop_path_rate:.4f}, mlp-drop {drop_path_rate:.4f}')
        if isinstance(exp_ratios, tuple):
            e1, e2 = exp_ratios
        else:
            e1, e2 = exp_ratios, 4
        k1, k2 = kernel_size

        self.stride = stride
        self.hyper_token_id = hyper_token_id

        self.gbr_before_skip = gbr_before_skip
        self.identity = stride == 1 and inp == oup
        self.use_conv_alone = False
        if e1 == 1 or e2 == 0:
            self.use_conv_alone = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp * e1, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp * e1),
                nn.ReLU6(inplace=True),
                ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                # pw-linear
                nn.Conv2d(inp * e1, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # conv (pw->dw->pw)
            self.se_flag = se_flag
            hidden_dim = round(inp * e1)

            self.conv1 = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(hidden_dim),
                ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
            )

            num_func = se_flag[0]
            self.act1 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper1 = HyperFunc(
                token_dim,
                hidden_dim * num_func,
                sel_token_id=hyper_token_id,
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[0] > 0 else nn.Sequential()

            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, k1, stride, k1 // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            num_func = se_flag[2]  # note here we used index 2 to be consistent with block2
            self.act2 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper2 = HyperFunc(
                token_dim,
                hidden_dim * num_func,
                sel_token_id=hyper_token_id,
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[2] > 0 else nn.Sequential()

            self.conv3 = nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
                ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
            )
            num_func = 1 if se_flag[3] == 1 else -1
            self.act3 = DyReLU(num_func=num_func, scale=2.)
            self.hyper3 = HyperFunc(
                token_dim,
                oup * num_func,
                sel_token_id=hyper_token_id,
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[3] > 0 else nn.Sequential()

            self.drop_path = DropPath(cnn_drop_path_rate)

            # l2g, gb, g2l
            self.local_global = Local2Global(
                inp,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                inp_res=inp_res,
                use_dynamic=gbr_dynamic[0],
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

            self.global_block = GlobalBlock(
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                mlp_token_exp=mlp_token_exp,
                use_dynamic=gbr_dynamic[1],
                use_ffn=gbr_ffn,
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate
            )

            oup_res = inp_res // (stride * stride)

            self.global_local = Global2Local(
                oup,
                oup_res,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                use_dynamic=gbr_dynamic[2],
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

    def forward(self, x):
        features, tokens = x
        if self.use_conv_alone:
            out = self.conv(features)
            if self.identity:
                out = self.drop_path(out) + features

        else:
            # step 1: local to global
            tokens, attn = self.local_global((features, tokens))
            tokens = self.global_block(tokens)

            # step 2: conv1 + conv2 + conv3
            out = self.conv1(features)

            # process attn: mean, downsample if stride > 1, and softmax
            if self.hyper_token_id == -1:
                attn = attn.mean(dim=1)  # bs x T x H x W
                if self.stride > 1:
                    _, _, H, W = out.shape
                    attn = F.adaptive_avg_pool2d(attn, (H, W))
                attn = torch.softmax(attn, dim=1)

            if self.se_flag[0] > 0:
                hp = self.hyper1((tokens, attn))
                out = self.act1((out, hp))
            else:
                out = self.act1(out)

            out = self.conv2(out)
            if self.se_flag[2] > 0:
                hp = self.hyper2((tokens, attn))
                out = self.act2((out, hp))
            else:
                out = self.act2(out)

            out = self.conv3(out)
            if self.se_flag[3] > 0:
                hp = self.hyper3((tokens, attn))
                out = self.act3((out, hp))
            else:
                out = self.act3(out)

            # step 3: global to local and skip
            if self.gbr_before_skip == True:
                out = self.global_local((out, tokens))
                if self.identity:
                    out = self.drop_path(out) + features
            else:
                if self.identity:
                    out = self.drop_path(out) + features
                out = self.global_local((out, tokens))

        return (out, tokens)


##########################################################################################################
# classifier
##########################################################################################################
class MergeClassifier(nn.Module):
    def __init__(
            self, inp,
            oup=1280,
            ch_exp=6,
            num_classes=1000,
            drop_rate=0.,
            drop_branch=[0.0, 0.0],
            group_num=1,
            token_dim=128,
            cls_token_num=1,
            last_act='relu',
            hyper_token_id=0,
            hyper_reduction_ratio=4
    ):
        super(MergeClassifier, self).__init__()

        self.drop_branch = drop_branch
        self.cls_token_num = cls_token_num

        hidden_dim = inp * ch_exp
        self.conv = nn.Sequential(
            ChannelShuffle(group_num) if group_num > 1 else nn.Sequential(),
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=group_num, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        self.last_act = last_act
        num_func = 2 if last_act == 'dyrelu' else 0
        self.act = DyReLU(num_func=num_func, scale=2.)

        self.hyper = HyperFunc(
            token_dim,
            hidden_dim * num_func,
            sel_token_id=hyper_token_id,
            reduction_ratio=hyper_reduction_ratio
        ) if last_act == 'dyrelu' else nn.Sequential()

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )

        if cls_token_num > 0:
            cat_token_dim = token_dim * cls_token_num
        elif cls_token_num == 0:
            cat_token_dim = token_dim
        else:
            cat_token_dim = 0

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + cat_token_dim, oup),
            nn.BatchNorm1d(oup),
            h_swish()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(oup, num_classes)
        )

    def forward(self, x):
        features, tokens = x

        x = self.conv(features)

        if self.last_act == 'dyrelu':
            hp = self.hyper(tokens)
            x = self.act((x, hp))
        else:
            x = self.act(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ps = [x]

        if self.cls_token_num == 0:
            avg_token = torch.mean(F.relu6(tokens), dim=0)
            ps.append(avg_token)
        elif self.cls_token_num < 0:
            pass
        else:
            for i in range(self.cls_token_num):
                ps.append(tokens[i])

        # drop branch
        if self.training and self.drop_branch[0] + self.drop_branch[1] > 1e-8:
            rd = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device)
            keep_local = 1 - self.drop_branch[0]
            keep_global = 1 - self.drop_branch[1]
            rd_local = (keep_local + rd).floor_()
            rd_global = -((rd - keep_global).floor_())
            ps[0] = ps[0].div(keep_local) * rd_local
            ps[1] = ps[1].div(keep_global) * rd_global

        x = torch.cat(ps, dim=1)
        x = self.fc(x)

        x = self.classifier(x)
        return x


""" Model Registry
Hacked together by / Copyright 2020 Ross Wightman
"""

import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy

__all__ = ['list_models', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
           'is_model_default_key', 'has_model_default_key', 'get_model_default_value', 'is_model_pretrained']

_module_to_models = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module = {}  # mapping of model names to module names
_model_entrypoints = {}  # mapping of model names to entrypoint fns
_model_has_pretrained = set()  # set of model names that have pretrained weight url present
_model_default_cfgs = dict()  # central repo for model default_cfgs


def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False  # check if model has a pretrained url to allow filtering on this
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]['url']
        _model_default_cfgs[model_name] = deepcopy(mod.default_cfgs[model_name])
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(filter='', module='', pretrained=False, exclude_filters='', name_matches_cfg=False):
    """ Return list of available model names, sorted alphabetically

    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        module (str) - Limit model selection to a specific sub-module (ie 'gen_efficientnet')
        pretrained (bool) - Include only models with pretrained weights if True
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
        name_matches_cfg (bool) - Include only models w/ model_name matching default_cfg name (excludes some aliases)

    Example:
        model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
        model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
    """
    if module:
        models = list(_module_to_models[module])
    else:
        models = _model_entrypoints.keys()
    if filter:
        models = fnmatch.filter(models, filter)  # include these models
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    if pretrained:
        models = _model_has_pretrained.intersection(models)
    if name_matches_cfg:
        models = set(_model_default_cfgs).intersection(models)
    return list(sorted(models, key=_natural_key))


def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def list_modules():
    """ Return list of module names that contain models / model entrypoints
    """
    modules = _module_to_models.keys()
    return list(sorted(modules))


def is_model_in_modules(model_name, module_names):
    """Check if a model exists within a subset of modules
    Args:
        model_name (str) - name of model to check
        module_names (tuple, list, set) - names of modules to search in
    """
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)


def has_model_default_key(model_name, cfg_key):
    """ Query model default_cfgs for existence of a specific key.
    """
    if model_name in _model_default_cfgs and cfg_key in _model_default_cfgs[model_name]:
        return True
    return False


def is_model_default_key(model_name, cfg_key):
    """ Return truthy value for specified model default_cfg key, False if does not exist.
    """
    if model_name in _model_default_cfgs and _model_default_cfgs[model_name].get(cfg_key, False):
        return True
    return False


def get_model_default_value(model_name, cfg_key):
    """ Get a specific model default_cfg value by key. None if it doesn't exist.
    """
    if model_name in _model_default_cfgs:
        return _model_default_cfgs[model_name].get(cfg_key, None)
    else:
        return None


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained



def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

default_cfgs = {
    'default': _cfg(url=''),
}

class MobileFormer(nn.Module):
    def __init__(
        self,
        block_args,
        num_classes=1000,
        img_size=224,
        width_mult=1.,
        in_chans=3,
        stem_chs=16,
        num_features=1280,
        dw_conv='dw',
        kernel_size=(3,3),
        cnn_exp=(6,4),
        group_num=1,
        se_flag=[2,0,2,0],
        hyper_token_id=0,
        hyper_reduction_ratio=4,
        token_dim=128,
        token_num=6,
        cls_token_num=1,
        last_act='relu',
        last_exp=6,
        gbr_type='mlp',
        gbr_dynamic=[False, False, False],
        gbr_norm='post',
        gbr_ffn=False,
        gbr_before_skip=False,
        gbr_drop=[0.0, 0.0],
        mlp_token_exp=4,
        drop_rate=0.,
        drop_path_rate=0.,
        cnn_drop_path_rate=0.,
        attn_num_heads = 2,
        remove_proj_local=True,
        ):

        super(MobileFormer, self).__init__()

        cnn_drop_path_rate = drop_path_rate
        mdiv = 8 if width_mult > 1.01 else 4
        self.num_classes = num_classes

        #global tokens
        self.tokens = nn.Embedding(token_num, token_dim) 

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chs, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.ReLU6(inplace=True)
        )
        input_channel = stem_chs

        # blocks
        layer_num = len(block_args)
        inp_res = img_size * img_size // 4
        layers = []
        for idx, val in enumerate(block_args):
            b, t, c, n, s, t2 = val # t2 for block2 the second expand
            block = eval(b)

            t = (t, t2)
            output_channel = _make_divisible(c * width_mult, mdiv) if idx > 0 else _make_divisible(c * width_mult, 4) 

            drop_path_prob = drop_path_rate * (idx+1) / layer_num
            cnn_drop_path_prob = cnn_drop_path_rate * (idx+1) / layer_num

            layers.append(block(
                input_channel, 
                output_channel, 
                s, 
                t, 
                dw_conv=dw_conv,
                kernel_size=kernel_size,
                group_num=group_num,
                se_flag=se_flag,
                hyper_token_id=hyper_token_id,
                hyper_reduction_ratio=hyper_reduction_ratio,
                token_dim=token_dim, 
                token_num=token_num,
                inp_res=inp_res,
                gbr_type=gbr_type,
                gbr_dynamic=gbr_dynamic,
                gbr_ffn=gbr_ffn,
                gbr_before_skip=gbr_before_skip,
                mlp_token_exp=mlp_token_exp,
                norm_pos=gbr_norm,
                drop_path_rate=drop_path_prob,
                cnn_drop_path_rate=cnn_drop_path_prob,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,        
            ))
            input_channel = output_channel

            if s == 2:
                inp_res = inp_res // 4

            for i in range(1, n):
                layers.append(block(
                    input_channel, 
                    output_channel, 
                    1, 
                    t, 
                    dw_conv=dw_conv,
                    kernel_size=kernel_size,
                    group_num=group_num,
                    se_flag=se_flag,
                    hyper_token_id=hyper_token_id,
                    hyper_reduction_ratio=hyper_reduction_ratio,
                    token_dim=token_dim, 
                    token_num=token_num,
                    inp_res=inp_res,
                    gbr_type=gbr_type,
                    gbr_dynamic=gbr_dynamic,
                    gbr_ffn=gbr_ffn,
                    gbr_before_skip=gbr_before_skip,
                    mlp_token_exp=mlp_token_exp,
                    norm_pos=gbr_norm,
                    drop_path_rate=drop_path_prob,
                    cnn_drop_path_rate=cnn_drop_path_prob,
                    attn_num_heads=attn_num_heads,
                    remove_proj_local=remove_proj_local,
                ))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # last layer of local to global
        self.local_global = Local2Global(
            input_channel,
            block_type = gbr_type,
            token_dim=token_dim,
            token_num=token_num,
            inp_res=inp_res,
            use_dynamic = gbr_dynamic[0],
            norm_pos=gbr_norm,
            drop_path_rate=drop_path_rate,
            attn_num_heads=attn_num_heads
        )

        # classifer
        self.classifier = MergeClassifier(
            input_channel, 
            oup=num_features, 
            ch_exp=last_exp,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_branch=gbr_drop,
            group_num=group_num,
            token_dim=token_dim,
            cls_token_num=cls_token_num,
            last_act = last_act,
            hyper_token_id=hyper_token_id,
            hyper_reduction_ratio=hyper_reduction_ratio
        )

        #initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        # setup tokens
        bs, _, _, _ = x.shape
        z = self.tokens.weight
        tokens = z[None].repeat(bs, 1, 1).clone()
        tokens = tokens.permute(1, 0, 2)
 
        # stem -> features -> classifier
        x = self.stem(x)
        x, tokens = self.features((x, tokens))
        tokens, attn = self.local_global((x, tokens))
        y = self.classifier((x, tokens))

        return y

def _create_mobile_former(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        MobileFormer, 
        variant, 
        pretrained,
        default_cfg=default_cfgs['default'],
        **kwargs)
    print(model)

    return model

common_model_kwargs = dict(
    cnn_drop_path_rate = 0.1,
    dw_conv = 'dw',
    kernel_size=(3, 3),
    cnn_exp = (6, 4),
    cls_token_num = 1,
    hyper_token_id = 0,
    hyper_reduction_ratio = 4,
    attn_num_heads = 2,
    gbr_norm = 'post',
    mlp_token_exp = 4,
    gbr_before_skip = False,
    gbr_drop = [0., 0.],
    last_act = 'relu',
    remove_proj_local = True,
)

@register_model
def mobile_former_508m(pretrained=False, **kwargs):

    #stem = 24
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  24, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  40, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  3,  40, 1, 1, 3], #3
        ['DnaBlock3', 6,  72, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  3,  72, 1, 1, 3], #5
        ['DnaBlock3', 6, 128, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4, 128, 1, 1, 4], #7
        ['DnaBlock',  6, 176, 1, 1, 4], #8
        ['DnaBlock',  6, 176, 1, 1, 4], #9
        ['DnaBlock3', 6, 240, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 240, 1, 1, 4], #11
        ['DnaBlock',  6, 240, 1, 1, 4], #12
    ]
   
    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1920,
        stem_chs = 24,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_508m", pretrained, **model_kwargs)
    return model

@register_model
def mobile_former_294m(pretrained=False, **kwargs):

    #stem = 16
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  16, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  24, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  4,  24, 1, 1, 4], #3
        ['DnaBlock3', 6,  48, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  4,  48, 1, 1, 4], #5
        ['DnaBlock3', 6,  96, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4,  96, 1, 1, 4], #7
        ['DnaBlock',  6, 128, 1, 1, 4], #8
        ['DnaBlock',  6, 128, 1, 1, 4], #9
        ['DnaBlock3', 6, 192, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 192, 1, 1, 4], #11
        ['DnaBlock',  6, 192, 1, 1, 4], #12
    ]
  
    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1920,
        stem_chs = 16,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_294m", pretrained, **model_kwargs)
    return model

@register_model
def mobile_former_214m(pretrained=False, **kwargs):

    #stem = 12
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  12, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  20, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  3,  20, 1, 1, 4], #3
        ['DnaBlock3', 6,  40, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  4,  40, 1, 1, 4], #5
        ['DnaBlock3', 6,  80, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4,  80, 1, 1, 4], #7
        ['DnaBlock',  6, 112, 1, 1, 4], #8
        ['DnaBlock',  6, 112, 1, 1, 4], #9
        ['DnaBlock3', 6, 160, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 160, 1, 1, 4], #11
        ['DnaBlock',  6, 160, 1, 1, 4], #12
    ]


    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1600,
        stem_chs = 12,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_214m", pretrained, **model_kwargs)
    return model

@register_model
def mobile_former_151m(pretrained=False, **kwargs):

    #stem = 12
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  12, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  16, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  3,  16, 1, 1, 3], #3
        ['DnaBlock3', 6,  32, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  3,  32, 1, 1, 3], #5
        ['DnaBlock3', 6,  64, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4,  64, 1, 1, 4], #7
        ['DnaBlock',  6,  88, 1, 1, 4], #8
        ['DnaBlock',  6,  88, 1, 1, 4], #9
        ['DnaBlock3', 6, 128, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 128, 1, 1, 4], #11
        ['DnaBlock',  6, 128, 1, 1, 4], #12
    ]

    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1280,
        stem_chs = 12,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_151m", pretrained, **model_kwargs)
    return model

@register_model
def mobile_former_96m(pretrained=False, **kwargs):

    #stem = 12
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  12, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  16, 1, 2, 4], #2 56x56 (1)
        ['DnaBlock3', 6,  32, 1, 2, 4], #3 28x28 (2)
        ['DnaBlock',  3,  32, 1, 1, 3], #4
        ['DnaBlock3', 6,  64, 1, 2, 4], #5 14x14 (3)
        ['DnaBlock',  4,  64, 1, 1, 4], #6
        ['DnaBlock',  6,  88, 1, 1, 4], #7
        ['DnaBlock3', 6, 128, 1, 2, 4], #8 7x7 (2)
        ['DnaBlock',  6, 128, 1, 1, 4], #9
    ]


    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1280,
        stem_chs = 12,
        token_num = 4,
        token_dim = 128,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_96m", pretrained, **model_kwargs)
    return model

@register_model
def mobile_former_52m(pretrained=False, **kwargs):

    #stem = 8
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 3,  12, 1, 2, 0], #1 56x56 (2)
        ['DnaBlock',  3,  12, 1, 1, 3], #2
        ['DnaBlock3', 6,  24, 1, 2, 4], #3 28x28 (2)
        ['DnaBlock',  3,  24, 1, 1, 3], #4
        ['DnaBlock3', 6,  48, 1, 2, 4], #5 14x14 (3)
        ['DnaBlock',  4,  48, 1, 1, 4], #6
        ['DnaBlock',  6,  64, 1, 1, 4], #7
        ['DnaBlock3', 6,  96, 1, 2, 4], #8 7x7 (2)
        ['DnaBlock',  6,  96, 1, 1, 4], #9
    ]

    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1024,
        stem_chs = 8,
        token_num = 3,
        token_dim = 128,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_52m", pretrained, **model_kwargs)
    return model

@register_model
def mobile_former_26m(pretrained=False, **kwargs):

    #stem = 8
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 3,  12, 1, 2, 0], #1 56x56 (2)
        ['DnaBlock',  3,  12, 1, 1, 3], #2
        ['DnaBlock3', 6,  24, 1, 2, 4], #3 28x28 (2)
        ['DnaBlock',  3,  24, 1, 1, 3], #4
        ['DnaBlock3', 6,  48, 1, 2, 4], #5 14x14 (3)
        ['DnaBlock',  4,  48, 1, 1, 4], #6
        ['DnaBlock',  6,  64, 1, 1, 4], #7
        ['DnaBlock3', 6,  96, 1, 2, 4], #8 7x7 (2)
        ['DnaBlock',  6,  96, 1, 1, 4], #9
    ]

    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 4,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1024,
        stem_chs = 8,
        token_num = 3,
        token_dim = 128,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_mobile_former("mobile_former_26m", pretrained, **model_kwargs)
    return model

