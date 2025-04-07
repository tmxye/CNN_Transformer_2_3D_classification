import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
# from .quant_noise import quant_noise
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"
    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ
    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert module.weight.size(1) % block_size == 0, "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert module.in_channels % block_size == 0, "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])

            # scale weights and apply mask
            mask = mask.to(torch.bool)  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

class Xformer(nn.Module):
    def __init__(
        self, 
        input_dim, 
        num_heads, 
        dropout=0.0, 
        scalar=2, 
        max_seq_len=280, 
        bias=True, 
        q_noise=0.0, 
        qn_block_size=8):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.scalar = scalar
        self.head_dim = (input_dim // num_heads) // scalar
        assert (
            input_dim == self.head_dim * num_heads * scalar
        ), "input_dim must be divisible by num_heads and scalar."
        self.scaling = (input_dim // num_heads) ** -0.5

        self.k_v_proj = quant_noise(nn.Linear(max_seq_len, max_seq_len//scalar, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(input_dim, input_dim//scalar, bias=bias), q_noise, qn_block_size)

        self.out_proj = quant_noise(nn.Linear(input_dim, input_dim, bias=bias), q_noise, qn_block_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, input_tensor):
        input_length, batch_size, hidden_dim = input_tensor.shape
        # X = input_tensor.flatten(2)

        X = rearrange(input_tensor, 'l b d -> b l d')
        q = self.q_proj(X)
        v = rearrange(X, 'b l d -> b d l') @ self.k_v_proj.weight.T[:input_length, :input_length//self.scalar+1]
        k = rearrange(q, 'b l d -> b d l') @ self.k_v_proj.weight.T[:input_length, :input_length//self.scalar+1]
        
        q = q * self.scaling
        
        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.num_heads)
        k = rearrange(k, 'b (d h) l -> (b h) d l', h=self.num_heads)
        v = rearrange(v, 'b (d h) l -> (b h) d l', h=self.num_heads)
        v = rearrange(v, 'b d l -> b l d')


        attn_weights = q @ k
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training
        )
        X = attn_probs @ v
        X = rearrange(X, '(b h) l d -> b l (d h)', h=self.num_heads)
        X = self.out_proj(X)
        X = rearrange(X, 'b l d -> l b d')

        return X

import torch
from fairseq.modules.multihead_attention import MultiheadAttention
# from xformer_pytorch import Xformer


hidden_dim = 512
n_heads = 4
batch_size = 40
length = 1024

baseline_attn = MultiheadAttention(hidden_dim, n_heads, self_attention=True).cuda()
test_input = torch.ones((length, batch_size, hidden_dim)).cuda()
dummy_out = baseline_attn(test_input, test_input, test_input)

# To use less hyperparameters, we let scalar = alpha = beta here.
scalar = 2
xformer_attn = Xformer(hidden_dim, n_heads, max_seq_len=length, scalar=scalar).cuda()
output = xformer_attn(test_input)

def Xformer_YXY():
    hidden_dim = 512
    n_heads = 4
    # batch_size = 40
    length = 1024
    scalar = 2
    return Xformer(hidden_dim, n_heads, max_seq_len=length, scalar=scalar)

