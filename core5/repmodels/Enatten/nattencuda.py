"""
Neighborhood Attention PyTorch Module (CUDA only)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import nn
from torch.nn.functional import pad
from timm.models.layers import trunc_normal_
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

from torch.utils.cpp_extension import load

# ELSA
from .elsa import elsa_op


# nattenav_cuda = load(
#     'nattenav_cuda', ['./src/nattenav_cuda.cpp', './src/nattenav_cuda_kernel.cu'], verbose=False)
# nattenqkrpb_cuda = load(
#     'nattenqkrpb_cuda', ['./src/nattenqkrpb_cuda.cpp', './src/nattenqkrpb_cuda_kernel.cu'], verbose=False)

nattenav_cuda = load(
    'nattenav_cuda', ['./core5/repmodels/natten/src/nattenav_cuda.cpp', './core5/repmodels/natten/src/nattenav_cuda_kernel.cu'], verbose=False)
nattenqkrpb_cuda = load(
    'nattenqkrpb_cuda', ['./core5/repmodels/natten/src/nattenqkrpb_cuda.cpp', './core5/repmodels/natten/src/nattenqkrpb_cuda_kernel.cu'], verbose=False)

# import nattenav_cuda
# import nattenqkrpb_cuda

# try:
#     from torch.utils.cpp_extension import load
#     nattenav_cuda = load(
#         'nattenav_cuda', ['./natten/src/nattenav_cuda.cpp', './natten/src/nattenav_cuda_kernel.cu'], verbose=False)
#     nattenqkrpb_cuda = load(
#         'nattenqkrpb_cuda', ['./natten/src/nattenqkrpb_cuda.cpp', './natten/src/nattenqkrpb_cuda_kernel.cu'], verbose=False)
# except:
#     try:
#         # import nattenav_cuda
#         # import nattenqkrpb_cuda
#         import nattenav_cuda
#         import nattenqkrpb_cuda
#     except:
#         raise RuntimeError("Could not load NATTEN CUDA extension. " +
#                            "Please make sure your device has CUDA, the CUDA toolkit for PyTorch is installed, and that you've compiled NATTEN correctly.")


class NATTENAVFunction(Function):
    """
    AV autograd function
    Computes neighborhood attention outputs given attention weights, and values.
    This calls the `AV` kernel.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value):
        attn = attn.contiguous()
        value = value.contiguous()
        out = nattenav_cuda.forward(
                attn, 
                value)
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenav_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_attn, d_value = outputs
        return d_attn, d_value


class NATTENQKRPBFunction(Function):
    """
    QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = nattenqkrpb_cuda.forward(
                query,
                key,
                rpb)
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenqkrpb_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb


class NeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention Module
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, and 11; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim*2, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        dilation = 1
        self.dim_qk = dim
        group_width = 8
        groups = 1
        kernel_size = 7
        self.kernel = 7
        self.attn = nn.Sequential(
            # nn.Conv2d(self.dim_qk, self.dim_qk, kernel_size, padding=(kernel_size // 2)*dilation, dilation=dilation, groups=self.dim_qk // group_width),
            nn.Conv2d(self.dim_qk, self.dim_qk, kernel_size, padding=(kernel_size // 2)*dilation, dilation=dilation, groups= 54),
            nn.GELU(),
            nn.Conv2d(self.dim_qk, kernel_size ** 2 * num_heads, 1, groups=groups))
        # ghost_add = torch.zeros(1, self.dim_v, kernel_size, kernel_size)
        ghost_add = torch.zeros(1, 432, kernel_size, kernel_size)
        trunc_normal_(ghost_add, std=.02)
        self.ghost_head = nn.Parameter(ghost_add, requires_grad=True)


    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        # print("self.qkv(x)", self.qkv(x).shape)     # torch.Size([1, 14, 14, 768])      # ([16, 14, 14, 768])
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        # print("qkv", qkv.shape)     # ([3, 1, 16, 14, 14, 16])      # ([3, 16, 16, 14, 14, 16])
        q, k, v = qkv[0], qkv[1], qkv[2]
        tmp_dim = q.shape[4]
        # q torch.Size([1, 16, 14, 14, 16])
        # k torch.Size([1, 16, 14, 14, 16])
        # v torch.Size([1, 16, 14, 14, 16])
        q = q * self.scale
        # print("self.rpb", self.rpb.shape)       # torch.Size([16, 13, 13])
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        # # print("attn", attn.shape)       # ([1, 16, 14, 14, 16])       # ([16, 16, 14, 14, 49])
        attn = attn.softmax(dim=-1)
        attn_A = self.attn_drop(attn)

        # Hadamard注意力可以描述如下
        # # print("attn.permute(0, 2, 3, 1, 4)", attn.permute(0, 2, 3, 1, 4).shape)       # torch.Size([1, 14, 14, 16, 49])
        # attn = attn.permute(0, 2, 3, 1, 4).reshape(B, H, W, self.num_heads * tmp_dim).permute(0, 3, 1, 2)
        # # print("attn", attn.shape)       # ([1, 14, 14, 256])    ([16, 14, 14, 784])
        # # attn = attn.permute(0, 3, 1, 2)
        # # print("attn", attn.shape)       # ([1, 256, 14, 14])    ([16, 784, 14, 14]
        # attn = attn.permute(0, 2, 3, 1).reshape(B, H, W, self.num_heads, tmp_dim).permute(0, 3, 1, 2, 4)
        # print("attn", attn.shape)       # ([1, 16, 14, 14, 16])  [16, 16, 14, 14, 49])
        q = q.permute(0, 2, 3, 1, 4).reshape(B, H, W, self.num_heads * tmp_dim).permute(0, 3, 1, 2)
        k = k.permute(0, 2, 3, 1, 4).reshape(B, H, W, self.num_heads * tmp_dim).permute(0, 3, 1, 2)
        hadamard_product = q * k * self.scale
        # if self.stride > 1:
        #     hadamard_product = F.avg_pool2d(hadamard_product, self.stride)
        h_attn = self.attn(hadamard_product)
        h_attn = h_attn.softmax(dim=1)          #  ([16, 784, 14, 14]
        # # tmp_had = h_attn.shape[1] // self.num_heads
        # # h_attn = h_attn.permute(0, 2, 3, 1).reshape(B, H, W, self.num_heads, tmp_had).permute(0, 3, 1, 2, 4)
        h_attn = h_attn.permute(0, 2, 3, 1).reshape(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        # # print("h_attn", h_attn.shape)       # ([16, 16, 14, 14, 25])
        # h_attn = h_attn.softmax(dim=-1)
        h_attn = self.attn_drop(h_attn)
        attn_A = torch.cat([attn_A, h_attn], dim=4)           # ([16, 16, 14, 14, 98])
        # print("attn", attn.shape)       # ([16, 16, 14, 14, 74])

        # Ghost头则受启发于GhostNet得到，可以描述如下
        ghost_mul = None
        v_tmp = v.permute(0, 2, 3, 1, 4).reshape(B, H, W, self.num_heads * tmp_dim).permute(0, 3, 1, 2)     #  ([16, 256, 14, 14]
        attn_tmp = attn_A.permute(0, 2, 3, 1, 4).reshape(B, H, W, -1).permute(0, 3, 1, 2)     #   ([16, 1568, 14, 14]
        ghost_add = self.ghost_head.expand(v_tmp.shape[0], v_tmp.shape[1], self.kernel, self.kernel)
        # print("self.ghost_add", ghost_add.shape)            # ([16, 256, 7, 7])
        attn_tmp = elsa_op(v_tmp, ghost_mul, ghost_add, attn_tmp, 0, 1, self.kernel, 1, 1)
        # print("attn", attn.shape)       # ([16, 256, 14, 14])
        attn_tmp = attn_tmp.permute(0, 2, 3, 1).reshape(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        # print("attn", attn.shape)       # ([16, 16, 14, 14, 16])

        # print("attn", attn.shape)       #([16, 16, 14, 14, 49])
        # print("v", v.shape)       #([16, 16, 14, 14, 16])
        x = NATTENAVFunction.apply(attn_A, v)
        # x = (x + attn_tmp)
        x = torch.cat([x, attn_tmp], dim=4)
        # print("x", x.shape)     # ([16, 16, 14, 14, 16])
        # x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C*2)
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        # return self.proj_drop(self.proj(x))
        return self.proj_drop(self.proj2(x))

