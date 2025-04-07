# MIT License
# 
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Mainly modified from:
# - https://github.com/hasibzunair/3D-image-classification-tutorial/blob/master/3D_image_classification.ipynb
# ==============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=2,
                 padding=1,
                 dilation=1,
                 groups=1,
                 norm=True,
                 activation=True,
                 ):
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, groups=groups,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.activation = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        print("xA", x.shape)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class BasicConv3d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=2,
                 padding=1,
                 dilation=1,
                 groups=1,
                 norm=True,
                 activation=True,
                 ):
        super(BasicConv3d, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, groups=groups,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm3d(out_channels) if norm else None
        self.activation = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        print("xA", x.shape)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



class BasicConv3dA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 norm=True,
                 activation=True,
                 ):
        super(BasicConv3dA, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, groups=groups,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm3d(out_channels) if norm else None
        self.activation = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        print("xA", x.shape)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class COVID19CT3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(COVID19CT3D, self).__init__()

        self.backbone = nn.Sequential(
            BasicConv3d(3,
                        64,
                        kernel_size=3),
            BasicConv3dA(64,
                        64,
                        kernel_size=3),
            BasicConv3d(64,
                        64,
                        kernel_size=3),
            BasicConv3d(64,
                        128,
                        kernel_size=3),
            BasicConv3d(128,
                        256,
                        kernel_size=3),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        # x = x.unsqueeze(4)
        # x = x.transpose(1, 3)
        # print("x", x.shape)
        # x = torch.cat([x, x, x, x, x], dim=4)

        x = self.backbone(x)
        x = self.head(x)

        return x
