import math
import torch
import torch.nn as nn

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        # print("mapper_x", mapper_x)     #  [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2]
        # print("mapper_y", mapper_y)     #  [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2]
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        # self.fc = nn.Sequential(
        #         nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
        #         nn.BatchNorm2d(channel // reduction),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
        #         nn.BatchNorm2d(channel),
        # )

        groups = reduction * 2
        mip = max(8, channel // groups)

        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        n,c,h,w = x.shape
        # print("x", x.shape)         # ([1, 64, 56, 56])     ([2, 64, 56, 56])
        identity = x
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        # # print("x_pooled", x_pooled.shape)       # ([1, 64, 56, 56])
        # y = self.dct_layer(x_pooled)
        # # print("y", y.shape)     # ([1, 64])
        # y = self.fc(y).view(n, c, 1, 1)
        # # print("y", y.shape)     # ([1, 64, 1, 1])

        y, result_h, result_w = self.dct_layer(x_pooled)

        x_h = result_h
        # print("x_h", x_h.shape)         # te([2, 64, 56, 1])
        x_w = result_w.permute(0, 1, 3, 2)
        x_z = y.view(n, c, 1, 1)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_z = self.conv1(x_z)
        x_z = self.bn1(x_z)
        x_z = self.relu(x_z)
        # print("x_z", x_z.shape)     # ([1, 8, 1, 1])
        x_z = x_z.expand(-1, -1, h, -1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_h = x_h + x_z
        x_w = x_w + x_z
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        # print("x_z", x_z.shape)      # ([1, 64, 56, 56])

        y = identity * x_w * x_h





        # y = self.fc(y)

        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        # print("mapper_x", mapper_x)     #  [0, 0, 48, 0, 0, 8, 8, 32, 40, 8, 24, 0, 0, 0, 24, 16]
        # print("mapper_y", mapper_y)     # [0, 8, 0, 40, 16, 0, 16, 0, 0, 48, 0, 32, 48, 24, 40, 16]
        # print("channel", channel)       # 64    96  128

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight
        # print("x", x.shape)               #(([2, 64, 56, 56])

        result = torch.sum(x, dim=[2,3])
        result_h = torch.sum(x, dim=[3]).unsqueeze(3)
        result_w = torch.sum(x, dim=[2]).unsqueeze(2)
        # print("result", result.shape)               # ([2, 64])
        # print("result_h", result_h.shape)               # ([2, 64, 56])
        # print("result_w", result_w.shape)               # ([2, 64, 56])
        return result, result_h, result_w

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        # print("dct_filter", dct_filter.shape)               #([64, 56, 56])
        return dct_filter