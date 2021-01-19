import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn
import torch.nn.functional as F
import pdb
import os

def cosine_distance(input, isAttention=True, doWrite=False):
    cos = nn.CosineSimilarity(dim=0, eps=1e-5)
    if isAttention:
        '''
        input : [B,C,C] or [B,HW,HW]
        '''
        _, channel, _ = input.size()
        x = input[0] # CxC
        sum = 0.0
        for i in range(channel):
            for j in range(channel):
                sum += (1 - cos(x[i], x[j]))/2.
        avg = sum / float(channel**2)

        # for j in range(channel):
        #     sum += (1 - cos(x[0], x[j]))/2.
        # avg = sum / float(channel)

        if doWrite:
            with open('att2.txt','a') as f:
                f.write(f'{avg}\n')
    else:
        '''
        input : B,C,H,W
        '''
        _, channel, height, width = input.size()
        x = input[0]
        x = x.view(channel,height*width).permute(1,0) # [HW,C]
        sum = 0.0
        for i in range(height*width):
            for j in range(height*width):
                sum += (1 - cos(x[i], x[j]))/2.
        avg = sum / float((height*width)**2)

        # for j in range(height*width):
        #     sum += (1 - cos(x[0], x[j]))/2.
        # avg = sum / float((height*width))

        if doWrite:
            with open('out2.txt','a') as f:
                f.write(f'{avg}\n')

    return avg

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio=0.25,
                 pooling_type='att',
                 fusion_types=('channel_add', )): # 'channel_mul' 도 추가해서 실험해봐야겠는데
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        self.gamma = nn.Parameter(torch.zeros(1))
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        # my custom code
        # if self.channel_mul_conv is not None:
        #     # [N, C, 1, 1]
        #     transform = self.channel_mul_conv(context)
        #     out = out * torch.sigmoid(transform)
        # if self.channel_add_conv is not None:
        #     # [N, C, 1, 1]
        #     # channel_add_term = self.channel_add_conv(context)
        #     out = out + transform

        # original code
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + self.gamma * channel_add_term
        return out

class ChannelBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio=0.25,
                 pooling_type='att',
                 fusion_types=('channel_add', )): # 'channel_mul' 도 추가해서 실험해봐야겠는데
        super(ChannelBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        self.gamma = nn.Parameter(torch.zeros(1))
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.BatchNorm2d(self.planes),
                # nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def channel_pool(self, x):
        m_batchsize, C, height, width = x.size()
        # print(f'input : {cosine_distance(x,isAttention=False)}')

        # context modeling
        proj_query = x.view(m_batchsize, C, -1)  # B,C,HW
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # B,HW,C
        energy = torch.bmm(proj_query, proj_key)  # B,C,C
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy)  # B,C,C
        # print(f'att : {cosine_distance(attention)}')
        # pdb.set_trace()
        proj_value = x.view(m_batchsize, C, -1)  # B,C,HW
        out = torch.bmm(attention, proj_value)  # B,C,HW

        context = out.view(m_batchsize, C, height, width)  # B,C,H,W

        return context

    def forward(self, x):
        context = self.channel_pool(x) # B,C,H,W
        out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + self.gamma * channel_add_term
        return out

class ChannelBlock_independent(nn.Module):

    def __init__(self,
                 inplanes,
                 spatial_size,
                 ratio=0.0625,
                 pooling_type='att',
                 fusion_types=('channel_add', )): # 'channel_mul' 도 추가해서 실험해봐야겠는데
        super(ChannelBlock_independent, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.reduction_spatial_size = int(spatial_size * ratio)

        self.gamma = nn.Parameter(torch.zeros(1))
        if pooling_type == 'att':
            # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.softmax = nn.Softmax(dim=3)

        if 'channel_add' in fusion_types:
            # self.channel_add_conv = nn.Sequential(
            #     nn.Linear(spatial_size, self.reduction_spatial_size, bias=False),
            #     # nn.BatchNorm2d(self.reduction_spatial_size),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(self.reduction_spatial_size, spatial_size, bias=False),
            # )
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(1, self.planes, kernel_size=1),
                nn.BatchNorm2d(self.planes),
                # nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, 1, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        # if self.pooling_type == 'att':
        #     kaiming_init(self.conv_mask, mode='fan_in')
        #     self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def channel_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)

        # [N,C,H,W]->[N, C, 1, 1]->[N, 1, 1, C]
        context_mask = self.avg_pool(x).view(batch, 1, 1, channel)
        # [N, 1, 1, C]
        context_mask = self.softmax(context_mask)
        # [N, 1, 1, HW]
        context = torch.matmul(context_mask, input_x)
        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        return context

    def forward(self, x):
        # [N, 1, H, W]
        batch, channel, height, width = x.size()
        context = self.channel_pool(x)

        out = x
        if self.channel_add_conv is not None:
            # [N, 1, H, W] -> [N,C,H,W]
            context = context.expand_as(x)
            channel_add_term = self.channel_add_conv(context)
            # channel_add_term = channel_add_term.view(batch, 1, height, width)
            out = out + self.gamma * channel_add_term
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        # my code
        self.transform = nn.Sequential(
            nn.Conv2d(self.chanel_in, self.chanel_in//4, kernel_size=1),
            nn.BatchNorm2d(self.chanel_in//4),
            # nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.chanel_in//4, self.chanel_in, kernel_size=1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # print(f'input : {cosine_distance(x, isAttention=False)}')
        # cosine_distance(x, isAttention=False, doWrite=True)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy) # B X HW X HW

        cosine_distance(attention, isAttention=True, doWrite=True)
        # print(f'attention : {cosine_distance(attention,isAttention=True,doWrite=True)}')
        # pdb.set_trace()
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.transform(out)
        # cosine_distance(out, isAttention=False,doWrite=True)
        # print(f'output : {cosine_distance(out, isAttention=False)}')
        # pdb.set_trace()
        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

        self.transform = nn.Sequential(
            nn.Conv2d(self.chanel_in, self.chanel_in // 4, kernel_size=1),
            nn.BatchNorm2d(self.chanel_in // 4),
            # nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.chanel_in // 4, self.chanel_in, kernel_size=1))
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        # print(f'input : {cosine_distance(x, isAttention=False)}')
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy)
        # print(f'attention : {cosine_distance(attention,isAttention=True)}')
        # pdb.set_trace()
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        # print(f'output : {cosine_distance(out, isAttention=False)}')
        out = self.transform(out)

        # print(f'output : {cosine_distance(out, isAttention=False)}')
        # pdb.set_trace()
        out = self.gamma*out + x
        return out
