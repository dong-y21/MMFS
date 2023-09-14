import torch
import torch.nn as nn
from models.modules.pix2pixMini_module import *

AVG = AvgQuant()
mul = SliceMul()

def get_int(channel_dict):
    for i in range(len(channel_dict['down'])):
        channel_dict['down'][i] = int(channel_dict['down'][i])
    for i in range(len(channel_dict['backbone'])):
        channel_dict['backbone'][i] = int(channel_dict['backbone'][i])
    for i in range(len(channel_dict['up'])):
        for j in range(len(channel_dict['up'][i])):
            if channel_dict['up'][i][j] is not None:
                channel_dict['up'][i][j] = int(channel_dict['up'][i][j])
    return channel_dict

def get_channel_dict(dict_name, ngf):
    if dict_name is None:
        raise('invalid channel_dict name')

    if dict_name == '3G':
        channel_dict = {
            'n_blocks': 8,
            'down': [ngf * 1, ngf * 2, ngf * 4, ngf * 8, ngf * 8],
            'backbone': [ngf * 8, ngf * 8],
            'hair_up': [
                [None, None, None, ngf * 4, 2],
                [None, None, None, ngf * 2, 2],
                [None, None, None, ngf * 1, 2],
                [None, None, None, ngf * 1, 1],
            ],
            'face_up': [
                [None, None, None, ngf * 4, 2],
                [None, None, None, ngf * 2, 2],
                [None, None, None, ngf * 2, 2],
                [None, None, None, ngf * 1, 1],
            ],
            'up': [
                [None, None, None, ngf * 1, 1],
            ],
        }

    elif dict_name == '470M':
        channel_dict = {
            'n_blocks': 4,
            'down': [ngf * 1, ngf * 2, ngf * 3, ngf * 6, ngf * 4],
            'backbone': [ngf * 4, ngf * 4],
            'hair_up': [
                [None, None, None, ngf * 2, 1],
                [None, None, None, ngf * 2, 1],
                [None, None, None, ngf * 1, 1],
                [None, None, None, ngf * 1, 1],
            ],
            'face_up': [
                [None, None, ngf * 4, ngf * 2, 1],
                [None, None, ngf * 2, ngf * 2, 1],
                [None, None, ngf * 2, ngf * 1, 1],
                [None, None, None, ngf * 1, 1],
            ],
            'up': [
                [None, None, ngf * 2, ngf * 1, 1],
            ],
        }
    else:
        raise('invalid_dict_name')
    return get_int(channel_dict)

chans = 8
in_channels = 3
out_channels = 4
face_branch_out_channels = 3
hair_branch_out_channels = 3

def conv(numIn, numOut, k, s=1, p=0, relu=True, bn=False):
    layers = []
    layers.append(Conv2dQuant(numIn, numOut, k, s, p, bias=True))
    if bn:
        layers.append(nn.BatchNorm2d(numOut))
        
    if relu is True:
        layers.append(HardQuant(0, 4))
    return nn.Sequential(*layers)

def mnconv(numIn, numOut, k, s=1, p=0, dilation=1, relu=True, bn = True):
    if k < 2:
        return conv(numIn, numOut, k, s, p, relu, bn)
    layers = []
    layers.append(Conv2dQuant(numIn, numIn, k, s, p, groups=numIn, dilation=dilation, bias=True))
    layers.append(nn.BatchNorm2d(numIn))
    layers.append(HardQuant(0, 4))
    layers.append(conv(numIn, numOut, 1, 1, 0, relu, bn))
    return nn.Sequential(*layers)

class hair_face_model(nn.Module):
    def __init__(self, ngf=chans, backbone_type='resnet', use_se=True, channel_dict_name = None, with_hair_branch=False, design=5):
        
        super().__init__()
        self.design = design
        self.with_hair_branch = with_hair_branch
        channel_dict = get_channel_dict(channel_dict_name, ngf)
        n_blocks = channel_dict['n_blocks']

        self.inconv = ConvBlock(in_channels, channel_dict['down'][0], stride=1)
        self.shortcut_ratio = [1,1,1,1]

        # Down-Sampling
        self.DownBlock1 = ConvBlock(channel_dict['down'][0], channel_dict['down'][1], stride=2)
        self.DownBlock2 = ConvBlock(channel_dict['down'][1], channel_dict['down'][2], stride=2)
        self.DownBlock3 = ConvBlock(channel_dict['down'][2], channel_dict['down'][3], stride=2)
        self.DownBlock4 = ConvBlock(channel_dict['down'][3], channel_dict['down'][4], stride=2)

        # Down-Sampling Bottleneck
        if backbone_type == 'resnet':
            backbone_block = ResnetBlock
        elif backbone_type == 'mobilenet':
            backbone_block = InvertedBottleneck
            n_blocks = n_blocks
        else:
            raise('invalid backbone type')
        ResBlock = []
        ResBlock += [backbone_block(channel_dict['down'][4], channel_dict['backbone'][0], use_bias=False, use_se=use_se)]
        for i in range(1, n_blocks - 1):
            ResBlock += [backbone_block(channel_dict['backbone'][0], channel_dict['backbone'][0], use_bias=False, use_se=use_se)]
        ResBlock += [backbone_block(channel_dict['backbone'][0], channel_dict['backbone'][1], use_bias=False, use_se=use_se)]
        self.ResBlock = nn.Sequential(*ResBlock)

        self.HairUpBlock4 = UpBlock(channel_dict['backbone'][1],    None, channel_dict['hair_up'][0][0], None, channel_dict['hair_up'][0][2], channel_dict['hair_up'][0][3], num_conv=channel_dict['hair_up'][0][4])
        self.HairUpBlock3 = UpBlock(channel_dict['hair_up'][0][3],  None, channel_dict['hair_up'][1][0], None, channel_dict['hair_up'][1][2], channel_dict['hair_up'][1][3], num_conv=channel_dict['hair_up'][1][4])
        self.HairUpBlock2 = UpBlock(channel_dict['hair_up'][1][3],  None, channel_dict['hair_up'][2][0], None, channel_dict['hair_up'][2][2], channel_dict['hair_up'][2][3], num_conv=channel_dict['hair_up'][2][4])

        self.FaceUpBlock4 = UpBlock(channel_dict['backbone'][1],    channel_dict['down'][3], channel_dict['face_up'][0][0], channel_dict['face_up'][0][1], channel_dict['face_up'][0][2], channel_dict['face_up'][0][3], num_conv=channel_dict['face_up'][0][4])
        self.FaceUpBlock3 = UpBlock(channel_dict['face_up'][0][3],  channel_dict['down'][2], channel_dict['face_up'][1][0], channel_dict['face_up'][1][1], channel_dict['face_up'][1][2], channel_dict['face_up'][1][3], num_conv=channel_dict['face_up'][1][4])
        self.FaceUpBlock2 = UpBlock(channel_dict['face_up'][1][3],  channel_dict['down'][1], channel_dict['face_up'][2][0], channel_dict['face_up'][2][1], channel_dict['face_up'][2][2], channel_dict['face_up'][2][3], num_conv=channel_dict['face_up'][2][4])

        self.UpBlock1 = mnUpBlock(channel_dict['hair_up'][2][3] + channel_dict['face_up'][2][3],     channel_dict['down'][0], channel_dict['up'][0][0],     channel_dict['up'][0][1], channel_dict['up'][0][2], channel_dict['up'][0][3], num_conv=channel_dict['up'][0][4])
        self.outconv = mnConvOutBlock(channel_dict['up'][0][3], out_channels)

        #self.shortcut_ratio = [1,1,1,1]

        if self.with_hair_branch:
            self.HairUpBlock1 = UpBlock(channel_dict['hair_up'][2][3], None, channel_dict['hair_up'][3][0], None, channel_dict['hair_up'][3][2],channel_dict['hair_up'][3][3])
            self.Hairoutconv = ConvOutBlock(channel_dict['hair_up'][3][3], hair_branch_out_channels)

            self.FaceUpBlock1 = UpBlock(channel_dict['face_up'][2][3], channel_dict['down'][0], channel_dict['face_up'][3][0], channel_dict['face_up'][3][1], channel_dict['face_up'][3][2], channel_dict['face_up'][3][3])
            self.Faceoutconv = ConvOutBlock(channel_dict['face_up'][3][3], face_branch_out_channels)

        self.up = UpsampleQuant(scale_factor=1.5, mode='bilinear')


    def forward(self, x , with_hair=False):
        x0 = self.inconv(x)
        x1 = self.DownBlock1(x0)
        x2 = self.DownBlock2(x1)
        x3 = self.DownBlock3(x2)
        x4 = self.DownBlock4(x3)
        x = self.ResBlock(x4)

        hair = self.HairUpBlock4(x)
        hair = self.HairUpBlock3(hair)
        hair = self.HairUpBlock2(hair)
        face = self.FaceUpBlock4(x, x3, self.shortcut_ratio[0])
        face = self.FaceUpBlock3(face, x2, self.shortcut_ratio[1])
        face = self.FaceUpBlock2(face, x1, self.shortcut_ratio[2])

        x = self.UpBlock1(torch.cat([hair,face], dim=1), x0, self.shortcut_ratio[3])
#         print(self.outconv)
        x = self.outconv(x)
        if not with_hair or not self.with_hair_branch:
            return x
        else:
            hair = self.HairUpBlock1(hair)
            hair = self.Hairoutconv(hair)

            face = self.FaceUpBlock1(face, x0, self.shortcut_ratio[3])
            face = self.Faceoutconv(face)

            return x, hair, face


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            Conv2dQuant(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            HardQuant(0, 4)
            #nn.ReLU(False))
        )
    def forward(self, x):
        x = self.conv(x)
        return x



class UpBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, mid_ch1, mid_ch2, mid_ch, out_ch, num_conv=1, use_bn=True):
        super(UpBlock, self).__init__()

        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = UpsampleQuant(scale_factor=2, mode='nearest')

        ## branch_1
        if mid_ch1 is None or in_ch1 == mid_ch1:
            self.conv1 = None
        else:
            self.conv1 = nn.Sequential(
                #nn.Conv2d(in_ch1, mid_ch1, 1, bias=False),
                Conv2dQuant(in_ch1, mid_ch1, 1, bias=True),
                #nn.ReLU(False),
                HardQuant(0, 4)
            )
        if mid_ch1 is None:
            mid_ch1 = in_ch1

        if in_ch2 is None:
            self.use_shortcut = False
            self.conv2 = None
        else:
            self.use_shortcut = True
            if mid_ch2 is None or in_ch2 == mid_ch2:
                self.conv2 = None
            else:
                self.conv2 = nn.Sequential(
                    #nn.Conv2d(in_ch2, mid_ch2, 1, bias=False),
                    Conv2dQuant(in_ch2, mid_ch2, 1, bias=True),
                    #nn.ReLU(False),
                    HardQuant(0, 4)
                )
            if mid_ch2 is None:
                mid_ch2 = in_ch2
        #print(self.conv1 is None, self.conv2 is None)
        combine_ch = mid_ch1
        if self.use_shortcut:
            combine_ch = combine_ch + mid_ch2
        if mid_ch is None or combine_ch == mid_ch:
            self.conv_combine = None
            mid_ch = combine_ch
        else:
            self.conv_combine = nn.Sequential(
                #nn.Conv2d(combine_ch, mid_ch, 1, bias=False),
                Conv2dQuant(combine_ch, mid_ch, 1, bias=True),
                #nn.ReLU(False),
                HardQuant(0, 4)
            )

        conv_list = []
        #conv_list.append(nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
        conv_list.append(Conv2dQuant(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True))
#         conv_list.append(mnconv(mid_ch, out_ch, k=3, s=1, p=1))

        if use_bn:
            conv_list.append(nn.BatchNorm2d(out_ch))
        #conv_list.append(nn.ReLU(False))
        conv_list.append(HardQuant(0, 4))
        for n in range(1, num_conv):
            conv_list.append(ResnetBlock(out_ch, out_ch, use_bias=False, use_se = False, use_bn=use_bn))
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x1, x2=None, ratio=None):

        if self.conv1 is not None:
            x1 = self.conv1(x1)
        x1 = self.up(x1)

        if self.use_shortcut:
            if self.conv2 is not None:
                x2 = self.conv2(x2)

        if self.use_shortcut:
            if ratio is None:
                x = torch.cat([x1, x2], dim=1)
            else:
                x = torch.cat([x1, x2], dim=1)
        else:
            x = x1

        if self.conv_combine is not None:
            x = self.conv_combine(x)

        x = self.conv(x)
        return x

class mnUpBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, mid_ch1, mid_ch2, mid_ch, out_ch, num_conv=1, use_bn=True):
        super(mnUpBlock, self).__init__()

        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = UpsampleQuant(scale_factor=2, mode='nearest')

        ## branch_1
        if mid_ch1 is None or in_ch1 == mid_ch1:
            self.conv1 = None
        else:
            self.conv1 = nn.Sequential(
                #nn.Conv2d(in_ch1, mid_ch1, 1, bias=False),
                Conv2dQuant(in_ch1, mid_ch1, 1, bias=True),
                #nn.ReLU(False),
                HardQuant(0, 4)
            )
        if mid_ch1 is None:
            mid_ch1 = in_ch1

        if in_ch2 is None:
            self.use_shortcut = False
            self.conv2 = None
        else:
            self.use_shortcut = True
            if mid_ch2 is None or in_ch2 == mid_ch2:
                self.conv2 = None
            else:
                self.conv2 = nn.Sequential(
                    #nn.Conv2d(in_ch2, mid_ch2, 1, bias=False),
                    Conv2dQuant(in_ch2, mid_ch2, 1, bias=True),
                    #nn.ReLU(False),
                    HardQuant(0, 4)
                )
            if mid_ch2 is None:
                mid_ch2 = in_ch2
        #print(self.conv1 is None, self.conv2 is None)
        combine_ch = mid_ch1
        if self.use_shortcut:
            combine_ch = combine_ch + mid_ch2
        if mid_ch is None or combine_ch == mid_ch:
            self.conv_combine = None
            mid_ch = combine_ch
        else:
            self.conv_combine = nn.Sequential(
                #nn.Conv2d(combine_ch, mid_ch, 1, bias=False),
                Conv2dQuant(combine_ch, mid_ch, 1, bias=True),
                #nn.ReLU(False),
                HardQuant(0, 4)
            )

        conv_list = []
        #conv_list.append(nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
        # conv_list.append(Conv2dQuant(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True))
        conv_list.append(Conv2dQuant(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, groups=mid_ch,bias=True))
        conv_list.append(nn.BatchNorm2d(mid_ch))
        conv_list.append(HardQuant(0, 4))
        conv_list.append(Conv2dQuant(mid_ch, out_ch, kernel_size=1,stride=1,padding=0))
        # conv_list.append(mnconv(mid_ch, out_ch, k=3, s=1, p=1))

        if use_bn:
            conv_list.append(nn.BatchNorm2d(out_ch))
        #conv_list.append(nn.ReLU(False))
        conv_list.append(HardQuant(0, 4))
        for n in range(1, num_conv):
            conv_list.append(ResnetBlock(out_ch, out_ch, use_bias=False, use_se = False, use_bn=use_bn))
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x1, x2=None, ratio=None):

        if self.conv1 is not None:
            x1 = self.conv1(x1)
        x1 = self.up(x1)

        if self.use_shortcut:
            if self.conv2 is not None:
                x2 = self.conv2(x2)

        if self.use_shortcut:
            if ratio is None:
                x = torch.cat([x1, x2], dim=1)
            else:
                x = torch.cat([x1, x2], dim=1)
        else:
            x = x1

        if self.conv_combine is not None:
            x = self.conv_combine(x)

        x = self.conv(x)
        return x

class mnConvOutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(mnConvOutBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv2dQuant(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=True),
            nn.BatchNorm2d(in_ch),
            HardQuant(0, 4),
            Conv2dQuant(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.Tanh()
            TanhOp(data_in_type='float', data_out_type='fixed'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )
    def forward(self, x0):
        x0 = self.conv(x0)
        return x0

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, use_bias, use_se = False, use_bn=True):
        super(ResnetBlock, self).__init__()
        conv_block = []
        #conv_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        conv_block += [Conv2dQuant(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),]
        if use_bn:
            conv_block += [nn.BatchNorm2d(dim),]
        #conv_block += [nn.ReLU(False)]
        conv_block += [HardQuant(0, 4)]

        #conv_block.append(nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1, bias=use_bias))
        conv_block.append(Conv2dQuant(dim, dim_out, kernel_size=3, stride=1, padding=1, bias=True))
        if use_bn:
            conv_block.append(nn.BatchNorm2d(dim_out))
        conv_block += [HardQuant(0, 4)]

        if use_se:
            conv_block.append(SqEx(dim_out, 4))

        self.conv_block = nn.Sequential(*conv_block)

        self.downsample = None
        if dim != dim_out:
            if use_bn:
                self.downsample = nn.Sequential(
                    #nn.Conv2d(dim, dim_out, kernel_size=1, stride=1, bias=use_bias),
                    #nn.BatchNorm2d(dim_out),
                    Conv2dQuant(dim, dim_out, kernel_size=1, stride=1, bias=True),
                    nn.BatchNorm2d(dim_out),
                )
            else:
                self.downsample = nn.Sequential(
                    #nn.Conv2d(dim, dim_out, kernel_size=1, stride=1, bias=use_bias),
                    Conv2dQuant(dim, dim_out, kernel_size=1, stride=1, bias=True),
                )

        #self.relu = nn.ReLU(False)
        #self.relu = HardQuant(0, 4)

    def forward(self, x):
        if self.downsample is None:
            y = AVG(x, self.conv_block(x))
        else:
            y = AVG(self.downsample(x), self.conv_block(x))
        #y = self.relu(y)
        return y
