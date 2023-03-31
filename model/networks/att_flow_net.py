import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

''' ##############################################################################
function:   conv3x3_LeakyReLU
            correlation
            backwarp
############################################################################ '''
def conv3x3_LeakyReLU(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def correlation(input1, input2, d=4):
    # naive pytorch implementation of the correlation layer.
    assert (input1.shape == input2.shape)
    _, _, H, W = input1.shape
    input2 = F.pad(input2, (d,d,d,d), value=0)
    cv = []
    for i in range(2 * d + 1):
        for j in range(2 * d + 1):
            cv.append((input1 * input2[:, :, i:(i + H), j:(j + W)]).sum(1).unsqueeze(1))
    return torch.cat(cv, 1)

def flow_attention(similarity, dist=4):
    B, C, H, W = similarity.shape
    nums = dist * 2 + 1
    nums2 = nums * nums
    assert (C == nums2)

    v_x = torch.linspace(-dist, dist, nums).repeat(nums).view(1, nums2, 1, 1).expand(B, -1, H, W).to(similarity.get_device()) #[b nums2, h, w]
    v_y = torch.linspace(-dist, dist, nums).repeat_interleave(nums).view(1, nums2, 1, 1).expand(B, -1, H, W).to(similarity.get_device()) #[b nums2, h, w]
    att_vx = torch.mean(similarity * v_x, dim=1, keepdim=True)
    att_vy = torch.mean(similarity * v_y, dim=1, keepdim=True)
    return torch.cat([att_vx, att_vy], 1)

def meshgrid(size, norm=False):
    b, c, h, w = size
    if norm:
        x = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, -1, h, -1) # [b 1 h w]
        y = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, -1, -1, w) # [b 1 h w]
    else:
        x = torch.arange(0, w).view(1, 1, 1, w).expand(b, -1, h, -1) # [b 1 h w]
        y = torch.arange(0, h).view(1, 1, h, 1).expand(b, -1, -1, w) # [b 1 h w]

    grid = torch.cat([ x, y ], 1) # [b 2 h w]
    return grid

def backwarp(im, flow, useMask=False):
    b, c, h, w = im.shape
    # make grid
    grid = meshgrid(im.shape, norm=True).to(im.get_device())
    assert grid.shape == flow.shape
    
    flow = torch.cat([ 2.0 * flow[:, 0:1, :, :] / (w - 1.0), 2.0 * flow[:, 1:2, :, :] / (h - 1.0)], 1)
    vgrid = (grid + flow).permute(0,2,3,1)
    output = F.grid_sample(im, vgrid, align_corners=True)

    if useMask:
        Mask = torch.autograd.Variable(torch.ones_like(im))
        Mask = F.grid_sample(Mask, vgrid, align_corners=True)
        Mask[Mask > 0.999] = 1.0
        Mask[Mask < 1.0] = 0.0
        output = output * Mask

    return output

''' ##############################################################################
class: 
############################################################################ '''
class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()

        self.feature1 = nn.Sequential(
            conv3x3_LeakyReLU(3,   16, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(16,  16, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(16,  16, kernel_size=3, stride=1)
        )
        self.feature2 = nn.Sequential(
            conv3x3_LeakyReLU(16,  32, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(32,  32, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(32,  32, kernel_size=3, stride=1)
        )
        self.feature3 = nn.Sequential(
            conv3x3_LeakyReLU(32,  64, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(64,  64, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(64,  64, kernel_size=3, stride=1)
        )
        self.feature4 = nn.Sequential(
            conv3x3_LeakyReLU(64,  96, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(96,  96, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(96,  96, kernel_size=3, stride=1)
        )
        self.feature5 = nn.Sequential(
            conv3x3_LeakyReLU(96,  128, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(128, 128, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(128, 128, kernel_size=3, stride=1)
        )
        self.feature6 = nn.Sequential(
            conv3x3_LeakyReLU(128, 196, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(196, 196, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(196, 196, kernel_size=3, stride=1)
        )

    def forward(self, inputs):
        feature1 = self.feature1(inputs)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature4 = self.feature4(feature3)
        feature5 = self.feature5(feature4)
        feature6 = self.feature6(feature5)
        return [ feature1, feature2, feature3, feature4, feature5, feature6 ]

class Refiner(nn.Module):
    def __init__(self, feat_ch=2):
        super(Refiner, self).__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=81 + 32 + 2 + feat_ch + 96 + 96 + 64 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            # torch.nn.Conv2d(in_channels=81 + 32 + 2 + feat_ch + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )

    def forward(self, tenInput):
        return self.netMain(tenInput)

class Decode_One_Level(nn.Module):
    def __init__(self, level, feat_ch=2):
        super(Decode_One_Level, self).__init__()
        self.level = level
        chs_Flow = [ None, None, 81+32+2+feat_ch, 81+64+2+feat_ch, 81+96+2+feat_ch, 81+128+2+feat_ch, 81, None ]
        Previous = chs_Flow[level + 1]
        Current =  chs_Flow[level + 0]

        if level < 6: self.netfeat = nn.Conv2d(Previous + 96 + 96 + 64 + 64 + 32, feat_ch, kernel_size=1, stride=1, padding=0)
        # if level < 6: self.netfeat = nn.Conv2d(Previous + 128 + 128 + 96 + 64 + 32, feat_ch, kernel_size=1, stride=1, padding=0)

        self.netOne = conv3x3_LeakyReLU(Current,                     96, kernel_size=3, stride=1)
        self.netTwo = conv3x3_LeakyReLU(Current + 96,                96, kernel_size=3, stride=1)
        self.netThr = conv3x3_LeakyReLU(Current + 96 + 96,           64,  kernel_size=3, stride=1)
        self.netFou = conv3x3_LeakyReLU(Current + 96 + 96 + 64,      64,  kernel_size=3, stride=1)
        self.netFiv = conv3x3_LeakyReLU(Current + 96 + 96 + 64 + 64, 32,  kernel_size=3, stride=1)
        self.netFLow = nn.Conv2d(Current + 96 + 96 + 64 + 64 + 32, 81, kernel_size=3, stride=1, padding=1)

    def forward(self, tenFirst, tenSecond, objPrevious):
        B, C, H, W = tenFirst.shape

        if objPrevious is None:
            tenFlow = None
            tenVolume = F.leaky_relu(correlation(tenFirst, tenSecond), negative_slope=0.1, inplace=False)
            tenFeat = torch.cat([ tenVolume ], 1)

        elif objPrevious is not None:
            _, _, H_in, W_in = objPrevious['tenFlow'].shape
            scale = ((H / H_in) + (W / W_in)) / 2
            tenFlow = F.interpolate(objPrevious['tenFlow'], size=(H, W), mode='bilinear', align_corners=True) * scale
            tenFeat = self.netfeat(F.interpolate(objPrevious['tenFeat'], size=(H, W), mode='bilinear', align_corners=True))

            tenWarp = backwarp(tenSecond, tenFlow) # * tenMask
            tenVolume = F.leaky_relu(input=correlation(tenFirst, tenWarp), negative_slope=0.1, inplace=False)
            tenFeat = torch.cat([ tenVolume, tenFirst, tenFlow, tenFeat ], 1)

        tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

        current_flow = flow_attention(self.netFLow(tenFeat), dist=4)
        tenFlow = current_flow if objPrevious is None else current_flow + tenFlow
        return {
            'tenFlow': tenFlow,
            'tenFeat': tenFeat
        }

class Att_PWC_Net(nn.Module):
    def __init__(self):
        super(Att_PWC_Net, self).__init__()
        self.netExtractor = Encode()

        self.netTwo = Decode_One_Level(2)
        self.netThr = Decode_One_Level(3)
        self.netFou = Decode_One_Level(4)
        self.netFiv = Decode_One_Level(5)
        self.netSix = Decode_One_Level(6)
        self.netRefiner = Refiner()
    
    def forward(self, tenFirst, tenSecond):
        assert tenFirst.shape == tenSecond.shape
        B,C,H,W = tenFirst.shape

        tenFirst = self.netExtractor(tenFirst)
        tenSecond = self.netExtractor(tenSecond)

        # Flows = []
        objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
        objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
        # Flows.append(objEstimate['tenFlow'])
        flow4 = objEstimate['tenFlow']

        objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
        # Flows.append(objEstimate['tenFlow'])
        flow3 = objEstimate['tenFlow']

        objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
        # Flows.append(objEstimate['tenFlow'])
        flow2 = objEstimate['tenFlow']

        objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)
        # Flows.append(objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat']))
        flow1 = objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])

        # Flows = [F.interpolate(flow * 4.0, [H//s, W//s], mode='bilinear', align_corners=True) for flow, s in zip(Flows, [8,4,2,1])]
        # Flows.reverse()

        return [flow1, flow2, flow3, flow4]

if __name__ == '__main__':
    pass
