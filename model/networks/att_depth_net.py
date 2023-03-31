import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


'''
###################################################################################################
layers
###################################################################################################
'''
def upsample(x, size):
    return F.interpolate(x, size=size, mode="nearest")

def Conv1x1(in_channels, out_channels, bias=False):
    return nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1, bias=bias)

def Conv3x3(in_channels, out_channels, use_refl=True):
    if use_refl is True:
        module = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(int(in_channels), int(out_channels), 3))
    else:
        module = nn.Sequential(nn.ZeroPad2d(1), nn.Conv2d(int(in_channels), int(out_channels), 3))
    return module

class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, inputs, flow, size):
        out_h, out_w = size
        n, c, h, w = inputs.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(inputs, grid, mode='nearest', align_corners=True)
        return output

class GridNet(nn.Module):
    def __init__(self):
        super(GridNet, self).__init__()
        # self.conv1 = nn.Sequential(Conv3x3(2, 16), nn.ELU())
        # self.conv2 = nn.Sequential(Conv3x3(16, 32), nn.ELU())

    def forward(self, feat):
        device = feat.get_device()
        x = self.get_grid(feat.shape).to(device)
        # x = self.conv1(x)
        # x = self.conv2(x)
        return x

    def get_grid(self, size):
        b, _, h, w = size
        x = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, -1, h, -1) # [b 1 h w]
        y = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, -1, -1, w) # [b 1 h w]
        grid = torch.cat([ x, y ], 1) # [b 2 h w]
        return grid

# def get_grid(img):
#     b, _, h, w = img.shape
#     x = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, -1, h, -1) # [b 1 h w]
#     y = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, -1, -1, w) # [b 1 h w]
#     grid = torch.cat([ x, y ], 1) # [b 2 h w]
#     return grid

'''
###################################################################################################
encode decode
###################################################################################################
'''
class DepthEncoder(nn.Module):
    def __init__(self, num_layers, pretrained=True):
        super(DepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101
                }

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))


        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        self.features.append(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x))))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class DepthDecoder(nn.Module):
    def __init__(self,  num_ch_enc):
        super(DepthDecoder, self).__init__()
        self.num_ch_dec = [64, 96, 128, 128, 256]

        self.pre4 = nn.Sequential(Conv3x3(num_ch_enc[4], self.num_ch_dec[4]), nn.ELU())
        self.pre3 = nn.Sequential(Conv3x3(num_ch_enc[3], self.num_ch_dec[3]), nn.ELU())
        self.pre2 = nn.Sequential(Conv3x3(num_ch_enc[2], self.num_ch_dec[2]), nn.ELU())
        self.pre1 = nn.Sequential(Conv3x3(num_ch_enc[1], self.num_ch_dec[1]), nn.ELU())
        self.pre0 = nn.Sequential(Conv3x3(num_ch_enc[0], self.num_ch_dec[0]), nn.ELU())

        self.iconv4 = nn.Sequential(Conv3x3(self.num_ch_dec[4],        self.num_ch_dec[4]), nn.ELU())
        self.iconv3 = nn.Sequential(Conv3x3(sum(self.num_ch_dec[3:5]), self.num_ch_dec[3]), nn.ELU())
        self.iconv2 = nn.Sequential(Conv3x3(sum(self.num_ch_dec[2:4]), self.num_ch_dec[2]), nn.ELU())
        self.iconv1 = nn.Sequential(Conv3x3(sum(self.num_ch_dec[1:3]), self.num_ch_dec[1]), nn.ELU())
        self.iconv0 = nn.Sequential(Conv3x3(sum(self.num_ch_dec[0:2]), self.num_ch_dec[0]), nn.ELU())


    def forward(self, input_features):
        l0, l1, l2, l3, l4 = input_features

        x4 = self.pre4(l4)
        x4 = self.iconv4(x4)
        x4_up = upsample(x4, size=l3.shape[2:4])
        
        x3 = self.pre3(l3)
        x3 = self.iconv3(torch.cat([x3, x4_up], 1))
        x3_up = upsample(x3, size=l2.shape[2:4])

        x2 = self.pre2(l2)
        x2 = self.iconv2(torch.cat([x2, x3_up], 1))
        x2_up = upsample(x2, size=l1.shape[2:4])

        x1 = self.pre1(l1)
        x1 = self.iconv1(torch.cat([x1, x2_up], 1))
        x1_up = upsample(x1, size=l0.shape[2:4])

        x0 = self.pre0(l0)
        x0 = self.iconv0(torch.cat([x0, x1_up], 1))
        return x0, x1, x2, x3

class DepthAware(nn.Module):
    def __init__(self, cfgs, in_channels, out_channels):
        super(DepthAware, self).__init__()
        self.out_channels = out_channels

        min_disp = 1 / cfgs['max_depth']
        max_disp = 1 / cfgs['min_depth']
        planes = torch.arange(out_channels + 1)  / out_channels
        self.samples = (torch.log(torch.tensor(max_disp / min_disp)) * (planes)).exp() * min_disp
        self.mu = self.samples[0:out_channels]
        self.sigma = self.samples[1:out_channels+1] - self.samples[0:out_channels]

        self.reduce1 = nn.Sequential(nn.Conv2d(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())
        self.reduce2 = nn.Sequential(nn.Conv2d(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())
        self.reduce3 = nn.Sequential(nn.Conv2d(in_channels[2], out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())
        self.reduce4 = nn.Sequential(nn.Conv2d(in_channels[3], out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())

        self.med0 = nn.Sequential(nn.Conv2d(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0), nn.Softmax(1))
        self.med1 = nn.Sequential(nn.Conv2d(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0), nn.Softmax(1))
        self.med2 = nn.Sequential(nn.Conv2d(in_channels[2], out_channels, kernel_size=1, stride=1, padding=0), nn.Softmax(1))
        self.med3 = nn.Sequential(nn.Conv2d(in_channels[3], out_channels, kernel_size=1, stride=1, padding=0), nn.Softmax(1))

    def forward(self, inputs, frame_idx):
        l0, l1, l2, l3 = inputs
        batch = l0.shape[0]
        device = l0.get_device()

        mu = self.mu.view(1, self.out_channels, 1, 1).expand(batch, self.out_channels, 1, 1).to(device)
        sigma = self.sigma.view(1, self.out_channels, 1, 1).expand(batch, self.out_channels, 1, 1).to(device)

        # 
        m0 = self.med0(l0)
        m1 = self.med1(l1)
        m2 = self.med2(l2)
        m3 = self.med3(l3)

        disp0 = torch.sum( (m0 * mu), dim=1, keepdim=True)
        disp1 = torch.sum( (m1 * mu), dim=1, keepdim=True)
        disp2 = torch.sum( (m2 * mu), dim=1, keepdim=True)
        disp3 = torch.sum( (m3 * mu), dim=1, keepdim=True)

        outputs = {}
        outputs[('disp', frame_idx, 0)] = disp0
        outputs[('disp', frame_idx, 1)] = disp1
        outputs[('disp', frame_idx, 2)] = disp2
        outputs[('disp', frame_idx, 3)] = disp3
        return outputs

class Depth_Net(nn.Module):
    def __init__(self, cfgs, num_layers=18):
        super(Depth_Net, self).__init__()
        self.encoder = DepthEncoder(num_layers=num_layers, pretrained=True)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)
        self.get_disp = DepthAware(cfgs, self.decoder.num_ch_dec, out_channels=48)

    def forward(self, img, frame_idx=0):
        enc_feat = self.encoder(img)
        dec_feat = self.decoder(enc_feat)
        outputs = self.get_disp(dec_feat, frame_idx)
        return outputs

if __name__ == '__main__':
    net = Depth_Net(depth_scale = 4,num_layers=18)
    img = torch.ones(1, 3, 128, 128)
    out = net(img)
    print(out)
