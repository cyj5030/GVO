import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from model.networks.att_flow_net import Att_PWC_Net, backwarp
import model.utils as utils

class Flow_Model(nn.Module):
    # PWC or AttPWC
    def __init__(self, cfgs):
        super(Flow_Model, self).__init__()
        self.cfgs = cfgs
        # net
        self.flow_net = Att_PWC_Net()
        # self.flow_net = PWC_Net()

        # print number of parameters
        self.n_params = [sum([p.data.nelement() for p in self.flow_net.parameters()])]
        print('Number of params in flow_net: {}'.format(*self.n_params))

    @staticmethod
    def consistent(flow_fw, flow_bw):
        def norm_flow(x):
            return torch.sum(x.pow(2.0), dim=1, keepdim=True)

        flow_fw_warped = backwarp(flow_bw, flow_fw)
        flow_bw_warped = backwarp(flow_fw, flow_bw)
        flow_diff_fw = torch.abs(flow_fw + flow_fw_warped)
        flow_diff_bw = torch.abs(flow_bw + flow_bw_warped)

        norm_fw = norm_flow(flow_fw) + norm_flow(flow_fw_warped)
        norm_bw = norm_flow(flow_bw) + norm_flow(flow_bw_warped)
        # if self.cfgs['dataset'] == 'tmu':
        #     beta = 0.05 
        #  0.01 kitti
        thresh_fw = torch.max( torch.tensor([3.0], device=flow_fw.get_device()), 0.01 * norm_fw + 0.5)
        thresh_bw = torch.max( torch.tensor([3.0], device=flow_bw.get_device()), 0.01 * norm_bw + 0.5)

        with torch.no_grad():
            mask_fw = (norm_flow(flow_diff_fw) < thresh_fw).float()
            mask_bw = (norm_flow(flow_diff_bw) < thresh_bw).float()
        return mask_fw, mask_bw, norm_flow(flow_diff_fw), norm_flow(flow_diff_bw)
    
    @staticmethod
    def consistent_multiscale(flows, flows_inv):
        masks_fw, masks_bw, consistents_fw, consistents_bw = [], [], [], []
        for fw, bw in zip(flows, flows_inv):
            m_fw, m_bw, c_fw, c_bw = Flow_Model.consistent(fw, bw)
            masks_fw.append(m_fw)
            masks_bw.append(m_bw)
            consistents_fw.append(c_fw)
            consistents_bw.append(c_bw)
        return masks_fw, masks_bw, consistents_fw, consistents_bw

    @staticmethod
    def occ_mask(flow):
        mask = torch.ones_like(flow[:, 0:1, :, :])
        _, _, H, W = mask.shape
        occ_mask = utils.transformerFwd(mask.permute(0,2,3,1), flow.permute(0,2,3,1), out_size=[H,W]) #[b,h,w,c]
        with torch.no_grad(): 
            occ_mask = torch.clamp(occ_mask, 0.0, 1.0)
        return occ_mask.permute(0,3,1,2) #[b,c,h,w]

    @staticmethod
    def occ_mask_multiscale(flows):
        B, _, H, W = flows[0].shape
        masks = []
        for flow in flows:
            masks.append(Flow_Model.occ_mask(flow))
        return masks

    @staticmethod
    def image_pyramid(image, num_scale):
        _, _, H, W = image.shape
        return [F.interpolate(image, [H//(2**s), W//(2**s)], mode='bilinear', align_corners=False) for s in range(num_scale)]

    @staticmethod
    def warped_images(images, flows):
        return [backwarp(image, flow) for image, flow in zip(images, flows)]

    def loss_L1(self, pyramid, warped_pyramid, occ_masks): 
        if occ_masks is None:
            occ_masks = [torch.ones_like(im) for im in pyramid]

        loss_list = []
        for image, warped_image, occ_mask in zip(pyramid, warped_pyramid, occ_masks):
            error_map = torch.abs(image - warped_image) * occ_mask
            # error_map = torch.pow(torch.abs((image - warped_image)) + 0.01, 0.6) * occ_mask
            # norm
            error_map = error_map.mean((1,2,3)) / (occ_mask.mean((1,2,3)) + 1e-12) # (B)
            loss_list.append(error_map[:,None])

        return torch.cat(loss_list, 1).sum(1) # (B)

    def loss_ssim(self, pyramid, warped_pyramid, occ_masks):
        if occ_masks is None:
            occ_masks = [torch.ones_like(im) for im in pyramid]

        loss_list = []
        for image, warped_image, occ_mask in zip(pyramid, warped_pyramid, occ_masks):
            error_map = torch.clamp((1.0 - utils.SSIM(image * occ_mask, warped_image * occ_mask)) / 2.0, 0, 1)
            # norm
            error_map = error_map.mean((1,2,3)) / (occ_mask.mean((1,2,3)) + 1e-12)
            loss_list.append(error_map[:,None])

        return torch.cat(loss_list, 1).sum(1) # (B)

    @staticmethod
    def gradients(ten):
        dy = ten[:,:,1:,:] - ten[:,:,:-1,:]
        dx = ten[:,:,:,1:] - ten[:,:,:,:-1]
        return dx, dy

    def grad2_error(self, image, tenObj):
        w = 10.0
        image_dx, image_dy = self.gradients(image)
        w_x = torch.exp(-w * torch.abs(image_dx).mean(1).unsqueeze(1))
        w_y = torch.exp(-w * torch.abs(image_dy).mean(1).unsqueeze(1))

        dx, dy = self.gradients(tenObj)
        ddx, _ = self.gradients(dx)
        _, ddy = self.gradients(dy)
        error = (w_x[:,:,:,1:] * torch.abs(ddx))[:,:,2:,:] + (w_y[:,:,1:,:] * torch.abs(ddy))[:,:,:,2:]
        return error / 2.0

    def loss_smooth(self, pyramid, flows):
        loss_list = []
        for image, flow in zip(pyramid, flows):
            B, _, H, W = flow.shape
            image = F.interpolate(image, (H, W), mode='area')
            error_map = self.grad2_error(image, flow / 20.0)
            error = error_map.mean((1, 2, 3))
            loss_list.append(error[:,None])

        return torch.cat(loss_list, 1).sum(1)

    def infer_flow(self, tenFirst, tenSecond):
        B, _, H, W = tenFirst.shape

        # optical flow
        flows     = self.flow_net(tenFirst, tenSecond)
        flows = [F.interpolate(flows[0] * 4.0, [H, W], mode='bilinear', align_corners=True)]
        # flows_inv = self.flow_net(tenSecond, tenFirst)

        # flows = [F.interpolate(flow * 4.0, [H//s, W//s], mode='bilinear', align_corners=True) for flow, s in zip(flows, [1,2,4,8])]
        # flows_inv = [F.interpolate(flow * 4.0, [H//s, W//s], mode='bilinear', align_corners=True) for flow, s in zip(flows_inv, [1,2,4,8])]

        # image pyramid for multiscale training
        # pyramid_first  = self.image_pyramid(tenFirst, len(flows))
        # pyramid_second = self.image_pyramid(tenSecond, len(flows_inv))

        # warped_pyramid_first = self.warped_images(pyramid_second, flows)
        # warped_pyramid_second = self.warped_images(pyramid_first, flows_inv)

        # computer consis mask and backwarp mask 
        # consis_mask_fw, consis_mask_bw, consis_fw, consis_bw = self.consistent_multiscale(flows, flows_inv)
        # backwarp_mask_fw = self.occ_mask_multiscale(flows_inv)
        # backwarp_mask_bw = self.occ_mask_multiscale(flows)

        # mask_fw = [cm * bm for cm, bm in zip(consis_mask_fw, backwarp_mask_fw)]
        # mask_bw = [cm * bm for cm, bm in zip(consis_mask_bw, backwarp_mask_bw)]

        extra_data = {}
        # extra_data['mask'] = mask_fw
        # extra_data['warp_image'] = warped_pyramid_first
        return flows, extra_data

    def forward(self, inputs):
        tenFirst = inputs["color", 0]
        tenSecond = inputs["color", 1]
        B, _, H, W = tenFirst.shape

        # optical flow
        flows     = self.flow_net(tenFirst, tenSecond)
        flows_inv = self.flow_net(tenSecond, tenFirst)

        # flow interpolate
        flows_rescale = [F.interpolate(flow * 4.0, [H//s, W//s], mode='bilinear', align_corners=True) for flow, s in zip(flows, [1,2,4,8])]
        flows_inv_rescale = [F.interpolate(flow * 4.0, [H//s, W//s], mode='bilinear', align_corners=True) for flow, s in zip(flows_inv, [1,2,4,8])]
        
        # image pyramid for multiscale training
        pyramid_first  = self.image_pyramid(tenFirst, len(flows))
        pyramid_second = self.image_pyramid(tenSecond, len(flows_inv))

        warped_pyramid_first = self.warped_images(pyramid_second, flows_rescale)
        warped_pyramid_second = self.warped_images(pyramid_first, flows_inv_rescale)

        # computer consis mask and backwarp mask 
        consis_mask_fw, consis_mask_bw, consis_fw, consis_bw = self.consistent_multiscale(flows_rescale, flows_inv_rescale)
        backwarp_mask_fw = self.occ_mask_multiscale(flows_inv_rescale)
        backwarp_mask_bw = self.occ_mask_multiscale(flows_rescale)

        mask_fw = [cm * bm for cm, bm in zip(consis_mask_fw, backwarp_mask_fw)]
        mask_bw = [cm * bm for cm, bm in zip(consis_mask_bw, backwarp_mask_bw)]

        # compute loss 
        L1_first = self.loss_L1(pyramid_first, warped_pyramid_first, mask_fw)
        L1_second = self.loss_L1(pyramid_second, warped_pyramid_second, mask_bw)

        ssim_first = self.loss_ssim(pyramid_first, warped_pyramid_first, mask_fw)
        ssim_second = self.loss_ssim(pyramid_second, warped_pyramid_second, mask_bw)

        flow_smooth = self.loss_smooth(pyramid_first, flows)
        flow_inv_smooth = self.loss_smooth(pyramid_second, flows_inv)

        losses_pack = collections.OrderedDict()
        losses_pack['F_rec'] = self.cfgs['w_F_rec'] * (0.15 * (L1_first + L1_second).mean() + 0.85 * (ssim_first + ssim_second).mean())
        losses_pack['F_smth'] = self.cfgs['w_F_smth'] * (flow_smooth + flow_inv_smooth).mean()

        total_loss = sum(_value for _key, _value in losses_pack.items())

        if self.cfgs['model'] == 'flow':
            return losses_pack, total_loss
        else:
            return losses_pack, flows_rescale[0], flows_inv_rescale[0], mask_fw[0], mask_bw[0], consis_fw[0], consis_bw[0]


if __name__ == "__main__":
    pass
