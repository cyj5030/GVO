import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np

import model.utils as utils
from model.flow_model import Flow_Model
from model.networks.ba_net import Geometric_BA
# from model.networks.depth_net import Depth_Net
from model.networks.att_depth_net import Depth_Net
from model.networks.samples import MovingMask, ray_angle_filter
from model.networks.pnp import PnPPose

class Depth_Model(nn.Module):
    def __init__(self, cfgs):
        super(Depth_Model, self).__init__()

        # prameters
        self.cfgs = cfgs

        self.image_W = cfgs['width']
        self.image_H = cfgs['height']
        self.max_depth = cfgs['max_depth']
        self.min_depth = cfgs['min_depth']

        self.UseStero = cfgs['use_stero']
        self.UseAutoMask = False if cfgs['use_stero'] else True
        self.NormDisp = False if cfgs['use_stero'] else True
        self.UseBackLoss = True if cfgs['train_length'] == 2 else False
        self.depth_scales = cfgs['depth_scales']
        self.resample = True # if self.cfgs['dataset'] == 'kitti' else False

        self.num_kpts = cfgs['num_keypoints']
        self.ratio_kpts = cfgs['ratio_keypoints']

        self.train_length = cfgs['train_length']
        self.frame_idx = np.arange(2) if self.train_length == 2 else np.arange(self.train_length) - 1
        self.src_idx = list(np.delete(self.frame_idx, np.where(self.frame_idx == 0)[0][0]))
        if self.UseStero:
            self.src_idx.append('stereo')

        # model 
        self.flow_model = Flow_Model(cfgs)
        self.depth_net = Depth_Net(cfgs)
        self.ba_net = Geometric_BA(cfgs)

        self.movingMask = MovingMask(cfgs)
        self.init_pnp = PnPPose(cfgs)

        self.grid2d = utils.meshgrid((cfgs['batchsize'], 1, self.image_H, self.image_W))
        # print number of parameters
        self.n_params = [sum([p.data.nelement() for p in self.depth_net.parameters()])]
        print('Number of params in depth_net: {}'.format(*self.n_params))

    def forward(self, inputs):
        outputs = {}
        loss_dict = {}
        
        # predict disp
        for i in self.frame_idx:
            outputs.update(self.depth_net(inputs["color", i], frame_idx=i))
        
        # predict poses
        tgt_image = inputs["color", 0]
        K = inputs['K']
        K_inv = inputs['K_inv']
        disp_tgt = torch.cat( [F.interpolate(outputs['disp', 0, 0], [self.image_H, self.image_W], mode="bilinear", align_corners=True),
                               F.interpolate(outputs['disp', 0, 1], [self.image_H, self.image_W], mode="bilinear", align_corners=True),
                               F.interpolate(outputs['disp', 0, 2], [self.image_H, self.image_W], mode="bilinear", align_corners=True),
                               F.interpolate(outputs['disp', 0, 3], [self.image_H, self.image_W], mode="bilinear", align_corners=True)], 1)

        for frame_id in self.src_idx:
            if frame_id == 'stereo':
                outputs[('pose', frame_id)] = inputs['stereo_T']
            else:
                flow_inputs = {('color', 0): tgt_image, ('color', 1): inputs['color', frame_id]}
                flow_loss, flow, flow_inv, mask_occ_fw, mask_occ_bw, consis_fw, consis_bw = self.flow_model(flow_inputs)
                pose_pack, loss_pack = self.predict_pose(flow, mask_occ_fw, consis_fw, disp_tgt, K, K_inv, self.ratio_kpts, self.num_kpts, frame_id, 1, inputs['state'])
                outputs[('flow', frame_id)] = flow
                outputs[('flow_inv', frame_id)] = flow_inv
                outputs.update(pose_pack)
                loss_dict.update(loss_pack)
                loss_dict.update(flow_loss)

                if self.UseBackLoss:
                    # disp_ref = F.interpolate(outputs['disp', frame_id, 0], [self.image_H, self.image_W], mode="bilinear", align_corners=True)
                    disp_ref = torch.cat( [F.interpolate(outputs['disp', frame_id, 0], [self.image_H, self.image_W], mode="bilinear", align_corners=True),
                                        F.interpolate(outputs['disp', frame_id, 1], [self.image_H, self.image_W], mode="bilinear", align_corners=True),
                                        F.interpolate(outputs['disp', frame_id, 2], [self.image_H, self.image_W], mode="bilinear", align_corners=True),
                                        F.interpolate(outputs['disp', frame_id, 3], [self.image_H, self.image_W], mode="bilinear", align_corners=True)], 1)
                    pose_pack, loss_pack = self.predict_pose(flow_inv, mask_occ_bw, consis_bw, disp_ref, K, K_inv, self.ratio_kpts, self.num_kpts, frame_id, -1, inputs['state'])
                    outputs.update(pose_pack)
                    loss_dict.update(loss_pack)

        # compute reconstruct loss
        loss_reconstruct = self.compute_loss(inputs, outputs)
        loss_dict.update(loss_reconstruct)

        losses_pack = collections.OrderedDict()
        for k, v in loss_dict.items():
            name = k if isinstance(k, str) else k[0]
            losses_pack[name] = v if name not in losses_pack else losses_pack[name] + v

        for k in list(losses_pack.keys()):
            if 'flow' in self.cfgs['model']:
                if k[0] == 'D':
                    del losses_pack[k]
            elif 'depth' in self.cfgs['model']:
                if k[0] == 'F':
                    del losses_pack[k]

        total_loss = sum(_value for _key, _value in losses_pack.items())
        return losses_pack, total_loss

    def infer_depth(self, tenFirst):
        disps = self.depth_net(tenFirst)['disp', 0, 0]
        return disps

    def infer_flow(self, tenFirst, tenSecond):
        return self.flow_model.infer_flow(tenFirst, tenSecond)

    def infer_vo(self, inputs, kpts=None):
        tenFirst = inputs['color', 0]
        tenSecond = inputs['color', 1]
        K = inputs['K']
        K_inv = inputs['K_inv']
        B, _, H, W = tenFirst.shape
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        
        # infer flow
        # start.record()
        flows     = self.flow_model.flow_net(tenFirst, tenSecond) # scale * [b,c,h,w]
        flows_inv = self.flow_model.flow_net(tenSecond, tenFirst) # scale * [b,c,h,w]

        flow = F.interpolate(flows[0] * 4.0, [H, W], mode='bilinear', align_corners=True)
        flow_inv = F.interpolate(flows_inv[0] * 4.0, [H, W], mode='bilinear', align_corners=True)
        # end.record()
        # torch.cuda.synchronize()
        # ms = start.elapsed_time(end)
        # with open('./debug/test-time/flow.txt', 'a') as f:
        #     f.write(str(ms) + '\n')

        # start.record()
        consis_mask_fw, consis_mask_bw, consis_fw, consis_bw = Flow_Model.consistent(flow, flow_inv)
        occ_mask_fw = Flow_Model.occ_mask(flow_inv)
        masks_fw = consis_mask_fw * occ_mask_fw

        masks_oob_fw = self.mask_flow_oob(flow)
        # end.record()
        # torch.cuda.synchronize()
        # ms = start.elapsed_time(end)
        # with open('./debug/test-time/mask.txt', 'a') as f:
        #     f.write(str(ms) + '\n')

        # infer depth
        # start.record()
        disp = self.depth_net(tenFirst)['disp', 0, 0] # scale * [b,c,h,w]
        disp, _ = self.disp_to_depth(disp)
        disp = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=True)
        # end.record()
        # torch.cuda.synchronize()
        # ms = start.elapsed_time(end)
        # with open('./debug/test-time/depth.txt', 'a') as f:
        #     f.write(str(ms) + '\n')

        # start.record()
        ref_scores = masks_fw * masks_oob_fw * (1 / consis_fw.abs().clamp(min=1e-2))
        ratio = self.cfgs['ratio_keypoints']
        if kpts is None:
            sample_match, sample_disp, pose_init, rigrid_scores = self.robust_sample(flow, disp, masks_fw, ref_scores, K, K_inv, 1.0, H*W)
        else:
            sample_match, sample_disp, pose_init, rigrid_scores = self.robust_sample(flow, disp, masks_fw, ref_scores, K, K_inv, ratio, kpts)
        # end.record()
        # torch.cuda.synchronize()
        # ms = start.elapsed_time(end)
        # with open('./debug/test-time/sample.txt', 'a') as f:
        #     f.write(str(ms) + '\n')

        # start.record()
        with torch.no_grad():
            pose, disp_update = self.ba_net(sample_match, sample_disp, pose_init, K, K_inv)
        # end.record()
        # torch.cuda.synchronize()
        # ms = start.elapsed_time(end)
        # with open('./debug/test-time/ba.txt', 'a') as f:
        #     f.write(str(ms) + '\n')

        # collect eval data
        eval_data = {}
        eval_data['mask_inlines'] = rigrid_scores
        eval_data['flow'] = flow
        eval_data['disp'] = disp
        if kpts is None:
            eval_data['ba_disp'] = disp_update.view(B, 1, H, W)
        else:
            eval_data['sample_points'] = sample_match
            # eval_data['ba_disp'] = disp_update

        return pose, eval_data

    def predict_pose(self, flow, mask, score, disp, K, K_inv, rkpts, nkpts, fid, ub, state):
        masks_oob = self.mask_flow_oob(flow)

        disp, _ = self.disp_to_depth(disp)

        # # compute ego_motion
        scores = mask * masks_oob * (1 / score.abs().clamp(min=1e-2))
        sample_match, sample_disp, pose_init, rigrid_scores, geo_losses = self.robust_sample(flow, disp, mask, scores, K, K_inv, rkpts, nkpts, UseGeo=True)

        with torch.no_grad():
            pose, disps_update = self.ba_net(sample_match, sample_disp[:, 0:1, :], pose_init, K, K_inv, state)

        loss_ba_geo = self.cfgs['w_F_geo'] * geo_losses.mean()

        outputs = {}
        loss_pack = {}
        outputs[('pose', fid, ub)] = pose
        outputs[('rigrid_score', fid, ub)] = rigrid_scores
        outputs[('ba_disp', fid, ub)] = disps_update
        outputs[('sample_disp', fid, ub)] = sample_disp
        loss_pack[('F_geo', fid, ub)] = loss_ba_geo

        return outputs, loss_pack

    def compute_loss(self, inputs, outputs):
        K = inputs['K']
        K_inv = inputs['K_inv']
        tgt_image = inputs["color", 0]

        loss_dict = {}

        for scale in range(self.depth_scales):
            '''
            Init
            '''
            disp = outputs['disp', 0, scale]
            disp_rescale = F.interpolate(disp, [self.image_H, self.image_W], mode="bilinear", align_corners=True)
            _, tgt_depth = self.disp_to_depth(disp_rescale)

            """
            feedforword loss
            """
            for fid in self.src_idx:
                pose = outputs[('pose', fid, 1)]
                ref_image = inputs["color", fid]
                ref_disp = F.interpolate(outputs['disp', fid, scale], [self.image_H, self.image_W], mode="bilinear", align_corners=True)
                _, ref_depth = self.disp_to_depth(ref_disp)

                # ba_loss
                sample_disp = outputs['sample_disp', fid, 1][:, 0:1, :]
                loss_dict[('D_ba', fid, 1)] = self.cfgs['w_D_ba'] * ((sample_disp - outputs['ba_disp', fid, 1]).abs()).mean()
                if self.UseBackLoss:
                    sample_disp_back = outputs['sample_disp', fid, -1][:, 0:1, :]
                    loss_dict[('D_ba', fid, -1)] = self.cfgs['w_D_ba'] * ((sample_disp_back - outputs['ba_disp', fid, -1]).abs()).mean()

                # recstruction
                rigid_mask = outputs['rigrid_score', fid, 1]
                warped_tgt_image, warped_tgt_depth, proj_depth, tgt_mask, corrds_tgt = \
                    self.reproject_warp(ref_image, ref_depth, tgt_depth, pose, K, K_inv)
                reconstruction_loss, consis_loss = self.depth_loss(tgt_image, warped_tgt_image, ref_image, tgt_mask, warped_tgt_depth, proj_depth, rigid_mask)
                loss_dict[('D_rec', scale, fid, 1)] = self.cfgs['w_D_rec'] * reconstruction_loss / self.depth_scales
                loss_dict[('D_cns', scale, fid, 1)] = self.cfgs['w_D_cns'] * consis_loss / self.depth_scales

                # flow consisit
                flow_depth_consist_loss = self.flow_consist_loss(outputs['flow', fid], corrds_tgt, outputs['rigrid_score', fid, 1])
                loss_dict[('F_cns', scale, fid, 1)] = self.cfgs['w_F_cns'] * flow_depth_consist_loss / self.depth_scales

            """
            backforword loss
            """
            if self.UseBackLoss:
                for fid in self.src_idx:
                    pose_inv = outputs[('pose', fid, -1)]
                    ref_image = inputs["color", fid]
                    ref_disp = F.interpolate(outputs['disp', fid, scale], [self.image_H, self.image_W], mode="bilinear", align_corners=True)
                    _, ref_depth = self.disp_to_depth(ref_disp)

                    rigid_mask = outputs['rigrid_score', fid, -1]
                    warped_ref_image, warped_ref_depth, proj_depth, ref_mask, corrds_ref = \
                        self.reproject_warp(tgt_image, tgt_depth, ref_depth, pose_inv, K, K_inv)
                    reconstruction_loss, consis_loss = self.depth_loss(ref_image, warped_ref_image, tgt_image, ref_mask, warped_ref_depth, proj_depth, rigid_mask)
                    loss_dict[('D_rec', scale, fid, -1)] = self.cfgs['w_D_rec'] * reconstruction_loss / self.depth_scales
                    loss_dict[('D_cns', scale, fid, -1)] = self.cfgs['w_D_cns'] * consis_loss / self.depth_scales

            """
            disp mean normalization
            """
            if self.NormDisp:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = self.get_smooth_loss(disp, tgt_image)
            
            # for fid in self.src_idx:
            #     smooth_loss += self.get_smooth_loss(outputs['disp', fid, scale], inputs["color", fid])

            loss_dict[('D_smth', scale)] = self.cfgs['w_D_smth'] * smooth_loss / (2 ** scale) / self.depth_scales

        return loss_dict

    def disp_to_depth(self, disp):
        min_disp = 1 / self.max_depth  # 0.01
        max_disp = 1 / self.min_depth  # 10

        # scaled_disp = min_disp + (max_disp - min_disp) * disp  # (10-0.01)*disp+0.01
        # depth = 1 / scaled_disp
        
        # att
        scaled_disp = disp.clamp(min=min_disp, max=max_disp)
        depth = 1 / scaled_disp
        return scaled_disp, depth
    
    def points_select(self, sMask, ratio=0.2, num_keypoints=2000, robust=True):
        # return B x 4 x N (n points)
        B, _, H, W = sMask.shape
        N = H * W
        device = sMask.get_device()

        sMask = sMask.view(B, 1, N)
        scores, indices = torch.topk(sMask, int(ratio * N), dim=2) # [B, 1, ratio*tnum]
        select_idxs = torch.randint(0, int(ratio * N), (num_keypoints, )).view(1, 1, num_keypoints).expand(B, 1, num_keypoints)

        # sample the non-zero matches
        if robust:
            nonzeros = torch.min(torch.sum(scores > 0, dim=-1)) 
            num_keypoints = np.minimum(nonzeros.detach().cpu().numpy(), num_keypoints)

            select_idxs = []
            for i in range(B):
                nonzero_idx = torch.nonzero(scores[i, 0, :]) # [nonzeros, 1]
                rand_int = torch.randint(0, nonzero_idx.shape[0], [num_keypoints])
                select_idx = nonzero_idx[rand_int, :] # [num, 1]
                select_idxs.append(select_idx)
            select_idxs = torch.stack(select_idxs, 0).permute(0, 2, 1) # [b, 1, X]

        # ratio == 1.0 for validation
        if ratio >= 1.0:
            indices = torch.arange(0, N).to(device).view(1, 1, N).expand(B, 1, N)
            select_idxs = torch.arange(0, N).to(device).view(1, 1, N).expand(B, 1, N)

        return indices, select_idxs

    def points_sample(self, ten, indices, rand_samples):
        B, C, H, W = ten.shape

        x = ten.view(B, C, H*W)
        if indices is not None:
            x = torch.gather(x, index=indices.repeat(1, C, 1), dim=2) # [b, C, X]
            x = torch.gather(x, index=rand_samples.repeat(1, C, 1), dim=2)
        return x

    def get_matchs(self, flow):
        device = flow.get_device()

        # grid = utils.meshgrid(flow.shape).to(device).float().detach()
        grid = self.grid2d.to(device) if self.training else self.grid2d[0:1, :, :, :].to(device)
        grid2 = torch.cat([(grid[:, 0:1, :, :] + flow[:, 0:1, :, :]), (grid[:, 1:2, :, :] + flow[:, 1:2, :, :])], 1)
        match = torch.cat([grid, grid2], 1)
        return match

    def mask_flow_oob(self, flow):
        device = flow.get_device()

        # grid = utils.meshgrid(flow.shape).to(device).float().detach()
        grid = self.grid2d.to(device) if self.training else self.grid2d[0:1, :, :, :].to(device)
        grid2 = torch.cat([(grid[:, 0:1, :, :] + flow[:, 0:1, :, :]), (grid[:, 1:2, :, :] + flow[:, 1:2, :, :])], 1)
        M1 = (grid2[:, 0:1, :, :] >= 0).float() * (grid2[:, 0:1, :, :] < self.image_W).float()
        M2 = (grid2[:, 1:2, :, :] >= 0).float() * (grid2[:, 1:2, :, :] < self.image_H).float()
        M = (M1 * M2).detach()
        return M

    def reproject_warp(self, ref_image, ref_depth, depth, pose, K, K_inv):
        '''
        input: *depth: [B x 1 x H x W]    depth map
            *image  [B x 3 x H x W]    image
            pose:  [B x 4 x 4]         pose
            K*:     [B x 3 x 3]         intrinsic

        output: warped_ref_img [B x 3 x H x W]
                warped_depth, computed_depth, mask [B x 1 x H x W]
        '''
        B, _, H, W = depth.shape
        device = depth.get_device()
        # initialize grid
        # grid2d = utils.meshgrid(depth.shape, norm=False).to(depth.get_device()).float().detach() # [B 2 HW]
        grid2d = self.grid2d.to(device)

        # grid2d = self.grid2d
        grid3d = torch.cat( [grid2d, torch.ones_like(depth)], 1)

        # to cam space
        p3d = K_inv.matmul(grid3d.view(B, 3, H*W)) * depth.view(B, 1, H*W) # [B 3 HW]

        # rot and trans
        K_pose = K.matmul(pose[:, :3, :])
        R, T = K_pose[:, :3, :3], K_pose[:, :3, 3:]
        p3d_hat = R.matmul(p3d) + T # b 3 hw

        # to image space
        X = p3d_hat[:, 0]  # [B, hw]
        Y = p3d_hat[:, 1]
        Z = p3d_hat[:, 2].clamp(min=1e-3)

        # norm
        X_norm = 2 * (X / Z) / (W - 1) - 1
        Y_norm = 2 * (Y / Z) / (H - 1) - 1

        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

        # sample
        corrds = torch.stack([X / Z, Y / Z], dim=2).view(B, H, W, 2)
        pixel_coords = torch.stack([X_norm, Y_norm], dim=2).view(B, H, W, 2)  # [B, H, W, 2]
        warped_tgt_img = F.grid_sample(ref_image.type_as(pixel_coords), pixel_coords, padding_mode='zeros', mode='nearest', align_corners=True) # [B C H W]
        warped_tgt_depth = F.grid_sample(ref_depth.type_as(pixel_coords), pixel_coords, padding_mode='zeros', mode='nearest', align_corners=True) # [B C H W]

        # mask
        valid_points = pixel_coords.abs().max(dim=-1)[0] <= 1
        valid_mask = valid_points.unsqueeze(1).float()

        return warped_tgt_img, warped_tgt_depth, Z.view(B, 1, H, W), valid_mask, corrds
    
    def depth_loss(self, ref_image, warped_ref_image, tgt_image, ref_mask, warped_depth, proj_depth, rigid_mask):
        # img_diff
        img_diff = torch.sqrt(torch.pow(ref_image - warped_ref_image, 2) + 1e-3 ** 2).mean(1, True)
        img_ssim = (0.5 * (1 - utils.SSIM(ref_image, warped_ref_image))).clamp(0, 1).mean(1, True)
        img_loss = (0.15 * img_diff + 0.85 * img_ssim)
        img_loss = img_loss / (ref_image.mean(1, True).clamp(min=1e-2))

        # depth consist
        depth_diff = (torch.abs(1.0 - proj_depth / (warped_depth + 1e-12)) * rigid_mask * ref_mask)
        # depth_diff = depth_diff.mean((1,2,3)) / ((ref_mask * rigid_mask).mean((1,2,3)) + 1e-12)

        # mask
        # img_loss = img_loss * (1 - depth_diff)

        # self.UseAutoMask = False
        if self.UseAutoMask:
            valid_mask = (img_diff.mean(dim=1, keepdim=True) < (ref_image - tgt_image).abs().mean(dim=1, keepdim=True)).float() * ref_mask
        else:
            valid_mask = ref_mask


        # photometric_loss = img_loss * valid_mask / (valid_mask.sum() + 1e-7)
        reconstruction_loss = self.mean_on_mask(img_loss, valid_mask)
        geometry_consistency_loss = self.mean_on_mask(depth_diff, rigid_mask * ref_mask)
        return reconstruction_loss, geometry_consistency_loss
    
    def mean_on_mask(self, diff, valid_mask):
        return (diff * valid_mask).sum() / (valid_mask.sum() + 1e-7)

    def flow_consist_loss(self, flow, reproj_corrds, mask):
        device = flow.get_device()
        B, _, H, W = flow.shape

        # grid = utils.meshgrid(flow.shape, norm=False).to(device).float().detach() # [B 2 H W]
        grid = self.grid2d.to(device)
        # grid2 = torch.cat([(grid[:, 0:1, :, :] + flow[:, 0:1, :, :]),
        #                    (grid[:, 1:2, :, :] + flow[:, 1:2, :, :])], 1)

        diff = torch.abs(grid + flow - reproj_corrds.detach().permute(0, 3, 1, 2).contiguous())
        return self.mean_on_mask(diff, mask)

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        a1 = 2.0
        a2 = 0.0
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-a1 * img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-a2 * img_dyy.abs().mean(1, True)))

        return smooth1# + smooth2

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def robust_sample(self, flow, disp, occ_mask, sMask, K, K_inv, ratio, num_kpts, UseGeo=False):
        match_pts = self.get_matchs(flow)

        indices, pts_samples = self.points_select(sMask, ratio=ratio, num_keypoints=num_kpts)
        sample_match = self.points_sample(match_pts, indices, pts_samples)
        sample_disp = self.points_sample(disp, indices, pts_samples)

        if self.resample:
            rigid_scores, geo_losses = self.movingMask(sample_match, match_pts, occ_mask) # B 1 H W, B
            if ratio < 1.0:
                # re-sample
                sMask = rigid_scores * occ_mask
                indices, pts_samples = self.points_select(sMask, ratio=ratio, num_keypoints=num_kpts)
                sample_match = self.points_sample(match_pts, indices, pts_samples)
                sample_disp = self.points_sample(disp, indices, pts_samples)

            pose_init = self.init_pnp(sample_match, sample_disp[:, 0:1, :], K)

            # filt out 
            if ratio < 1.0:
                sample_match, sample_disp, _, _ = ray_angle_filter(sample_match, sample_disp, pose_init, K, K_inv)
        else:
            pose_init = self.init_pnp(sample_match, sample_disp[:, 0:1, :], K)
            rigid_scores = torch.ones_like(occ_mask)
            geo_losses = torch.Tensor([0])

        if UseGeo:
            return sample_match, sample_disp, pose_init, sMask, geo_losses
        else:
            return sample_match, sample_disp, pose_init, sMask

if __name__ == "__main__":
    pass
