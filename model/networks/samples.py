'''
This code was ported from existing repos
[LINK] https://github.com/B1ueber2y/TrianFlow
'''
import torch
import numpy as np
import cv2

class MovingMask(torch.nn.Module):
    def __init__(self, cfgs):
        super(MovingMask, self).__init__()
        self.cfgs = cfgs

        self.W = cfgs['width']
        self.H = cfgs['height']

        self.thres_rigid = 0.5 #if cfgs['dataset'] == 'kitti' else 1.0
        self.thres_inliner = 0.1

    def forward(self, pts_coord, flow_coords, occ_mask):
        B, _, N = pts_coord.shape
        _, _, H, W = flow_coords.shape
        device = pts_coord.get_device()
        
        # RANSAC 8-point and best F selection
        pts_coord_np = pts_coord.permute(0, 2, 1).detach().cpu().numpy()
        F = []
        for i in range(B):
            f, m = cv2.findFundamentalMat(pts_coord_np[i, :, 0:2], pts_coord_np[i, :, 2:4], cv2.FM_RANSAC, 0.1, 0.99)
            # if self.cfgs['dataset'] == 'kitti':
            #     f, m = cv2.findFundamentalMat(pts_coord_np[i, :, 0:2], pts_coord_np[i, :, 2:4], cv2.FM_RANSAC, 0.1, 0.99)
            # else:
            #     f, m = cv2.findFundamentalMat(pts_coord_np[i, :, 0:2], pts_coord_np[i, :, 2:4], cv2.FM_RANSAC, 0.1, 0.99)
            F.append(f)
        F = torch.from_numpy(np.stack(F, axis=0)).float().to(device)

        # dist map
        _, dist_map = self.compute_epipolar_loss(F, flow_coords.view(B, 4, H*W), occ_mask.view(B, 1, H*W))
        dist_map = dist_map.view([B, H, W, 1]).permute(0, 3, 1, 2)

        # Compute geo loss for regularize correspondence.
        rigid_mask, inlier_mask, rigid_score = self.get_rigid_mask(dist_map, self.thres_rigid, self.thres_inliner)
        
        # We only use rigid mask to filter out the moving objects for computing geo loss.
        geo_loss = (dist_map * (rigid_mask - inlier_mask)).mean((1,2,3)) / (rigid_mask - inlier_mask).mean((1,2,3))
        
        return rigid_score.detach(), geo_loss
    
    def compute_epipolar_loss(self, fmat, match, mask):
        # fmat: [b, 3, 3] match: [b, 4, h*w] mask: [b,1,h*w]
        num_batch = match.shape[0]
        match_num = match.shape[-1]

        points1 = match[:,:2,:]
        points2 = match[:,2:,:]
        ones = torch.ones(num_batch, 1, match_num).to(points1.get_device())
        points1 = torch.cat([points1, ones], 1) # [b,3,n]
        points2 = torch.cat([points2, ones], 1).transpose(1,2) # [b,n,3]

        # compute fundamental matrix loss
        fmat = fmat.unsqueeze(1)
        fmat_tiles = fmat.view([-1,3,3])
        epi_lines = fmat_tiles.bmm(points1) #[b,3,n]  [b*n, 3, 1]
        dist_p2l = torch.abs((epi_lines.permute([0, 2, 1]) * points2).sum(-1, keepdim=True)) # [b,n,1]
        a = epi_lines[:,0,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        b = epi_lines[:,1,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        dist_div = torch.sqrt(a*a + b*b) + 1e-6
        dist_map = dist_p2l / dist_div # [B, n, 1]
        loss = (dist_map * mask.transpose(1,2)).mean([1,2]) / mask.mean([1,2])
        return loss, dist_map
    
    def get_rigid_mask(self, dist_map, rigid_thres, inlier_thres):
        rigid_mask = (dist_map < rigid_thres).float()
        inlier_mask = (dist_map < inlier_thres).float()
        rigid_score = rigid_mask * 1.0 / (1.0 + dist_map)
        return rigid_mask, inlier_mask, rigid_score

def ray_angle_filter(match, disp, pose, K, K_inv, return_angle=False):
    # match: [b, 4, n] P: [B, 3, 4]
    # b, n = match.shape[0], match.shape[2]
    B, _, N = match.shape
    device = match.get_device()

    static_pose = torch.cat([torch.eye(3), torch.zeros([3,1])], -1).view(1, 3, 4).to(device) # [b,3,4]

    P1 = K.matmul(static_pose.repeat(B, 1, 1))
    P2 = K.matmul(pose[:, :3, :])

    RT1 = K_inv.bmm(P1) # [b, 3, 4]
    RT2 = K_inv.bmm(P2)
    ones = torch.ones([B,1,N]).to(match.get_device())
    pts1 = torch.cat([match[:,:2,:], ones], 1)
    pts2 = torch.cat([match[:,2:,:], ones], 1)
    
    ray1_dir = (RT1[:,:,:3].transpose(1,2)).bmm(K_inv).bmm(pts1)# [b,3,n]
    ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
    ray1_origin = (-1) * RT1[:,:,:3].transpose(1,2).bmm(RT1[:,:,3].unsqueeze(-1)) # [b, 3, 1]
    ray2_dir = (RT2[:,:,:3].transpose(1,2)).bmm(K_inv).bmm(pts2) # [b,3,n]
    ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
    ray2_origin = (-1) * RT2[:,:,:3].transpose(1,2).bmm(RT2[:,:,3].unsqueeze(-1)) # [b, 3, 1]

    # We compute the angle betwwen vertical line from ray1 origin to ray2 and ray1.
    p1p2 = (ray1_origin - ray2_origin).repeat(1,1,N)
    verline = ray2_origin.repeat(1,1,N) + torch.sum(p1p2 * ray2_dir, dim=1, keepdim=True) * ray2_dir - ray1_origin.repeat(1,1,N) # [b,3,n]
    cosvalue = torch.sum(ray1_dir * verline, dim=1, keepdim=True)  / \
        ((torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12) * (torch.norm(verline, dim=1, keepdim=True, p=2) + 1e-12))# [b,1,n]

    mask = (cosvalue > 0.001).float() # we drop out angles less than 1' [b,1,n]
    flag = 0
    num = torch.min(torch.sum(mask, -1)).int()
    if num.cpu().detach().numpy() == 0:
        flag = 1
        filt_match = match[:,:,:1000]
        filt_disp = disp[:,:,:1000]
        filt_mask = mask[:,:,:1000]
        if return_angle:
            return filt_match, flag, torch.zeros_like(mask).to(filt_match.get_device())
        else:
            return filt_match, filt_disp, filt_mask, flag
    nonzero_idx = []
    for i in range(B):
        idx = torch.nonzero(mask[i,0,:])[:num] # [num,1]
        nonzero_idx.append(idx)
    nonzero_idx = torch.stack(nonzero_idx, 0) # [b,num,1]
    filt_match = torch.gather(match.transpose(1,2), index=nonzero_idx.repeat(1,1,4), dim=1).transpose(1,2) # [b,4,num]
    filt_disp = torch.gather(disp.transpose(1,2), index=nonzero_idx.repeat(1,1,disp.shape[1]), dim=1).transpose(1,2) # [b,4,num]
    filt_mask = torch.gather(mask.transpose(1,2), index=nonzero_idx.repeat(1,1,1), dim=1).transpose(1,2) # [b,4,num]
    if return_angle:
        return filt_match, flag, mask
    else:
        return filt_match, filt_disp, filt_mask, flag