import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

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

def project_2d3d(points, depth, K_inv):
    '''
        points: [B x 2 x N]     image coordinates
        depth:  [B x 1 x N]     depth map
        K_inv:  [B x 3 x 3]      intrinsic
    '''
    B, _, N = points.shape

    # initialize regular grid
    ones = torch.ones(B, 1, N).to(points.get_device())
    xy3d = torch.cat([points, ones], 1) # [B x 3 x N]
    return torch.matmul(K_inv, xy3d) * depth

def homogeneous(pose):
    B, _, _ = pose.shape
    ext = torch.tensor([0., 0., 0., 1.]).view(1, 1, 4).expand(B, -1, -1).to(pose.get_device())
    return torch.cat( [pose, ext], 1)

def pose2R(pose):
    return pose[:, :3, :3]

def pose2T(pose):
    return pose[:, :3, 3:]

def euclidean_transformation(points3d, pose):
    # R * point3d + T
    return torch.matmul(pose2R(pose), points3d) + pose2T(pose)

def project_3d2d(points3d, K):
    # K * T / Z
    p3d = torch.matmul(K, points3d)
    return p3d[:, 0:2, :] / p3d[:, 2:, :].clamp(min=1e-2)

def reproject_error(pts2dOne, pts2dTwo, dispOne, pose, K, K_inv):
    pts_depth = 1 / dispOne.clamp(min=1e-2)
    point3d = project_2d3d(pts2dOne, pts_depth, K_inv)
    point3d_hat = euclidean_transformation(point3d, pose)
    point2d_hat = project_3d2d(point3d_hat, K)
    return pts2dTwo - point2d_hat # [B, 2, N]

def reproject_mask(pts2d, disp, pose, K, K_inv, size):
    B, _, N = pts2d.shape
    H, W = size
    pts2d_1 = pts2d[:, 0:2, :]
    pts2d_2 = pts2d[:, 2:4, :]

    pts_depth = 1 / disp.clamp(min=1e-2)
    point3d = project_2d3d(pts2d_1, pts_depth, K_inv)
    point3d_hat = euclidean_transformation(point3d, pose)
    point2d_hat = project_3d2d(point3d_hat, K)

    points_2d = point2d_hat.view(B, 2, N)

    # mask beyond boundary
    m_star = torch.Tensor([0, 0]).to(points_2d.get_device())
    m_end  = torch.Tensor([W-1, H-1]).to(points_2d.get_device())
    oob_mask = (points_2d.permute(0,2,1) > m_star).all(-1, True).float() * (points_2d.permute(0,2,1) < m_end).all(-1, True).float() # [B 1 N]
    oob_mask = oob_mask.permute(0, 2, 1).detach() # detach [B 1 N]

    return oob_mask


def huber_weight(err, mask=None, b=9):
    # err [B 2 N]
    err_norm = err.pow(2).sum(dim=1, keepdim=True).sqrt().expand_as(err)
    weight = (b  / err_norm.clamp(min=1e-12)).clamp(max=1).permute(0, 2, 1)
    weight = torch.sqrt(weight) * mask.permute(0, 2, 1)
    return torch.diag_embed(weight).detach() # [B, N, 2, 2]

def jacobi_pose(points3d, K):
    '''
        points3d:    [B x 3 x N]
        K:           [B x 3 x 3]
    '''
    B, _, N = points3d.shape
    fx, fy = K[:, 0:1, 0:1], K[:, 1:2, 1:2]
    X, Y, Z = points3d[:, 0:1, :], points3d[:, 1:2, :], points3d[:, 2:3, :].clamp(min=1e-12)
    inv_Z, inv_Z2 = 1 / Z, 1 / (Z * Z)
    full_zeros = torch.zeros_like(X)

    # J = [T, R]
    J = [fx*inv_Z,   full_zeros, -fx*X*inv_Z2, -fx*X*Y*inv_Z2,      fx + (fx*X*X*inv_Z2), -fx*Y*inv_Z,
         full_zeros, fy*inv_Z,   -fy*Y*inv_Z2, -fy-(fy*Y*Y*inv_Z2), fy*X*Y*inv_Z2,        fy*X*inv_Z]
    # 
    return -torch.cat( J, 1 ).view(B, 2, 6, N)# [B 2 6 N]

def jacobi_disp(points3d, pose, points2d, K, K_inv, disps):
    B, _, N = points3d.shape
    fx, fy = K[:, 0:1, 0:1], K[:, 1:2, 1:2]
    X, Y, Z = points3d[:, 0:1, :], points3d[:, 1:2, :], points3d[:, 2:3, :].clamp(min=1e-12) # [B 1 N]
    inv_Z, inv_Z2 = 1 / Z, 1 / (Z * Z)
    full_zeros = torch.zeros_like(X)
    
    J1 = [fx*inv_Z,   full_zeros, -fx*X*inv_Z2,
         full_zeros, fy*inv_Z,   -fy*Y*inv_Z2]
    J1 = torch.cat( J1, 1 ).view(B, 2, 3, N).permute(0, 3, 1, 2)

    T = pose2T(pose) # [B 3 1]
    Tx, Ty, Tz = T[:, 0:1, :].expand_as(X), T[:, 1:2, :].expand_as(Y), T[:, 2:3, :].expand_as(Z)
    DD = -1 / disps
    J_2 = torch.cat( [DD*(X-Tx), (Y-Ty)*DD, (Z-Tz)*DD], 1 ).view(B, 3, 1, N).permute(0, 3, 1, 2)

    J = -J1.matmul(J_2).permute(0, 2, 3, 1)
    return J

def hessian(J_pose, J_disp, weight):
    '''
        J_pose:      [B x 2 x 6 x N]
        J_disp:      [B x 2 x 1 x N]
        weight:      [B x N x 2 x 2]
    '''
    batch, _, _, N = J_pose.shape

    J_pose_T = J_pose.permute(0, 3, 2, 1).matmul(weight).matmul(weight)
    J_disp_T = J_disp.permute(0, 3, 2, 1).matmul(weight).matmul(weight)

    B = torch.matmul(J_pose_T, J_pose.permute(0, 3, 1, 2)).mean(dim=1) # [B x 6 x 6]
    E = torch.matmul(J_pose_T, J_disp.permute(0, 3, 1, 2)) # [B x N x 6 x 1] 
    C = torch.matmul(J_disp_T, J_disp.permute(0, 3, 1, 2)) # [B x N x 1 x 1]
    return B, E, C 

def calc_g(errors, J_pose, J_disp, weight):
    ''' compute right part of eq.
        errors:      [B x 2 x N]
        J_pose:      [B x 2 x 6 x N]
        J_disp:      [B x 2 x 1 x N]
    '''
    B, _, N = errors.shape
    e = errors.permute(0, 2, 1).view(B, N, 2, 1)  # [B x N x 2 x 1]

    J_pose_T = J_pose.permute(0, 3, 2, 1).matmul(weight)
    J_disp_T = J_disp.permute(0, 3, 2, 1).matmul(weight)

    g_pose = torch.matmul(-J_pose_T, e).mean(dim=1)  # [B x 6 x 1]
    g_disp = torch.matmul(-J_disp_T, e) # [B x N x 3 x 1]
    
    return g_pose, g_disp

def marginalization(B, E, C, V, W, L):
    '''
        B:    [B x 6 x 6]      leftup matrix
        E:    [B x N x 6 x 3]  rightup matrix
        C:    [B x N x 1 x 1]  rightbuttom matrix
        V:    [B x 6 x 1]      error of pose
        W:    [B x N x 1 x 1]  error of points3d
    '''
    batch, N, _, _ = E.shape
    E_trans = E.permute(0, 1, 3, 2)

    diag_B = torch.diagonal(B, 0, dim1=-2, dim2=-1).diag_embed()
    diag_C = torch.diagonal(C, 0, dim1=-2, dim2=-1).diag_embed()

    B = B + diag_B * L
    C = C + diag_C * L.view(batch, 1, 1, 1)
    C_inv = 1 / C.clamp(min=1)

    I = torch.eye(6).view(1, 6, 6).expand_as(B).to(B.get_device())
    B = B + I

    Hsc = torch.matmul(E, C_inv).matmul(E_trans).mean(dim=1)
    bsc = torch.matmul(E, C_inv).matmul(W).mean(dim=1)
    left = B - Hsc * (1.0 / (1 + L)) # [B x 6 x 6]
    right = V - bsc * (1.0 / (1 + L))# [B x 6 x 1]

    SVecI = 1 / (left.diagonal(0, dim1=-2, dim2=-1) + 10.0).sqrt()
    HFinalScaled = SVecI.diag_embed() @ left @ SVecI.diag_embed()
    delta_pose, _ = torch.solve(SVecI.diag_embed() @ right, HFinalScaled)
    delta_pose = SVecI.diag_embed() @ delta_pose

    p3d_sc = E_trans.matmul(delta_pose.view(batch, 1, 6, 1)) * (1.0 / (1 + L)).view(batch, 1, 1, 1)
    delta_points3d = C_inv.matmul(W - p3d_sc) # [B x N x 3 x 1]
    return delta_pose, delta_points3d

def v2m(vec): # input: [B, 3, 1]
    ''' rotation vection to antisymmetric matrix '''
    a1, a2, a3 = vec[:, 0:1, :], vec[:, 1:2, :], vec[:, 2:3, :]
    full_zeros = torch.zeros_like(a1)
    return torch.cat( [full_zeros, -a3, a2, a3, full_zeros, -a1, -a2, a1, full_zeros], 1).view(-1, 3, 3)

def se3_pose(se3):
    '''
    se3[:,0:3] = T
    se3[:,3:6] = R
    '''
    B, _, _ = se3.shape
    se3_T, se3_R = se3[:, 0:3, 0:1], se3[:, 3:6, 0:1]
    Nr = torch.sqrt(torch.sum(se3_R.pow(2.0), dim=1, keepdim=True)).clamp(min=1e-12)
    r =  se3_R / Nr
    
    I = torch.eye(3).to(se3.get_device()).view(1, 3, 3).expand(B, -1, -1)

    R = torch.cos(Nr)*I + (1 - torch.cos(Nr))*r.matmul(r.permute(0,2,1)) + torch.sin(Nr)*v2m(r)
    J = I + (Nr - torch.sin(Nr)) / Nr*v2m(r).matmul(v2m(r)) + (1-torch.cos(Nr))/Nr*v2m(r)
    T = J.matmul(se3_T)
    return torch.cat([R, T], 2) # [B, 3, 4]

def mean_on_mask(data, mask, dim, keepdim=False):
    return (data * mask).sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim)

class Geometric_BA(nn.Module):
    def __init__(self, cfgs):
        super(Geometric_BA, self).__init__()
        self.cfgs = cfgs
        self.num_iter = cfgs['ba_iter']

        self.W = cfgs['width']
        self.H = cfgs['height']

        self.min_disp = 1 / cfgs['max_depth']
        self.max_disp = 1 / cfgs['min_depth']

    def levenberg_marquardt(self, pts_coord, pts_disp, pose, K, K_inv, L, mask, wp):
        batch, _, N = pts_coord.shape

        pts_depth = 1 / pts_disp
        point3d = project_2d3d(pts_coord[:,0:2,:], pts_depth, K_inv)
        point3d_hat = euclidean_transformation(point3d, pose)
        point2d_hat = project_3d2d(point3d_hat, K)
        errors = (pts_coord[:, 2:, :] - point2d_hat) # B 2 N

        J_pose = jacobi_pose(point3d_hat, K) # B 2 6 N
        J_disp = jacobi_disp(point3d_hat, pose, pts_coord[:,0:2,:], K, K_inv, pts_disp) # B 2 1 N
        if self.training:
            J_disp = J_disp * wp

        hw = huber_weight(errors, mask) # [B N 2 2]
        B, E, C = hessian(J_pose, J_disp, hw) # 
        g_pose, g_disp = calc_g(errors, J_pose, J_disp, hw)

        delta_pose, delta_disp = marginalization(B, E, C, g_pose, g_disp, L)

        return delta_pose, delta_disp.view(batch, N, 1).permute(0, 2, 1)

    def forward(self, pts_coord, pts_disp, pose, K, K_inv, state=None):
        '''
        inputs: pts_coord: [B x 4 x N]
                pts_disp:  [B x 1 x N]
                K:         [B x 3 x 3]
        '''
        H, W = self.H, self.W
        batch, C, N = pts_disp.shape
        device = pts_coord.get_device()
        masks = torch.ones(pts_disp.shape).to(device)
        
        # init Lambda for levenberg_marquardt
        initE = reproject_error(pts_coord[:, 0:2, :], pts_coord[:, 2:4, :], pts_disp, pose, K, K_inv)
        initE = torch.mean(initE.norm(dim=1, keepdim=True), dim=2, keepdim=True)
        sigm = 5 
        L_min = 1 
        L_max = 10000 
        L = L_min + (L_max-L_min) * torch.exp(- initE / sigm)

        # pose = torch.eye(4).view(1,4,4).repeat(batch, 1, 1).to(device)
        # L = L* 0.0 + 0.1

        wp = 0.1
        for i in range(self.num_iter):
            mask_update = reproject_mask(pts_coord, pts_disp, pose, K, K_inv, (H, W))
            nums_valid_pts = mask_update.sum( dim=(1, 2) ).view(batch, 1, 1).expand_as(mask_update)
            masks = torch.where(nums_valid_pts > N / 4.0, masks * mask_update, masks)

            delta_pose, delta_disp = self.levenberg_marquardt(pts_coord, pts_disp, pose, K, K_inv, L, masks, wp) # [b 6 1] [b n 1 1]
            pts_disp_new = (pts_disp + delta_disp).clamp(self.min_disp, self.max_disp) 
            pose_new = homogeneous(se3_pose(delta_pose)).matmul(pose)

            # L update
            e1 = reproject_error(pts_coord[:, 0:2, :], pts_coord[:, 2:4, :], pts_disp_new, pose_new, K, K_inv)
            e2 = reproject_error(pts_coord[:, 0:2, :], pts_coord[:, 2:4, :], pts_disp, pose, K, K_inv) # B 2 N

            r1 = mean_on_mask(e1.norm(dim=1, keepdim=True), masks, dim=2, keepdim=True)
            r2 = mean_on_mask(e2.norm(dim=1, keepdim=True), masks, dim=2, keepdim=True)
            M = (r1 < r2).float().detach()
            M_pts = M.expand_as(pts_disp_new).detach()
            M_pose = M.expand_as(pose_new).detach()

            pts_disp = torch.where(M_pts == 1, pts_disp_new, pts_disp)
            pose = torch.where(M_pose == 1, pose_new, pose)
            L = torch.where(M == 1, L * 0.5, L * 10.0).clamp(min=1e-2, max=1e+6)

        return pose, pts_disp

if __name__ == "__main__":
    pass