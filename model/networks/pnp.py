import cv2
import numpy as np
import torch

class PnPPose(torch.nn.Module):
    def __init__(self, cfgs):
        super(PnPPose, self).__init__()
        self.W = cfgs['width']
        self.H = cfgs['height']
        self.device = cfgs['device']

    def forward(self, pts_coord, pts_disp, K):
        batch, _, _ = pts_coord.shape
        pose = torch.eye(4).view(1, 4, 4).expand(batch, 4, 4).to(self.device)

        for i in range(batch):
            pose[i], inliner = self.solve_pose_pnp(pts_coord[i, 0:2, :], pts_coord[i, 2:4, :], pts_disp[i][0], K[i])
        return pose

    def unprojection(self, xy, depth, K):
        # xy: [N, 2] image coordinates of match points
        # depth: [N] depth value of match points
        N = xy.shape[0]
        # initialize regular grid
        ones = np.ones((N, 1))
        xy_h = np.concatenate([xy, ones], axis=1)
        xy_h = np.transpose(xy_h, (1,0)) # [3, N]
        #depth = np.transpose(depth, (1,0)) # [1, N]
        
        K_inv = np.linalg.inv(K)
        points = np.matmul(K_inv, xy_h) * depth
        points = np.transpose(points) # [N, 3]
        return points

    def solve_pose_pnp(self, xy1, xy2, disp, K):
        # Use pnp to solve relative poses.
        # xy1, xy2: [2, N] depth1: [H, W]
        xy1 = np.transpose(xy1.detach().cpu().numpy(), (1,0))
        xy2 = np.transpose(xy2.detach().cpu().numpy(), (1,0))
        depth1 = 1 / disp.detach().cpu().numpy()
        K = K.detach().cpu().numpy()

        img_h, img_w = self.H, self.W

        # Unproject to 3d space
        points1 = self.unprojection(xy1, depth1, K)

        # ransac
        best_rt = []
        max_inlier_num = 0
        max_ransac_iter = 5
        for i in range(max_ransac_iter):
            if xy2.shape[0] > 4:
                flag, r, t, inlier = cv2.solvePnPRansac(objectPoints=points1, imagePoints=xy2, cameraMatrix=K, distCoeffs=None, iterationsCount=1000, reprojectionError=1)
                if flag and inlier.shape[0] > max_inlier_num:
                    best_rt = [r, t]
                    max_inlier_num = inlier.shape[0]
                    break
        best_rt = [r, t]

        pose = np.eye(4)
        if len(best_rt) != 0:
            r, t = best_rt
            pose[:3,:3] = cv2.Rodrigues(r)[0]
            pose[:3,3:] = t

        return torch.from_numpy(pose), inlier