import os
import glob
import cv2
import torch

from data.datasets.utils import *

class KITTI_Flow_2012(torch.utils.data.Dataset):
    def __init__(self, in_path, part='train', out_size=(256, 832), nums=None):
        super(KITTI_Flow_2012, self).__init__()
        self._name = 'KITTI_Flow_2012'
        self.image1_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'colored_0', '*_10.png')))[: nums]
        self.image2_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'colored_0', '*_11.png')))[: nums]
        self.occ_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'flow_occ', '*_10.png')))[: nums]
        self.noc_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'flow_noc', '*_10.png')))[: nums]
        
        assert len(self.image1_list) == len(self.image2_list)
        assert len(self.occ_list) == len(self.noc_list)

        self.out_size = out_size
        self.length = len(self.image1_list)

    def __len__(self):
        return self.length   

    def __getitem__(self, idx):
        H, W = self.out_size
        image1 = cv2.resize(cv2.imread(self.image1_list[idx]), (W, H))
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.0
        image2 = cv2.resize(cv2.imread(self.image2_list[idx]), (W, H))
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0

        occ = read_flow_png(self.occ_list[idx])
        noc_mask = read_flow_png(self.noc_list[idx])[:, :, 2]
        return image1, image2, occ, noc_mask

class KITTI_Flow_2015(torch.utils.data.Dataset):
    def __init__(self, in_path, part='train', out_size=(256, 832), nums=None):
        super(KITTI_Flow_2015, self).__init__()
        self._name = 'KITTI_Flow_2015'
        self.image1_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'image_2', '*_10.png')))[: nums]
        self.image2_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'image_2', '*_11.png')))[: nums]
        self.occ_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'flow_occ', '*_10.png')))[: nums]
        self.noc_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'flow_noc', '*_10.png')))[: nums]
        self.gt_mask_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'obj_map', '*_10.png')))[: nums]
        
        assert len(self.image1_list) == len(self.image2_list)
        assert len(self.occ_list) == len(self.noc_list)

        self.out_size = out_size
        self.length = len(self.image1_list)

    def __len__(self):
        return self.length   

    def __getitem__(self, idx):
        H, W = self.out_size
        image1 = cv2.resize(cv2.imread(self.image1_list[idx]), (W, H))
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.0
        image2 = cv2.resize(cv2.imread(self.image2_list[idx]), (W, H))
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0

        occ = read_flow_png(self.occ_list[idx])
        noc_mask = read_flow_png(self.noc_list[idx])[:, :, 2]
        gt_mask = cv2.imread(self.gt_mask_list[idx], -1)
        gt_mask[gt_mask > 0.0] = 1.0
        return image1, image2, occ, noc_mask, gt_mask

class KITTI_Flow_2015_Ex(torch.utils.data.Dataset):
    def __init__(self, in_path, part='train', out_size=(256, 832), nums=None, mode='g1'):
        super(KITTI_Flow_2015_Ex, self).__init__()
        self._name = 'KITTI_Flow_2015_Ex'
        # self.image1_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'image_2', '*_10.png')))[: nums]
        # self.image2_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'image_2', '*_11.png')))[: nums]
        self.occ_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'flow_occ', '*_10.png')))[: nums]
        self.noc_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'flow_noc', '*_10.png')))[: nums]
        self.gt_mask_list = sorted(glob.glob(os.path.join(in_path, part + 'ing', 'obj_map', '*_10.png')))[: nums]
        
        seq_mode = ['sequences_Geometric_Noise', 'sequences_Low_FPS', 'sequences_Photometric_Noise']
        if mode[0] == 'g':
            scene_name = os.path.join(in_path, seq_mode[0] + '_' + mode, 'image_2')
        elif mode[0] == 'p':
            scene_name = os.path.join(in_path, seq_mode[2] + '_' + mode, 'image_2')
        else:
            raise ValueError
        self.image1_list = sorted(glob.glob(os.path.join(scene_name, '*_10.png')))[: nums]
        self.image2_list = sorted(glob.glob(os.path.join(scene_name, '*_11.png')))[: nums]

        assert len(self.image1_list) == len(self.image2_list)
        assert len(self.occ_list) == len(self.noc_list)

        self.out_size = out_size
        self.length = len(self.image1_list)

    def __len__(self):
        return self.length   

    def __getitem__(self, idx):
        H, W = self.out_size
        image1 = cv2.resize(cv2.imread(self.image1_list[idx]), (W, H))
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.0
        image2 = cv2.resize(cv2.imread(self.image2_list[idx]), (W, H))
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.0

        occ = read_flow_png(self.occ_list[idx])
        noc_mask = read_flow_png(self.noc_list[idx])[:, :, 2]
        gt_mask = cv2.imread(self.gt_mask_list[idx], -1)
        gt_mask[gt_mask > 0.0] = 1.0
        return image1, image2, occ, noc_mask, gt_mask

if __name__ == '__main__':
    pass

