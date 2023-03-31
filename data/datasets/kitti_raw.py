import os
import torch
import torchvision
import numpy as np
import glob
from PIL import Image
import cv2

from data.datasets.utils import *

class KITTI_Raw(torch.utils.data.Dataset):
    def __init__(self, in_path, num=3, stride=1, out_size=(256, 832)):  # (256, 832)
        '''
        in_path/data/drive/image_02/0000000000.png
        ...
        in_path/data/calib_cam_to_cam.txt
        ...
        '''
        super(KITTI_Raw, self).__init__()
        self._name = 'KITTI_Raw'
        self.out_size = out_size
        self.in_path = in_path
        self.num = num
        self.stride = stride

        self.frame_ids = np.arange(2) if num == 2 else np.arange(num) - 1
        self.train_list = self.collect_scenes()

        # transforms
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        self.fn_resize = torchvision.transforms.Resize(self.out_size)
        self.fn_color_aug = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.to_tensor = torchvision.transforms.ToTensor()

    def collect_scenes(self):
        scenes_list = glob.glob(os.path.join(self.in_path, '20*', '*_sync', 'image_02'))
        train_list = []
        for scene in scenes_list:
            cam_file = os.path.join(os.path.dirname(os.path.dirname(scene)), 'calib_cam_to_cam.txt')
            images = sorted(glob.glob(os.path.join(scene, '*.png')))

            # get image pair
            iids = np.arange(0, len(images), 1)
            iids = np.reshape(iids, (1, iids.shape[0]))
            for i in range(self.num-1):
                iids = np.concatenate([iids, np.roll(iids[-1:, :], -self.stride)], axis=0)
            iids = iids[:, :(1-self.num) * self.stride]

            # to list
            for i in np.transpose(iids, (1, 0)):
                str_ = ''
                for j in i:
                    str_ = str_ + images[j].replace(self.in_path, '')[1:] + ' '
                str_ = str_ + cam_file.replace(self.in_path, '')[1:]
                train_list.append(str_)
        return train_list
    
    def __len__(self):
        return len(self.train_list)

    def pre_process(self, outputs, image_lists, do_flip):
        # step 1: load resize, flip and color transforms
        for i, p in enumerate(image_lists):
            # load left
            filename_left = os.path.join(self.in_path, p)
            im_left = load_rgb_from_file(filename_left, do_flip)
            W, H = im_left.size
            im_left = self.fn_resize(im_left)
            
            outputs[('color', self.frame_ids[i])] = self.to_tensor(im_left)

            # load transforms left
            outputs[('color_aug', self.frame_ids[i])] = self.to_tensor(self.fn_color_aug(im_left))

            # load right
            if self.frame_ids[i] == 0:
                filename_right = os.path.join(self.in_path, p.replace('image_02', 'image_03'))
                im_right = load_rgb_from_file(filename_right,do_flip)
                im_right = self.fn_resize(im_right)
                outputs[('color', 'stereo')] = self.to_tensor(im_right)
                outputs[('color_aug', 'stereo')] = self.to_tensor(self.fn_color_aug(im_right))
        return (H, W)

    def __getitem__(self, idx):
        """ Returns a single training item from the dataset as a dictionary.
            Values correspond to torch tensors.
            Keys in the dictionary are either strings or tuples:
                ("color", <frame_id>)          for raw colour images,
                ("color_aug", <frame_id>)      for augmented colour images,
                ("K") or ("inv_K")             for camera intrinsics,
                "stereo_T"                     for camera extrinsics
            <frame_id> is either:
                an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
            or
                "stereo" for the opposite image in the stereo pair.
        """
        data = self.train_list[idx].split(' ')
        image_lists, cam_file = data[:-1], data[-1]

        # load image seqs
        outputs = {}
        do_flip = np.random.rand() > 0.5
        in_size = self.pre_process(outputs, image_lists, do_flip)

        # load camera intrinsics
        cam_file = os.path.join(self.in_path, cam_file)
        outputs['K'] = load_cam_intrinsic(cam_file, in_size, self.out_size, position='P_rect_02')
        outputs['K_inv'] = outputs['K'].inverse()
        
        # load camera extrinsics of right camera
        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = -1 if do_flip else 1
        stereo_T[0, 3] = -1 * baseline_sign * 0.015
        outputs["stereo_T"] = torch.from_numpy(stereo_T)

        return outputs


class KITTI_Depth_Test(torch.utils.data.Dataset):
    def __init__(self, raw_pth, out_size=(256, 832), nums=None):
        super(KITTI_Depth_Test, self).__init__()
        self._name = 'KITTI_Depth_Test'
        self.out_size = out_size
        self.raw_pth = raw_pth
        self.gt_depths = np.load('./data/prepare/gt_depths.npz', allow_pickle=True)['data']
        self.image_list, self.cam_list = self.collect_test_file('./data/prepare/test_files.txt')

        if nums is not None:
            star = 680
            end = np.minimum(star + nums, len(self.image_list))
            self.image_list = self.image_list[star:end]
            self.cam_list = self.cam_list[star:end]
            self.gt_depths = self.gt_depths[star:end]
    
    def collect_test_file(self, file_name):
        with open(file_name) as f:
            lines = f.readlines()

        image_lists, cam_lists = [], []
        for line in lines:
            path, idx, _ = line.strip().split(' ')
            image_lists.append(os.path.join(self.raw_pth, path, 'image_02/data/' + idx + '.png'))
            path.split('/')[0]
            cam_lists.append(os.path.join(self.raw_pth, path.split('/')[0], 'calib_cam_to_cam.txt'))
        return image_lists, cam_lists

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        H, W, _ = image.shape
        H_out, W_out = self.out_size
        image = cv2.resize(image, (W_out, H_out)) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        gt_depth = torch.from_numpy(self.gt_depths[idx]).float()
        return image, gt_depth

if __name__ == '__main__':
    pass
