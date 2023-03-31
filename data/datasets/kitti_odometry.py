import os
import numpy as np
from PIL import Image
import glob
import cv2
import torch
import torchvision

from data.datasets.utils import *

class KITTI_Odometry(torch.utils.data.Dataset):
    '''
    KITTI odometry color dataset
    data_path/pose/00.txt - 10.txt
    data_path/sequences/[00 - 10]/[image_2, image_3, calib.txt, times.txt]/[000000.png - xxxxxx.png]
    '''
    def __init__(self, path, sequence, train_nums=2, stride=1, eval_nums=None, out_size=(256, 832)):
        super(KITTI_Odometry, self).__init__()
        self._name = 'KITTI_Odometry'
        self.out_size = out_size
        self.root = path
        self.train_nums = train_nums
        self.stride = stride
        self.eval_nums = eval_nums
        self.save_prefix = ''

        self.seqs = ['train', '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        assert sequence in self.seqs
        self.sequences = self.seqs[1:10] if sequence == 'train' else [sequence]

        self.frame_ids = np.arange(2) if train_nums == 2 else np.arange(train_nums) - 1
        self.image_list = self.collect_scenes()
        self.image_list = self.image_list[0::stride]

        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        self.fn_resize = torchvision.transforms.Resize(self.out_size)
        self.fn_color_aug = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.to_tensor = torchvision.transforms.ToTensor()

        self.do_flip = True
        if sequence != 'train':
            self.gt_pose_file = os.path.join(path, 'poses', sequence + '.txt')
            self.do_flip = False
            if eval_nums is not None:
                star = 520
                end = np.minimum(star + eval_nums, len(self.image_list))
                self.image_list = self.image_list[star:end]
    
    def collect_scenes(self):
        scenes_list = [os.path.join(self.root, 'sequences', q_) for q_ in self.sequences]
        train_list = []
        for scene in scenes_list:
            cam_file = os.path.join(scene, 'calib.txt')
            images = sorted(glob.glob(os.path.join(scene, 'image_2', '*.png')))

            # get image pair
            iids = np.arange(0, len(images), 1)
            iids = np.reshape(iids, (1, iids.shape[0]))
            for i in range(self.train_nums-1):
                iids = np.concatenate([iids, np.roll(iids[-1:, :], -self.stride)], axis=0)
            iids = iids[:, :(1-self.train_nums) * self.stride]

            # to list
            for i in np.transpose(iids, (1, 0)):
                str_ = ''
                for j in i:
                    str_ = str_ + images[j].replace(self.root, '')[1:] + ' '
                str_ = str_ + cam_file.replace(self.root, '')[1:]
                train_list.append(str_)
        return train_list

    def __len__(self):
        return len(self.image_list)

    def pre_process(self, outputs, image_lists, do_flip):
        # step 1: load resize, flip and color transforms
        for i, p in enumerate(image_lists):
            # load left
            filename_left = os.path.join(self.root, p)
            im_left = load_rgb_from_file(filename_left, do_flip)
            W, H = im_left.size
            im_left = self.fn_resize(im_left)
            
            outputs[('color', self.frame_ids[i])] = self.to_tensor(im_left)

            # load transforms left
            outputs[('color_aug', self.frame_ids[i])] = self.to_tensor(self.fn_color_aug(im_left))

            # load right
            if self.frame_ids[i] == 0:
                filename_right = os.path.join(self.root, p.replace('image_2', 'image_3'))
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
        data = self.image_list[idx].split(' ')
        image_lists, cam_file = data[:-1], data[-1]

        # load image seqs
        outputs = {}
        do_flip = np.random.rand() > 0.5 if self.do_flip else False
        in_size = self.pre_process(outputs, image_lists, do_flip)

        # load camera intrinsics
        cam_file = os.path.join(self.root, cam_file)
        outputs['K'] = load_cam_intrinsic(cam_file, in_size, self.out_size, position='P2')
        outputs['K_inv'] = outputs['K'].inverse()
        
        # load camera extrinsics of right camera
        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = -1 if do_flip else 1
        stereo_T[0, 3] = -1 * baseline_sign * 0.015
        outputs["stereo_T"] = torch.from_numpy(stereo_T)

        return outputs

class KITTI_OdometryEx(torch.utils.data.Dataset):
    '''
    KITTI odometry color dataset
    data_path/sequences_***/[09 - 10]/image_2/[000000.png - xxxxxx.png]
    mode = [g1-g3, s2-s3, p1-p6]
    '''
    def __init__(self, path, mode='g1', sequence='09', out_size=(256, 832)):
        super(KITTI_OdometryEx, self).__init__()
        self._name = 'KITTI_OdometryEx'
        self.path = path
        self.out_size = out_size
        self.eval_nums = None
        self.sequences = [sequence]
        self.stride = 1

        assert sequence in ['09', '10']
        seq_mode = ['sequences_Geometric_Noise', 'sequences_Low_FPS', 'sequences_Photometric_Noise']
        if mode[0] == 'g':
            scene_name = os.path.join(path, seq_mode[0] + '_' + mode, sequence)
        elif mode[0] == 's':
            scene_name = os.path.join(path, seq_mode[1] + '_' + mode, sequence)
            self.stride = int(mode[1])
        elif mode[0] == 'p':
            scene_name = os.path.join(path, seq_mode[2] + '_' + mode, sequence)
        else:
            raise ValueError
        self.save_prefix = '_' + scene_name.split('/')[-2].replace('sequences_', '')

        self.image_list = self.collect_scenes(scene_name)
        self.gt_pose_file = os.path.join(path, 'poses', sequence + '.txt')
        self.frame_ids = np.arange(2)

        self.fn_resize = torchvision.transforms.Resize(out_size, interpolation=Image.ANTIALIAS)
        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.image_list)
    
    def collect_scenes(self, scene):
        train_list = []
        
        cam_file = os.path.join(scene, 'calib.txt')
        images = sorted(glob.glob(os.path.join(scene, 'image_2', '*.png')))

        # get image pair
        iids = np.arange(0, len(images), 1)
        iids = np.reshape(iids, (1, iids.shape[0]))
        iids = np.concatenate([iids, np.roll(iids[-1:, :], -1)], axis=0)
        iids = iids[:, :-1]

        # to list
        for i in np.transpose(iids, (1, 0)):
            str_ = ''
            for j in i:
                str_ = str_ + images[j].replace(self.path, '')[1:] + ' '
            str_ = str_ + cam_file.replace(self.path, '')[1:]
            train_list.append(str_)
        return train_list

    def __getitem__(self, idx):
        data = self.image_list[idx].split(' ')
        image_lists, cam_file = data[:-1], data[-1]

        # load image seqs
        outputs = {}
        for i, imname in enumerate(image_lists):
            im = load_rgb_from_file(os.path.join(self.path, imname), False)
            W, H = im.size
            im = self.fn_resize(im)
            outputs[('color', self.frame_ids[i])] = self.to_tensor(im)

        # load camera intrinsics
        cam_file = os.path.join(self.path, cam_file)
        outputs['K'] = load_cam_intrinsic(cam_file, (H, W), self.out_size, position='P2')
        outputs['K_inv'] = outputs['K'].inverse()

        return outputs

if __name__ == '__main__':
    pass

