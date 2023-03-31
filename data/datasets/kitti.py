import os

import numpy as np
from PIL import Image
import glob
import cv2
import torch
import torchvision

from data.datasets.utils import *
# from utils import *

def load_kitti(cfgs):
    outputs = {}
    path_prepare = cfgs['kitti_raw_prepare']
    path_raw = cfgs['kitti_raw']
    path_odo = cfgs['kitti_odometry']
    path_2012 = cfgs['kitti_flow_2012']
    path_2015 = cfgs['kitti_flow_2015']
    size = (cfgs['height'], cfgs['width'])
    train_nums = cfgs['train_length']
    stride = cfgs['train_stride']

    if cfgs['Train_Eval'] == 'flow':
        outputs['train'] = KITTI_Raw(path_prepare, num=train_nums, stride=stride, out_size=size)
        outputs['val'] = [
            KITTI_Flow_2015(path_2015, part='train', out_size=size, nums=10)
        ]
        outputs['test'] = [
            KITTI_Flow_2012(path_2012, part='train', out_size=size, nums=None),
            KITTI_Flow_2015(path_2015, part='train', out_size=size, nums=None)
        ]
    elif cfgs['Train_Eval'] == 'depth':
        outputs['train'] = KITTI_Raw(path_prepare, num=train_nums, stride=stride, out_size=size)
        outputs['val'] = [
            KITTI_Depth_Test(path_raw, out_size=size, nums=10),
        ]
        outputs['test'] = [
            KITTI_Depth_Test(path_raw, out_size=size, nums=None),
        ]
    elif cfgs['Train_Eval'] == 'vo':
        outputs['train'] = KITTI_Odometry(path_odo, 'train', train_nums=train_nums, stride=stride, eval_nums=None, out_size=size)
        outputs['val'] = [
            KITTI_Odometry(path_odo, '09', eval_nums=5, out_size=size)
        ]
        outputs['test'] = [
            KITTI_Odometry(path_odo, '09', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '10', stride=stride, eval_nums=None, out_size=size)
        ]
    elif cfgs['Train_Eval'] == 'vo-robust':
        outputs['train'] = KITTI_Odometry(path_odo, 'train', train_nums=train_nums, stride=stride, eval_nums=None, out_size=size)
        outputs['val'] = [
            KITTI_Odometry(path_odo, '09', eval_nums=5, out_size=size)
        ]
        outputs['test'] = [
            KITTI_Odometry(path_odo, '09', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '10', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '09', stride=1, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '10', stride=1, eval_nums=None, out_size=size),
        ]
    else:
        raise ValueError
    return outputs




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
        self.fn_resize = torchvision.transforms.Resize(self.out_size, interpolation=Image.ANTIALIAS)
        self.fn_color_aug = torchvision.transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)
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
    import time
    # dataset = KITTI_Odometry('/home/cyj/datasets/KITTI/data_odometry_color', 'train')
    dataset = KITTI_Depth_Test('/home/cyj/datasets/KITTI/raw')
    # dataset = KITTI_Raw('/home/cyj/dataset/kitti/kitti_train', num=3)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # ct = time.time()
    for i, data in enumerate(dataloader):
        image, _ = data
        torchvision.utils.save_image(image[:, [2,1,0], :, :], '/home/cyj/code/VO/debug/Eigen_images/' + str(i).zfill(6) + '.png')
        # print('done!')
        # duration = time.time() - ct
        # print(duration)
        # ct = time.time()
        # for key_, value_ in data.items():
        #     print(key_, value_.shape)
        # torchvision.utils.save_image(data['color', 0], '/home/cyj/code/python/VO/debug/color.png')
        # torchvision.utils.save_image(data['color_aug', 0], '/home/cyj/code/python/VO/debug/aug.png')
        # torchvision.utils.save_image(data['color', 'stereo'], '/home/cyj/code/python/VO/debug/color_s.png')
        # torchvision.utils.save_image(data['color_aug', 'stereo'], '/home/cyj/code/python/VO/debug/aug_s.png')
        # break

