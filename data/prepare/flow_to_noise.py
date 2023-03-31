import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import glob

import cv2
from tqdm import tqdm
from multiprocessing import Pool

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

class ToNoise(object):
    def __init__(self, root, config):
        self.root = root
        self.scenes_list = [root]
        self.config = config

    def creat_gaussian_kernel(self, sigma):
        r = int(np.round(sigma * 2.5))
        if r == 0:
            kernel = np.array(1.0)
        else:
            x = np.linspace(-r, r, r*2+1, dtype=np.float)
            y = np.linspace(-r, r, r*2+1, dtype=np.float)
            xx, yy = np.meshgrid(x, y)
            # kernel = (1 / (sigma * np.sqrt(2*3.1415))) * np.exp(-(xx**2 + yy**2) / (2*sigma * sigma))
            kernel = (1 / (sigma**2 * 2*3.141592)) * np.exp(-(xx**2 + yy**2) / (2*sigma * sigma))
            # kernel = np.exp(-(xx**2 + yy**2) / (2*sigma * sigma))
            kernel = kernel / np.sum(kernel)
        return kernel, r
    
    def point_gaussian(self, data_pack):
        im, grid_noise = data_pack
        assert im.shape[0:2] == grid_noise.shape
        H, W = grid_noise.shape

        max_r = int(np.ceil(np.max(grid_noise) * 2.5))
        img_pad = cv2.copyMakeBorder(im, max_r, max_r, max_r, max_r, cv2.BORDER_REPLICATE)
        for row in range(H):
            for col in range(W):
                row_pad = row + max_r
                col_pad = col + max_r

                sigma = grid_noise[row][col]
                kernel2d, r = self.creat_gaussian_kernel(sigma)
                im[row, col, 0] = np.sum(img_pad[row_pad-r:row_pad+r+1, col_pad-r:col_pad+r+1, 0] * kernel2d)
                im[row, col, 1] = np.sum(img_pad[row_pad-r:row_pad+r+1, col_pad-r:col_pad+r+1, 1] * kernel2d)
                im[row, col, 2] = np.sum(img_pad[row_pad-r:row_pad+r+1, col_pad-r:col_pad+r+1, 2] * kernel2d)
        return im

    def collect_Photometric_Noise(self, delta):
        # delta = self.config['delta_p']
        name = 'sequences_Photometric_Noise' + '_p' + str(delta)
        for scene in self.scenes_list:
            images_file = sorted(glob.glob(os.path.join(scene, 'image_2', '*.png')))

            # mkdir
            out_scene = os.path.join(scene.replace('training', name), 'image_2')
            if not os.path.exists(out_scene):
                os.makedirs(out_scene)

            # cp images
            for imgname in tqdm(images_file):
                im = cv2.imread(imgname) / 255.0
                H, W, _ = im.shape

                grid_noise = np.random.random((150, 600)) * delta
                grid_noise = cv2.resize(grid_noise, (W, H), interpolation=cv2.INTER_LINEAR)
                grid_noise = np.clip(grid_noise, 0, delta)

                # multiprocess
                nhsplit, nvsplit = 1, 6
                im_split = [np.array_split(vi, nvsplit, axis=1) for vi in np.vsplit(im, nhsplit)]
                gn_split = [np.array_split(vi, nvsplit, axis=1) for vi in np.vsplit(grid_noise, nhsplit)]
                datas = []
                for i in range(nhsplit):
                    for j in range(nvsplit):
                        datas.append([ im_split[i][j], gn_split[i][j] ])

                with Pool(nhsplit*nvsplit) as p:
                    ims = p.map(self.point_gaussian, datas)
                im = np.concatenate([ np.concatenate(ims[i:i+nvsplit], axis=1) for i, v in enumerate(ims) if i%nvsplit==0], axis=0)
                cv2.imwrite(imgname.replace('training', name), im * 255)

    def collect_Geometric_Noise(self, delta):
        # delta = self.config['delta_g']
        name = 'sequences_Geometric_Noise' + '_g' + str(delta)
        for scene in self.scenes_list:
            images_file = sorted(glob.glob(os.path.join(scene, 'image_2', '*.png')))


            # mkdir
            out_scene = os.path.join(scene.replace('training', name), 'image_2')
            if not os.path.exists(out_scene):
                os.makedirs(out_scene)

            # cp images
            for imgname in tqdm(images_file):
                im = torch.from_numpy(cv2.imread(imgname)).permute(2, 0, 1).unsqueeze(0).float()
                _, _, H, W = im.shape

                grid_noise = torch.zeros([1, 1, 3, 3]).uniform_(-delta, delta)
                grid_noise = F.interpolate(grid_noise, (H, W), mode='bicubic', align_corners=True).clamp(min=-delta, max=delta)
                grid = (meshgrid(im.shape, norm=False).float() + grid_noise).permute(0,2,3,1)
                grid = torch.cat([ 2.0 * grid[:, :, :, 0:1] / (W - 1.0) - 1, 2.0 * grid[:, :, :, 1:2] / (H - 1.0) - 1], 3)
                im_noise = F.grid_sample(im, grid, mode='bilinear', align_corners=True)
                cv2.imwrite(imgname.replace('training', name), im_noise[0].permute(1,2,0).numpy())
            

    def collect_lowFPS(self, stride):
        # stride = self.config['delta_s']
        name = 'sequences_Low_FPS' + '_s' + str(stride)
        for scene in self.scenes_list:
            images_file = sorted(glob.glob(os.path.join(scene, 'image_2', '*_10.png')))

            # mkdir
            out_scene = os.path.join(scene.replace('training', name), 'image_2')
            if not os.path.exists(out_scene):
                os.makedirs(out_scene)

            # cp images
            src_image_file = [fim for i, fim in enumerate(images_file) if i%stride == 0]
            for ftim in tqdm(src_image_file):
                os.system('cp ' + ftim + ' ' + out_scene)
                os.system('cp ' + ftim.replace('_10', '_11') + ' ' + out_scene)
    
    def run(self):
        # for s in self.config['delta_s']:
        #     self.collect_lowFPS(s)
        for g in self.config['delta_g']:
            self.collect_Geometric_Noise(g)
        # for p in self.config['delta_p']:
        #     self.collect_Photometric_Noise(p)
            

        

def main():
    path = '/home/cyj/datasets/KITTI/2015/training'
    config = {}
    # config['delta_s'] = [2, 3]
    config['delta_g'] = [3]
    config['delta_p'] = [6,5,4,3,2,1]
    
    noise = ToNoise(path, config)
    noise.run()

if __name__ == '__main__':
    main()
    





