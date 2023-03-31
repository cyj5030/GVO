import os
import sys
import numpy as np
import glob

import imageio
from tqdm import tqdm
# import torch.multiprocessing as mp


class KITTI_RAW(object):
    def __init__(self, in_dir, out_dir, static_frames_txt, test_scenes_txt):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.test_scenes = self.collect_test_scenes(test_scenes_txt)
        self.static_frames = self.collect_static_frame(static_frames_txt)

    def __len__(self):
        raise NotImplementedError

    def collect_scenes(self):
        train_scenes = {}
        date_list = os.listdir(self.in_dir)
        for date in date_list:
            if not os.path.isdir(os.path.join(self.in_dir, date)):
                continue
            scene_list = os.listdir(os.path.join(self.in_dir, date))
            for scene in scene_list:
                scene_path = os.path.join(self.in_dir, date, scene)
                if not os.path.isdir(scene_path):
                    continue
                if scene in self.test_scenes:
                    continue
                image_list = glob.glob(os.path.join(scene_path, 'image_02', 'data', '*.png'))
                image_list = sorted(list(set(image_list).difference(set(self.static_frames))))
                if len(image_list) > 2:
                    train_scenes[os.path.join(date, scene)] = image_list
        return train_scenes

    def collect_static_frame(self, static_frames_txt):
        with open(static_frames_txt) as f:
            lines = f.readlines()
        static_frames = []
        for line in lines:
            line = line.strip()
            date, drive, frame_id = line.split(' ')
            image = '%.10d.png' % (int(frame_id))
            static_frames.append(os.path.join(self.in_dir, date, drive, 'image_02', 'data', image))
        return static_frames
    
    def collect_test_scenes(self, test_scenes_txt):
        with open(test_scenes_txt) as f:
            lines = f.readlines()
        test_scenes = []
        for line in lines:
            line = line.strip()
            test_scenes.append(line + '_sync')
        return test_scenes

    def run(self):
        train_scenes = self.collect_scenes()

        for scene, image_list in tqdm(train_scenes.items()):
            src_cam = os.path.join(self.in_dir, scene[:10], 'calib_cam_to_cam.txt')
            target_cam = os.path.join(self.out_dir, scene[:10], 'calib_cam_to_cam.txt')

            target_path = os.path.join(self.out_dir, scene)
            if not os.path.isdir(target_path):
                os.makedirs(os.path.join(target_path, 'image_02'))
                os.makedirs(os.path.join(target_path, 'image_03'))

            command = []
            command.append('cp ' + src_cam + ' ' + target_cam)
            for src_image in image_list:
                target_image = os.path.join(target_path, 'image_02', os.path.basename(src_image))
                command.append('cp ' + src_image + ' ' + target_image)

                target_image = os.path.join(target_path, 'image_03', os.path.basename(src_image))
                command.append('cp ' + src_image + ' ' + target_image)

            for c in command:
                os.system(c)
    
    def __getitem__(self, idx):
        raise NotImplementedError

def main(args):
    in_path = args.raw_path
    out_path = args.out_path
    static_frames_txt = './static_frames.txt'
    test_scenes_txt = './test_scenes.txt'
    kitti_raw_dataset = KITTI_RAW(in_path, out_path, static_frames_txt, test_scenes_txt)
    kitti_raw_dataset.run()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parser for all scripts.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-raw_path', type=str, default='/home/cyj/datasets/KITTI/raw')
    parser.add_argument('-out_path', type=str, default='/home/cyj/datasets/KITTI/train_test')
    args = parser.parse_args()

    main(args)
    





