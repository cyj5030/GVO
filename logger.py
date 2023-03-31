import os
import torch
import cv2
import glob
import numpy as np
import shutil
import datetime
import collections
from evaluation import flow_to_image
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as Image
from matplotlib import pyplot as plt

class Logger():
    def __init__(self, save_path='./checkpoints', prefix='VO', reuse=False, reuse_root=None):
        self.save_path = save_path
        self.root = os.path.join(save_path, prefix + '_Log')
        self.mkdir(self.root)

        if prefix != reuse_root:
            shutil.rmtree(self.root)
            self.mkdir(self.root)

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def write_image(self, images, folder, prefix):
        root = os.path.join(self.root, folder)
        self.mkdir(root)
        cv2.imwrite( os.path.join(root, prefix + '.png'), images)
    
    def write_flow(self, flow, folder, prefix):
        root = os.path.join(self.root, folder)
        self.mkdir(root)
        cv2.imwrite( os.path.join(root, prefix + '.png'), flow_to_image(flow))
    
    def write_disp(self, disp, folder, prefix):
        root = os.path.join(self.root, folder)
        self.mkdir(root)
        vmax = np.percentile(disp, 95)
        normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp)[:,:,:3] * 255).astype(np.uint8)
        im = Image.fromarray(colormapped_im)
        
        im.save(os.path.join(root, prefix + '.png'))

    def save_traj(self, poses, prefix):
        """ poses: list of global poses """
        f = open(os.path.join(self.root, prefix + '.txt'), 'w')
        for i in range(len(poses)):
            pose = poses[i][0:3, :].flatten()[:12] # [3x4]
            line = " ".join([str(j) for j in pose])
            f.write(line + '\n')
        f.close()
        return os.path.join(self.root, prefix + '.txt')

    def save_traj_tmu(self, poses, time_stamp, prefix, order='wxyz'):
        """ poses: list of global poses """
        f = open(os.path.join(self.root, prefix + '.txt'), 'w')
        f.write('#time_stampe x y z qw qx qy qz' + '\n')
        for i in range(len(poses)):
            time = time_stamp[i]
            pose = poses[i]
            pose = pose[[0,1,2,4,5,6,3]] if order == 'xyzw' else pose
            line = str(time) + ' ' + " ".join([str(j) for j in pose])
            f.write(line + '\n')
        f.close()
        return os.path.join(self.root, prefix + '.txt')

    def plotPath(self, seq, poses_gt, poses_result, axis_y='z'):
        plot_path = os.path.join(self.root, 'plot_path')
        self.mkdir(plot_path)

        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        plot_num =-1

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xz = []
            # for pose in poses_dict[key]:
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                ax_y = 2 if axis_y == 'z' else 1
                pos_xz.append([pose[0,3],  pose[ax_y,3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:,0],  pos_xz[:,1], label = key)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel(axis_y + ' (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_"+(seq)
        plt.savefig(os.path.join(plot_path, png_title + '.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.show()

    def print_losses(self, losses_dict, state_dict, write=False):
        now = datetime.datetime.now()
        print_line = '[%s: epoch:%d, iter:%d]' % (now.strftime('%Y-%m-%d %H:%M:%S'), state_dict['epoch'], state_dict['itr'])
        for k, v in losses_dict.items():
            print_line = print_line + ' | %s: %.3f |' % (k, v)
        print(print_line)
        
        if write:
            with open(os.path.join(self.root, 'log.txt'), 'a') as f:
                f.write(print_line + '\n')
                
    def print_config(self, config, write=False):
        lines = ''
        for _key, _value in config.items():
            lines = lines + _key + ': ' + str(_value) + '\n'
        print(lines)

        if write:
            with open(os.path.join(self.root, 'config.txt'), 'w') as f:
                f.write(lines)

    def print_eval_pose(self, error, prefix, write=False):
        # head
        str_line = []
        str_line.append('{:>10}, '.format('index'))
        for k in error.keys():
            str_line[-1] += '{:>30}, '.format(k)
        
        str_line.append('{:>10}, '.format('average'))
        for v in error.values():
            str_line[-1] += '{:30.4f}, '.format(v)
        print(str_line[0])
        print(str_line[-1])

        if write:
            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(os.path.join(self.root, prefix + '_eval.txt'), 'a') as f:
                f.writelines([now_time + '\n', str_line[0] + '\n', str_line[-1] + '\n', '\n' ])

    def print_eval(self, error_lists, prefix, write=False):
        # each element is a dict
        length = len(error_lists)

        # head
        str_line = []
        val = collections.OrderedDict()
        str_line.append('{:>10}, '.format('index'))
        for k in error_lists[0].keys():
            str_line[-1] += '{:>10}, '.format(k)
            val[k] = 0

        # each image
        for i, item in enumerate(error_lists):
            str_line.append('{:10d}, '.format(i))
            for k, v in item.items():
                str_line[-1] += '{:10.4f}, '.format(v) if v is not None else 'None'
                val[k] += v / length if v is not None else 0
        
        # avg
        str_line.append('{:>10}, '.format('average'))
        for k, v in val.items():
            str_line[-1] += '{:10.4f}, '.format(v)
        print(str_line[0])
        print(str_line[-1])

        if write:
            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' -- num_item = ' + str(length)
            with open(os.path.join(self.root, prefix + '_eval.txt'), 'a') as f:
                f.writelines([now_time + '\n', str_line[0] + '\n', str_line[-1] + '\n', '\n' ])

            with open(os.path.join(self.root, prefix + '_eval_all.txt'), 'w') as f:
                f.writelines([c + '\n' for c in str_line])

    def save(self, net, state, folder):
        self.mkdir(os.path.join(self.root, folder))
        prefix = 'checkpoint_' + str(state['itr']).zfill(6)
        torch.save({'state': state, 'model': net.state_dict()},
                    os.path.join(self.root, folder, prefix + '.pth'))
    
    def load(self, folder, device, checkpoints=None):
        model_list = sorted(glob.glob(os.path.join(self.save_path, folder + '_Log', 'model', '*.pth')))
        path = model_list[-1] if checkpoints is None else os.path.join(self.save_path, folder + '_Log', 'model', 'checkpoint_' + str(checkpoints).zfill(6) + '.pth')
        state = torch.load(path, map_location=device)
        return state['state'], state['model']
    
    def load_from_ckpt(self, ckpt):
        state = torch.load(ckpt, map_location="cpu")
        return state['state'], state['model']

if __name__ == "__main__":
    pass    