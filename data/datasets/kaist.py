import os

import numpy as np
from PIL import Image
import glob
import cv2
import torch
import torchvision
import copy
import collections
# from data.datasets.lie_algebra import SO3_CPU

from data.datasets.utils import *
# from utils import *

KAIST_Train = [
    'urban26-dongtan',
    'urban29-pankyo',
    'urban30-gangnam',
    'urban32-yeouido',
    'urban34-yeouido',
    'urban37-seoul',
    'urban38-pankyo',
]

KAIST_Test = [
    'urban27-dongtan',
    'urban28-pankyo',
    'urban31-gangnam',
    'urban33-yeouido',
    'urban35-seoul',
    'urban36-seoul',
    'urban39-pankyo'
]

statitcs_frame = {
    'urban18-highway': [],
    'urban19-highway': [],
    'urban20-highway': [],
    'urban21-highway': [],
    'urban22-highway': [],
    'urban23-highway': [],
    'urban24-highway': [],
    'urban25-highway': [],
    'urban26-dongtan': [['1544580706215305091.png', '1544580715415400739.png'],
                        ['1544580757815642291.png', '1544580841516880220.png'],
                        ['1544580870717126462.png', '1544580881117141310.png'],
                        ['1544581046519016318.png', '1544581123220478229.png']],

    'urban27-dongtan': [['1544582680935957336.png', '1544582805337195649.png'],
                        ['1544582884140504640.png', '1544582985439452557.png'],
                        ['1544583165341459212.png', '1544583203341641878.png'],
                        ['1544583232441925341.png', '1544583256542195648.png'],
                        ['1544583296042567094.png', '1544583299642678694.png'],
                        ['1544583380143339358.png', '1544583386943580627.png'],
                        ['1544583428543834488.png', '1544583556645266664.png'],
                        ['1544583592645520155.png', '1544583695246396479.png'],
                        ['1544583726348797733.png', '1544583749746820893.png']],

    'urban28-pankyo':  [['1544590798702415701.png', '1544590812600182253.png'],
                        ['1544590849300711245.png', '1544590867301043366.png'],
                        ['1544591004803489254.png', '1544591037603931608.png'],
                        ['1544591077604633630.png', '1544591118905409619.png'],
                        ['1544591178206480984.png', '1544591281408038718.png'],
                        ['1544591311108433732.png', '1544591330508412915.png'],
                        ['1544591434910523305.png', '1544591445810456201.png'],
                        ['1544591462210919155.png', '1544591492511327656.png'],
                        ['1544591518813063555.png', '1544591528912367925.png'],
                        ['1544591636213302550.png', '1544591716914696895.png'],
                        ['1544591932318401245.png', '1544591945718146652.png'],
                        ['1544591988318695197.png', '1544592056119626700.png'],
                        ['1544592113520343730.png', '1544592183921534741.png'],
                        ['1544592380924830850.png', '1544592491025985233.png'],
                        ['1544592582627366255.png', '1544592641428267605.png'],],

    'urban29-pankyo':  [['1544594866260664638.png', '1544594882560734830.png'],
                        ['1544595100763720740.png', '1544595208865191870.png'],
                        ['1544595263365987035.png', '1544595269366345472.png']],

    'urban30-gangnam': [['1544676855830694269.png', '1544676880931150726.png'],
                        ['1544676904131283525.png', '1544676923631584113.png'],
                        ['1544676931831736904.png', '1544677015833130890.png'],
                        ['1544677207635615610.png', '1544677252736264402.png'],
                        ['1544677295936984092.png', '1544677314037383720.png'],
                        ['1544677331237614883.png', '1544677442439218885.png'],
                        ['1544677521441757815.png', '1544677597941346846.png'],
                        ['1544677735843707470.png', '1544677785544182402.png'],
                        ['1544677848945182270.png', '1544677871547032394.png'],
                        ['1544677931646733754.png', '1544677956146569363.png']],

    'urban31-gangnam': [['1544679067163065836.png', '1544679092563461249.png'],
                        ['1544679337766869915.png', '1544679406267982131.png']],

    'urban32-yeouido': [['1544680093077228343.png', '1544680116977851132.png'],
                        ['1544680179478564113.png', '1544680239179446772.png'],
                        ['1544680270380154952.png', '1544680298380558174.png'],
                        ['1544680508982925131.png', '1544680551983723254.png'],
                        ['1544680591984123890.png', '1544680600084400074.png'],
                        ['1544680632284643639.png', '1544680728586127362.png'],
                        ['1544680762387059035.png', '1544680776786690011.png'],
                        ['1544680862487666703.png', '1544680948788838965.png'],
                        ['1544681001889307699.png', '1544681033289606971.png'],
                        ['1544681066390149960.png', '1544681078290331374.png']],
    
    'urban33-yeouido': [['1544681140891186546.png', '1544681170093575614.png'],
                        ['1544681272092753970.png', '1544681292392969142.png'],
                        ['1544681349493910470.png', '1544681388494021765.png'],
                        ['1544681558896340519.png', '1544681603797138001.png'],
                        ['1544681682597793889.png', '1544681689598218257.png'],
                        ['1544681796099439032.png', '1544681826499563134.png'],
                        ['1544681939501282655.png', '1544682112503326295.png'],
                        ['1544682146003967681.png', '1544682166904294250.png'],
                        ['1544682261505032813.png', '1544682274905608339.png'],
                        ['1544682324906212634.png', '1544682342806325523.png']],
    
    'urban34-yeouido': [['1544682698811064490.png', '1544682716010734620.png'],
                        ['1544682911713547598.png', '1544682940013917599.png']],
    
    'urban35-seoul':   [],

    'urban36-seoul':   [],

    'urban37-seoul':   [],

    'urban38-pankyo':  [['1559193232373910975.png', '1559193241074110916.png'],
                        ['1559193276574705577.png', '1559193316075127269.png'],
                        ['1559193404576780132.png', '1559193424277016552.png'],
                        ['1559193437277226697.png', '1559193548579103909.png'],
                        ['1559193572179425462.png', '1559193590379512751.png'],
                        ['1559193712181313054.png', '1559193781382118288.png'],
                        ['1559193845583359513.png', '1559193862483625549.png'],
                        ['1559193974685438789.png', '1559194022686215145.png'],
                        ['1559194129388014823.png', '1559194165187966038.png'],
                        ['1559194373691087490.png', '1559194382592286572.png'],
                        ['1559194415991648697.png', '1559194457092212219.png'],
                        ['1559194503893015885.png', '1559194624994709119.png'],
                        ['1559194652295266098.png', '1559194699995707088.png'],
                        ['1559194795397375535.png', '1559194821897551586.png'],
                        ['1559194845797925269.png', '1559194865097995793.png'],
                        ['1559194876398367280.png', '1559194884598645916.png'],
                        ['1559194899398627234.png', '1559195007699942043.png'],
                        ['1559195040300625534.png', '1559195145402890577.png'],
                        ['1559195215803141779.png', '1559195255103579458.png'],
                        ['1559195341805010490.png', '1559195364705265681.png']],

    'urban39-pankyo':  [['1559195795611174799.png', '1559195806111307938.png'],
                        ['1559195861712328697.png', '1559195895612551430.png'],
                        ['1559195906312612778.png', '1559195924812852025.png'],
                        ['1559195944313042127.png', '1559195963213340387.png'],
                        ['1559196086515072228.png', '1559196112115587489.png'],
                        ['1559196224817029685.png', '1559196237717247872.png'],
                        ['1559196288118053229.png', '1559196297018236977.png'],
                        ['1559196349618796649.png', '1559196440319963785.png'],
                        ['1559196466920585680.png', '1559196477920605429.png'],
                        ['1559196486022180532.png', '1559196501021035891.png'],
                        ['1559196539921459071.png', '1559196582822069686.png'],
                        ['1559196601723001591.png', '1559196653323022190.png'],
                        ['1559196735724217032.png', '1559196793124907532.png'],
                        ['1559196886026586754.png', '1559196931127032728.png'],
                        ['1559196960327346681.png', '1559196982927587750.png'],
                        ['1559197033228400052.png', '1559197045328679625.png'],
                        ['1559197395233545082.png', '1559197405033929785.png'],
                        ['1559197472034869535.png', '1559197558235909204.png'],
                        ['1559197626636930806.png', '1559197662137509957.png'],]
}
class KAIST(torch.utils.data.Dataset):
    '''
    KAIST dataset
    '''
    def __init__(self, root, scene, eval_nums=10, out_size=(256, 512)):
        super(KAIST, self).__init__()
        self._name = 'KAIST_ODOMETRY'
        self.scene = scene
        self.out_size = out_size
        self.root = root
        self.eval_nums = eval_nums
        self.save_prefix = ''
        # self.so3 = SO3_CPU()

        self.image_list, self.time_stamp = self.collect_scenes(scene)
        if scene != 'train':
            self.gt_pose_file = os.path.join(self.root, scene, 'global_pose.csv')

        self.fn_resize = torchvision.transforms.Resize(self.out_size)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.do_flip = True if scene=='train' else False

        if eval_nums is not None:
            star = 0
            self.image_list = self.image_list[star:star+eval_nums]
            self.time_stamp = self.time_stamp[star:star+eval_nums]
    
    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self, idx):
        image_lists = self.image_list[idx].split(',')

        # load image seqs
        outputs = {}
        do_flip = np.random.rand() > 0.5 if self.do_flip else False
        K = self.pre_process(outputs, image_lists, do_flip)

        # load camera intrinsics
        outputs['K'] = torch.from_numpy(K).float()
        outputs['K_inv'] = outputs['K'].inverse().float()
        return outputs

    def collect_scenes(self, scene):
        images_list = []
        time_stamps = []
        if scene == 'train':
            images_list = preprocess(self.root)
        else:
            seq_file = os.path.join(self.root, scene)
            images = sorted(glob.glob(os.path.join(seq_file, 'image', 'stereo_left', '*.png')))
            time_stamps = [line.split('/')[-1][:-4] for line in images]

            # get image pair
            nums = len(images)
            for i in range(nums-1):
                images[i] = images[i] + ',' + images[i+1]

            images_list = images_list + images[:-1]
        return images_list, time_stamps

    def pre_process(self, outputs, image_lists, do_flip):
        # step 1: load resize, flip and color transforms
        for i, p in enumerate(image_lists):
            # load left
            try:
                im_left = load_bayer_from_file(p, do_flip)
            except:
                print(p)
            # calib =  yload(p.split('image')[0][:-1], 'calibration', 'left.yaml')
            cv_file = cv2.FileStorage(os.path.join(p.split('image')[0][:-1], 'calibration', 'left.yaml'), cv2.FILE_STORAGE_READ)
            H, W = cv_file.getNode("image_height").real(), cv_file.getNode("image_width").real()
            distortion = cv_file.getNode("distortion_coefficients").mat()
            K = cv_file.getNode("camera_matrix").mat()
            cv_file.release()

            new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, im_left.size, alpha=0)
            im_undistort = cv2.undistort(np.array(im_left), K, distortion, new_K)
            im_undistort = self.fn_resize(Image.fromarray(im_undistort))

            zoom_x, zoom_y = self.out_size[1] / W, self.out_size[0] / H
            new_K[0,:], new_K[1,:] = K[0,:] * zoom_x, K[1,:] * zoom_y

            outputs[('color', i)] = self.to_tensor(im_undistort)

        return new_K

def rotation_error(pose_error):
    a = pose_error[:, 0, 0]
    b = pose_error[:, 1, 1]
    c = pose_error[:, 2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(np.maximum(np.minimum(d,1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    dx = pose_error[:, 0, 3]
    dy = pose_error[:, 1, 3]
    dz = pose_error[:, 2, 3]
    return np.sqrt(dx**2+dy**2+dz**2)

def matching_time_indices(stamps_1: np.ndarray, stamps_2: np.ndarray,
                          max_diff: float = 0.01,
                          offset_2: float = 0.0):
    matching_indices_1 = []
    matching_indices_2 = []
    stamps_2 = copy.deepcopy(stamps_2)
    stamps_2 += offset_2
    for index_1, stamp_1 in enumerate(stamps_1):
        diffs = np.abs(stamps_2 - stamp_1)
        index_2 = int(np.argmin(diffs))
        if diffs[index_2] <= max_diff:
            matching_indices_1.append(index_1)
            matching_indices_2.append(index_2)
    return matching_indices_1, matching_indices_2

def find_index(str_list, str):
    matched_indexes = []
    i = 0
    length = len(str_list)
    
    while i < length:
        if str in str_list[i]:
            matched_indexes.append(i)
        i += 1
    assert len(matched_indexes) == 1
    return matched_indexes[0]

def get_pairs(str_list):
    if not str_list:
        return []
    nums = len(str_list)
    pairs = copy.deepcopy(str_list)
    for i in range(nums-1):
        pairs[i] = pairs[i] + ',' + pairs[i+1]
    return pairs[:-1]

def preprocess(root):
    total_pairs = []
    info = []
    nums = 0
    for key, value in statitcs_frame.items():
        if key not in KAIST_Train:
            continue

        imagefile = sorted(glob.glob(os.path.join(root, key, 'image', 'stereo_left', '*.png')))
        train_pairs = []
        if value:
            star_idx = 0
            for v in value:
                end_index = find_index(imagefile, v[0])
                np.array(imagefile[star_idx:end_index])
                train_pairs = train_pairs + get_pairs(imagefile[star_idx:end_index])
                star_idx = find_index(imagefile, v[1]) + 1
        else:
            train_pairs = get_pairs(imagefile)
        
        total_pairs = total_pairs + train_pairs
        nums += len(train_pairs)
        info.append('%s has %d num of pairs.\n' % (key, len(train_pairs)))
    info.append('total: %d.' % (nums))

    with open(os.path.join(root, 'train.txt'), 'w') as f:
        write_total_pairs = [name.replace(root, '') + '\n' for name in total_pairs]
        f.writelines(write_total_pairs)

    with open(os.path.join(root, 'info.txt'), 'w') as f:
        f.writelines(info)
    return total_pairs

if __name__ == "__main__":
    preprocess('/home/cyj/datasets/KAIST')