from .kitti_flow import KITTI_Flow_2012, KITTI_Flow_2015
from .kitti_raw import KITTI_Depth_Test, KITTI_Raw
from .kitti_odometry import KITTI_Odometry, KITTI_OdometryEx
from .kaist import KAIST, KAIST_Test

def load_kaist(cfgs):
    outputs = {}
    path = cfgs['kaist']
    size = (cfgs['height'], cfgs['width'])

    outputs['train'] = KAIST(path, 'train', eval_nums=None, out_size=size)
    outputs['val'] = [
            KAIST(path, KAIST_Test[0], eval_nums=10, out_size=size)
        ]
    nt = len(KAIST_Test)
    outputs['test'] = [KAIST(path, KAIST_Test[i], eval_nums=None, out_size=size) for i in range(nt)]
    return outputs

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
        outputs['all'] = [
            KITTI_Odometry(path_odo, '00', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '01', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '02', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '03', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '04', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '05', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '06', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '07', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '08', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '09', stride=stride, eval_nums=None, out_size=size),
            KITTI_Odometry(path_odo, '10', stride=stride, eval_nums=None, out_size=size),
        ]
    elif cfgs['Train_Eval'] == 'vo-robust':
        outputs['train'] = KITTI_Odometry(path_odo, 'train', train_nums=train_nums, stride=stride, eval_nums=None, out_size=size)
        outputs['val'] = [
            KITTI_Odometry(path_odo, '09', eval_nums=5, out_size=size)
        ]
        outputs['test'] = []
        modes = ['s2', 's3', 'g1', 'g2', 'g3', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']
        seqs = ['09', '10']
        for i in modes:
            for j in seqs:
                outputs['test'].append(KITTI_OdometryEx(path_odo, i, j, out_size=size))
    else:
        raise ValueError
    return outputs