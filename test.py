import utils
import model
import logger
import os
import torch
import numpy as np

from data.datasets import load_kitti, load_kaist
import evaluation as Eval

def tester(net, datasets, log, cfgs):
    device = cfgs['device']

    # load parameters
    state, model_params = log.load_from_ckpt(cfgs["ckpt_file"])
    net.load_state_dict(model_params, strict=True)
    net.to(device)
    print('Load params finished.')

    # tset flow
    # Eval.eval_kitti_flow(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs, write=False)
    
    # eigen depth
    # Eval.eval_kitti_depth(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs)

    # test odometry
    Eval.eval_kitti_odometry(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs, kpts=cfgs['num_keypoints'], write=False)

    # Eval.eval_kaist_odometry(net=net, datasets=datasets['test'], logger=log, cfgs=cfgs, kpts=cfgs['num_keypoints'], write=False)

def run(cfgs):
    # init logger
    log = logger.Logger(save_path=cfgs['log_path'], prefix=cfgs['log_name'])
    log.print_config(cfgs, write=True)

    # loading dataset
    print('Loading dataset...')
    if cfgs['dataset'] == 'kitti':
        datasets = load_kitti(cfgs)
    elif cfgs['dataset'] == 'kaist':
        datasets = load_kaist(cfgs)

    # model
    print('Loading model...')
    net = model.model_loader(cfgs)

    tester(net, datasets, log, cfgs)

def main(args):
    config = utils.ConfigFromFile('./config/kitti.yaml')
    config['ckpt_file'] = args["ckpt_file"]
    config['log_name'] = args["output_file"]
    config['model'] = args["model"]
    config['device'] = args["device"]
    config['Train_Eval'] = args["Train_Eval"]
    run(config)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parser for all scripts.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ckpt_file', type=str, default='/home/cyj/code/GVO/checkpoints/vo_Log/vo.pth')
    parser.add_argument('-output_file', type=str, default='test_depth')
    parser.add_argument('-model', type=str, default='depth')
    parser.add_argument('-device', type=str, default='cuda:3')
    parser.add_argument('-Train_Eval', type=str, default='vo')
    args = vars(parser.parse_args())

    main(args)