import os
import numpy as np
import torch
import cv2
import collections
from tqdm import tqdm

def calc_error_rate(epe_map, gt_flow, mask):
    bad_pixels = np.logical_and(
        epe_map * mask > 3,
        epe_map * mask / np.maximum( np.sqrt(np.sum(np.square(gt_flow), axis=2)), 1e-10) > 0.05)
    return bad_pixels.sum() / mask.sum()

def eval_flow(flow, gt_flow, noc_mask, moving_mask=None):
    epe_map = np.sqrt( np.sum(np.square(flow[:, :, 0:2] - gt_flow[:, :, 0:2]), axis=2) )

    error = np.sum(epe_map * gt_flow[:, :, 2]) / np.sum(gt_flow[:, :, 2])
    error_noc = np.sum(epe_map * noc_mask) / np.sum(noc_mask)
    error_occ = np.sum(epe_map * (gt_flow[:, :, 2] - noc_mask)) / max(np.sum(gt_flow[:, :, 2] - noc_mask), 1.0)
    error_rate = calc_error_rate(epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2])

    error_move_rate, error_static_rate, error_move, error_static = None, None, None, None
    if moving_mask is not None:
        error_move_rate = calc_error_rate( epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * moving_mask)
        error_static_rate = calc_error_rate( epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * (1.0 - moving_mask))
        error_move = np.sum(epe_map * gt_flow[:, :, 2] * moving_mask) / np.sum(gt_flow[:, :, 2] * moving_mask)
        error_static = np.sum(epe_map * gt_flow[:, :, 2] * ( 1.0 - moving_mask)) / np.sum(gt_flow[:, :, 2] * (1.0 - moving_mask))

    error_pack = collections.OrderedDict()
    error_pack['err'] = error
    error_pack['Noc'] = error_noc
    error_pack['Occ'] = error_occ
    error_pack['Rate'] = error_rate
    error_pack['Move'] = error_move
    error_pack['Static'] = error_static
    error_pack['MoveRate'] = error_move_rate
    error_pack['StaticRate'] = error_static_rate
    
    return error_pack

def extra_data_show(logger, folder, data_pack, idx, out_size):
    def np2im(M):
        return (M - np.min(M)) / (np.max(M) - np.min(M)) * 255

    for k, v in data_pack.items():
        prefix = str(idx).zfill(6) + '_' + k
        im = v[0][0].detach().cpu().numpy().transpose(1, 2, 0)
        W, H = out_size
        in_H, in_W, C = im.shape
        im = cv2.resize(im, out_size, interpolation=cv2.INTER_LINEAR)
        if C == 2:
            im[:, :, 0] = im[:, :, 0] / in_W * W
            im[:, :, 1] = im[:, :, 1] / in_H * H
            logger.write_flow(im, folder, prefix=prefix)
        else:
            logger.write_image(np2im(im), folder, prefix=prefix)

@torch.no_grad()
def eval_kitti_flow(net, datasets, logger, cfgs, write=True):
    device = cfgs['device']
    net = net.eval()
    for dataset in datasets:
        if 'KITTI_Flow' not in dataset._name:
            continue

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        print('START %s FLOW EVALUATION ...' % (dataset._name))

        evals = []
        for i, data in tqdm(enumerate(dataloader)):
            image1 = data[0].to(device)
            image2 = data[1].to(device)
            occ = data[2].to(device).cpu().numpy()[0]
            noc_mask = data[3].to(device).cpu().numpy()[0]
            gt_mask = data[4].to(device).cpu().numpy()[0] if '2015' in dataset._name else None

            flows, extra_data = net.infer_flow(image1, image2)
            
            # process flow
            flow = flows[0][0].detach().cpu().numpy().transpose(1,2,0)
            H, W, _ = occ.shape
            in_H, in_W, _ = flow.shape
            flow[:, :, 0] = flow[:, :, 0] / in_W * W
            flow[:, :, 1] = flow[:, :, 1] / in_H * H
            flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)

            # eval flow
            evals.append(eval_flow(flow, occ, noc_mask, moving_mask=gt_mask))

            # extra_data visual
            if write:
                extra_data_show(logger, folder='extra_data', data_pack=extra_data, idx=i, out_size=(W, H))
                logger.write_flow(flow, folder='flow', prefix=str(i).zfill(6))
                logger.write_flow(occ[:,:,0:2], folder='gt', prefix=str(i).zfill(6))
                logger.write_flow((flow - occ[:,:,0:2]) * occ[:,:,2:3], folder='flow_diff', prefix=str(i).zfill(6))

        logger.print_eval(evals, prefix=dataset._name, write=True)
        print('END!')

if __name__ == "__main__":
    pass