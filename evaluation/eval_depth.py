import os
import torch
import numpy as np
import cv2
import collections

def process_depth(gt_depth, pred_depth, min_depth, max_depth):
    mask = gt_depth > 0
    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth
    gt_depth[gt_depth < min_depth] = min_depth
    gt_depth[gt_depth > max_depth] = max_depth
    return gt_depth, pred_depth, mask

def compute_errors(gt, pred, gt_disp=None, pred_disp=None):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))
    abs_rel = np.mean(np.abs(gt - pred) / (gt))
    sq_rel = np.mean(((gt - pred)**2) / (gt))

    d1_all = 0.0
    if gt_disp is not None and pred_disp is not None:
        mask = gt_disp > 0
        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all = 100.0 * bad_pixels.sum() / mask.sum()

    return abs_rel, sq_rel, rmse, rmse_log, log10, d1_all, a1, a2, a3

def eval_depth(gt_depth, pred_depth, gt_disp=None, pred_disp=None, min_depth=0.1, max_depth=80, kitti=True):
    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
    
    if kitti:
        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                        0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

    gt_depth = gt_depth[mask]
    pred_depth = pred_depth[mask]
    scale = np.median(gt_depth) / np.median(pred_depth)
    pred_depth *= scale

    gt_depth, pred_depth, _ = process_depth(gt_depth, pred_depth, min_depth, max_depth) 
    # [maskkk[mask]]
    abs_rel, sq_rel, rms, rms_log, log_10, d1_all, a1, a2, a3 = compute_errors(gt_depth, pred_depth, gt_disp, pred_disp)

    error_pack = collections.OrderedDict()
    error_pack['AbsRel'] = abs_rel
    error_pack['SqRel'] = sq_rel
    error_pack['RMS'] = rms
    error_pack['RMSlog'] = rms_log
    error_pack['log10'] = log_10
    error_pack['D1'] = d1_all
    error_pack['<1.25'] = a1
    error_pack['<1.25^2'] = a2
    error_pack['<1.25^3'] = a3

    return error_pack

def process_disp(disp, out_size):
    disp = disp[0].detach().cpu().numpy().transpose(1,2,0)
    H, W = out_size
    return cv2.resize(disp, (W, H))


@torch.no_grad()
def eval_kitti_depth(net, datasets, logger, cfgs, write=True):
    device = cfgs['device']
    net = net.eval()
    
    for dataset in datasets:
        if 'Depth' not in dataset._name:
            continue

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        print('START %s DEPTH EVALUATION ...' % (dataset._name))

        evals = []
        for i, data in enumerate(dataloader):
            image = data[0].to(device)
            gt_depth = data[1][0].numpy()
            H, W = gt_depth.shape

            disp = net.infer_depth(image)

            # process depth
            disp = process_disp(disp, (H, W))
            depth = 1.0 / (disp + 1e-4)

            # eval depth
            # mask = cv2.resize(image[0].detach().cpu().numpy().transpose(1,2,0), (W, H)).mean(2) > 0.5
            evals.append(eval_depth(gt_depth, depth, kitti=True))

            # extra_data visual
            if write:
                logger.write_disp(disp, folder='disp', prefix=str(i).zfill(6))

        logger.print_eval(evals, prefix='Eigen_Depth', write=True)
        print('END!')

if __name__ == "__main__":
    pass