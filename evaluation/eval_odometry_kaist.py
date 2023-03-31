import os
import numpy as np
import torch
import glob
import cv2
import collections
import copy
from torchvision.transforms import transforms
from tqdm import tqdm

''' ################################################################################################################
eval from https://github.com/B1ueber2y/TrianFlow
################################################################################################################ ''' 
def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y)/np.sum(X ** 2)
    return scale

def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

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

class EvalOdom():
    # ----------------------------------------------------------------------
	# poses: N,4,4
	# pose: 4,4
	# ----------------------------------------------------------------------
    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)
        # self.step_size = step_size

    def loadPoses(self, file_name):
        # ----------------------------------------------------------------------
		# Each line in the file should follow one of the following structures
		# (1) idx pose(3x4 matrix in terms of 12 numbers)
		# (2) pose(3x4 matrix in terms of 12 numbers)
		# ----------------------------------------------------------------------
        with open(file_name, 'r') as f:
            s = f.readlines()

        poses = []
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ")]
            withIdx = int(len(line_split) == 13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            # poses[frame_idx] = P
            poses.append(P.reshape([16]))
        return np.array(poses)
    
    def align_time_stamp(self, gt_csv, time_stamp, pose_pred):
        data_gt = np.genfromtxt(gt_csv, delimiter=",", skip_header=0)
        time_stamps_gt = data_gt[:, 0] / 1e+9

        time_stamp = np.array(time_stamp).astype('float64') / 1e+9
        index, index_gt = matching_time_indices(time_stamp, time_stamps_gt)

        num = len(index_gt)
        gt_rot3x4 = data_gt[index_gt, 1:].reshape([-1, 3, 4])
        expand_corrd = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 1, 4]).repeat(num, axis=0)
        gt_rot4x4 = np.concatenate([ gt_rot3x4, expand_corrd ], axis=1)

        pose_pred = pose_pred[index]

        gt = {}
        pose = {}
        for i in range(num):
            pose[i] = pose_pred[i].reshape([4, 4])
            gt[i] = gt_rot4x4[i]
        return gt, pose

    def trajectory_distances(self, poses):
        # ----------------------------------------------------------------------
        # poses: dictionary: [frame_idx: pose]
        # ----------------------------------------------------------------------
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]   
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))
        return dist

    def rotation_error(self, pose_error):
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5*(a+b+c-1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    def translation_error(self, pose_error):
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx**2+dy**2+dz**2)

    def last_frame_from_segment_length(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        err = []
        dist = self.trajectory_distances(poses_gt)
        self.step_size = 10

        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(dist, first_frame, len_)

                # ----------------------------------------------------------------------
				# Continue if sequence not long enough
				# ----------------------------------------------------------------------
                if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
                    continue

                # ----------------------------------------------------------------------
				# compute rotational and translational errors
				# ----------------------------------------------------------------------
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)

                # ----------------------------------------------------------------------
				# compute speed 
				# ----------------------------------------------------------------------
                num_frames = last_frame - first_frame + 1.0
                speed = len_/(0.1*num_frames)

                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err
        
    def save_sequence_errors(self, err, file_name):
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def compute_overall_err(self, seq_err):
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err

    def compute_segment_error(self, seq_errs):
        # ----------------------------------------------------------------------
		# This function calculates average errors for different segment.
		# ----------------------------------------------------------------------

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []
        # ----------------------------------------------------------------------
		# Get errors
		# ----------------------------------------------------------------------
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])
        # ----------------------------------------------------------------------
		# Compute average
		# ----------------------------------------------------------------------
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def compute_ATE(self, gt, pred):
        """Compute RMSE of ATE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        """
        errors = []
        idx_0 = list(pred.keys())[0]
        gt_0 = gt[idx_0]
        pred_0 = pred[idx_0]

        for i in pred:
            cur_gt = gt[i]
            gt_xyz = cur_gt[:3, 3] 

            cur_pred = pred[i]
            pred_xyz = cur_pred[:3, 3]

            align_err = gt_xyz - pred_xyz

            errors.append(np.sqrt(np.sum(align_err ** 2)))
        ate = np.sqrt(np.mean(np.asarray(errors) ** 2)) 
        return ate
    
    def compute_RPE(self, gt, pred):
        """Compute RPE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            rpe_trans
            rpe_rot
        """
        trans_errors = []
        rot_errors = []
        for i in list(pred.keys())[:-1]:
            gt1 = gt[i]
            gt2 = gt[i+1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred[i]
            pred2 = pred[i+1]
            pred_rel = np.linalg.inv(pred1) @ pred2
            rel_err = np.linalg.inv(gt_rel) @ pred_rel
            
            trans_errors.append(self.translation_error(rel_err))
            rot_errors.append(self.rotation_error(rel_err))
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        return rpe_trans, rpe_rot

    def scale_optimization(self, gt, pred):
        """ Optimize scaling factor
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            new_pred (4x4 array dict): predicted poses after optimization
        """
        pred_updated = copy.deepcopy(pred)
        xyz_pred = []
        xyz_ref = []
        for i in pred:
            pose_pred = pred[i]
            pose_ref = gt[i]
            xyz_pred.append(pose_pred[:3, 3])
            xyz_ref.append(pose_ref[:3, 3])
        xyz_pred = np.asarray(xyz_pred)
        xyz_ref = np.asarray(xyz_ref)
        scale = scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated

    def aligment_rescale(self, pred, gt, alignment=False):
        # assert(alignment == '6DoF' or alignment == '7DoF')
         # Pose alignment to first frame
        idx_0 = sorted(list(pred.keys()))[0]
        pred_0 = pred[idx_0]
        gt_0 = gt[idx_0]

        poses_result = {}
        poses_gt = {}
        for cnt in pred:
            poses_result[cnt] = np.linalg.inv(pred_0) @ pred[cnt]
            poses_gt[cnt] = np.linalg.inv(gt_0) @ gt[cnt]

        # get XYZ
        xyz_gt = []
        xyz_result = []
        for cnt in poses_result:
            xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
            xyz_result.append([poses_result[cnt][0, 3], poses_result[cnt][1, 3], poses_result[cnt][2, 3]])
        xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
        xyz_result = np.asarray(xyz_result).transpose(1, 0)

        # 7Dof optimization
        r, t, scale = umeyama_alignment(xyz_result, xyz_gt, alignment)
        align_transformation = np.eye(4)
        align_transformation[:3:, :3] = r
        align_transformation[:3, 3] = t
        
        for cnt in poses_result:
            poses_result[cnt][:3, 3] *= scale
            poses_result[cnt] = align_transformation @ poses_result[cnt]

        return poses_result, poses_gt

    def eval(self, gt_txt, result_txt, time_stamp):
        # gt_dir: the directory of groundtruth poses txt
        # results_dir: the directory of predicted poses txt       
        self.gt_txt = gt_txt

        poses_result = self.loadPoses(result_txt)
        poses_gt, poses_result = self.align_time_stamp(self.gt_txt, time_stamp, poses_result)

        # aligment
        # poses_result_6dof, poses_gt_6dof = self.aligment_rescale(poses_result, poses_gt, alignment=False)
        poses_result_7dof, poses_gt_7dof = self.aligment_rescale(poses_result, poses_gt, alignment=True)

        # ----------------------------------------------------------------------
        # compute sequence errors
        # ----------------------------------------------------------------------
        seq_err = self.calc_sequence_errors(poses_gt_7dof, poses_result_7dof)

        # ----------------------------------------------------------------------
        # Compute segment errors
        # ----------------------------------------------------------------------
        avg_segment_errs = self.compute_segment_error(seq_err)

        # ----------------------------------------------------------------------
        # compute overall error
        # ----------------------------------------------------------------------
        ave_t_err, ave_r_err = self.compute_overall_err(seq_err)

        # ----------------------------------------------------------------------
        # Compute RPE
        # ----------------------------------------------------------------------
        rpe_trans, rpe_rot = self.compute_RPE(poses_gt_7dof, poses_result_7dof)

        # ----------------------------------------------------------------------
        # Compute ATE
        # ----------------------------------------------------------------------
        ate = self.compute_ATE(poses_gt_7dof, poses_result_7dof)

        # collection
        error_pack = collections.OrderedDict()
        error_pack['Translational error (%)'] = ave_t_err*100
        error_pack['Rotational error (deg/100m)'] = ave_r_err/np.pi*180*100
        error_pack['ATE'] = ate
        error_pack['RPE (m)'] = rpe_trans
        error_pack['RPE (deg)'] = rpe_rot * 180 / np.pi
        return error_pack, poses_gt_7dof, poses_result_7dof


''' ################################################################################################################
pre-process
################################################################################################################ ''' 
def process(x):
    if len(x.shape) == 4:
        x = x[0].detach().cpu().numpy().transpose(1, 2, 0)
        in_H, in_W, C = x.shape
        if C == 2:
            x[:, :, 0] = x[:, :, 0] / in_W
            x[:, :, 1] = x[:, :, 1] / in_H
    else:
        x = x[0].detach().cpu().numpy()
    return x

def im_norm(M):
    return (M - np.min(M)) / (np.max(M) - np.min(M)) * 255

def write_match(data, matchs, logger, folder, name):
    im1 = (data['color', 0].permute(0,2,3,1).cpu().numpy()[0] * 255).astype('uint8')
    im2 = (data['color', 1].permute(0,2,3,1).cpu().numpy()[0] * 255).astype('uint8')
    kp1 = [cv2.KeyPoint(match[0], match[1], 1) for match in matchs.transpose()]
    kp2 = [cv2.KeyPoint(match[2], match[3], 1) for match in matchs.transpose()]
    good = [cv2.DMatch(i, i, 1, 0) for i in range(len(kp1))]
    img3 = cv2.drawMatches(im1, kp1, im2, kp2, good, None)
    logger.write_image(img3, folder=folder, prefix=name)

def write_data(data, logger, folder, name):
    C = data.shape[2]
    data = np.squeeze(data)

    if C == 1 and 'disp' in folder:
        logger.write_disp(data, folder=folder, prefix=name)
    elif C == 2:
        logger.write_flow(data, folder=folder, prefix=name)
    else:
        logger.write_image(im_norm(data), folder=folder, prefix=name)

@torch.no_grad()
def eval_kaist_odometry(net, datasets, logger, cfgs, kpts=None, write=True):

    device = cfgs['device']
    net = net.eval()
    for dataset in datasets:
        if 'KAIST_ODOMETRY' not in dataset._name:
            continue

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        print('START %s ODOMETRY EVALUATION ...' % (dataset._name))

        poses = []
        global_pose = np.eye(4)
        poses.append(copy.deepcopy(global_pose))

        for i, data in tqdm(enumerate(dataloader)):
            for key, itms in data.items():
                data[key] = itms.to(device)

            pose, eval_data = net.infer_vo(data, kpts)

            # process
            for key, value in eval_data.items():
                eval_data[key] = process(value)

            rel_pose = pose[0].detach().cpu().numpy()
            global_pose = np.matmul(global_pose, np.linalg.inv(rel_pose))
            poses.append(copy.deepcopy(global_pose))

            # visual
            if write:
                for key, value in eval_data.items():
                    if key == 'sample_points':
                        write_match(data, value, logger, folder='vo_' + key, name=dataset.time_stamp[i])
                    else:
                        write_data(value, logger, folder='vo_' + key, name=dataset.time_stamp[i])

        if dataset.eval_nums is None:
            eval_seq = dataset.scene
            save_name = 'kaist_' + eval_seq + dataset.save_prefix
            result_txt = logger.save_traj(poses, prefix=save_name)
            eval_tool = EvalOdom()

            loss_pack, poses_gt, poses_result = eval_tool.eval(dataset.gt_pose_file, result_txt, dataset.time_stamp)
            logger.print_eval_pose(loss_pack, prefix='odometry_' + save_name, write=True)
            logger.plotPath(save_name, poses_gt, poses_result, axis_y='y')
        print('END!')

if __name__ == "__main__":
    pass