import cv2
import torch
import numpy as np
import png
from PIL import Image
import math

def read_flow_png(flow_file):
    """
    Read optical flow from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2**15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow

def pil_loader(path):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

def load_rgb_from_file(path, do_flip):
    color = pil_loader(path)
    # to BGR
    r, g, b = color.split()
    color = Image.merge("RGB", (b, g, r))

    if do_flip:
        color = color.transpose(Image.FLIP_LEFT_RIGHT)

    return color

def load_bayer_from_file(path, do_flip):
    image = cv2.imread(path, -1)
    color = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)
    color = Image.fromarray(color)
    # color = pil_loader(path)
    # to BGR
    # r, g, b = color.split()
    # color = Image.merge("RGB", (b, g, r))

    if do_flip:
        color = color.transpose(Image.FLIP_LEFT_RIGHT)

    return color

def load_poses(file_name):
    '''
    the pose file: Each line == 12 number of 3x4 matrix
    '''
    def load_from_vector(line):
        pose = np.eye(4)
        line_split = [float(i) for i in line.split(" ")]
        for row in range(3):
            for col in range(4):
                pose[row, col] = line_split[row*4 + col]
        return pose

    with open(file_name, 'r') as f:
        src = f.readlines()

    poses = [load_from_vector(line) for line in src]
    return poses

def load_cam_intrinsic(cam_file, in_size, out_size, position):
    with open(cam_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if position in line:
            K = line.strip('\n').split(' ')[1:]
    K = np.array([float(v) for v in K]).reshape(3,4)[:3,:3]

    # rescale
    H, W = in_size
    zoom_x, zoom_y = out_size[1]/W, out_size[0]/H
    K[0,:], K[1,:] = K[0,:] * zoom_x, K[1,:] * zoom_y
    return torch.from_numpy(K).float()

def q2r(mat):
    r = []
    for m in mat:
        r.append(quaternion_matrix(m))
    return r

def quaternion_matrix(quaternion_xyz):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    _EPS = np.finfo(float).eps * 4.0
    quaternion = np.roll(quaternion_xyz[3:], 1)
    # quaternion = quaternion_xyz[3:]
    x = quaternion_xyz[0]
    y = quaternion_xyz[1]
    z = quaternion_xyz[2]
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], x],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], y],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], z],
        [                0.0,                 0.0,                 0.0, 1.0]])

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q