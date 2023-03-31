import torch
import torch.nn as nn
import numpy as np


def transformerFwd(U,
                   flo,
                   out_size,
                   name='SpatialTransformerFwd'):
    """Forward Warping Layer described in 
    'Occlusion Aware Unsupervised Learning of Optical Flow by Yang Wang et al'

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    flo: float
        The optical flow used for forward warping 
        having the shape of [num_batch, height, width, 2].
    backprop: boolean
        Indicates whether to back-propagate through forward warping layer
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        rep = torch.ones(size=[n_repeats], dtype=torch.long).unsqueeze(1).transpose(1,0)
        x = x.view([-1,1]).mm(rep)
        return x.view([-1]).int()

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, height, width, channels = im.shape[0], im.shape[1], im.shape[2], im.shape[3]
        out_height = out_size[0]
        out_width = out_size[1]
        max_y = int(height - 1)
        max_x = int(width - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width - 1.0) / 2.0
        y = (y + 1.0) * (height - 1.0) / 2.0

        # do sampling
        x0 = (torch.floor(x)).int()
        x1 = x0 + 1
        y0 = (torch.floor(y)).int()
        y1 = y0 + 1

        x0_c = torch.clamp(x0, 0, max_x)
        x1_c = torch.clamp(x1, 0, max_x)
        y0_c = torch.clamp(y0, 0, max_y)
        y1_c = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width).to(im.get_device())

        base_y0 = base + y0_c * dim2
        base_y1 = base + y1_c * dim2
        idx_a = base_y0 + x0_c
        idx_b = base_y1 + x0_c
        idx_c = base_y0 + x1_c
        idx_d = base_y1 + x1_c

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.view([-1, channels])
        im_flat = im_flat.float()

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

        zerof = torch.zeros_like(wa)
        wa = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y0_c, y0)).unsqueeze(1), wa, zerof)
        wb = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y1_c, y1)).unsqueeze(1), wb, zerof)
        wc = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y0_c, y0)).unsqueeze(1), wc, zerof)
        wd = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y1_c, y1)).unsqueeze(1), wd, zerof)

        zeros = torch.zeros(
            size=[
                int(num_batch) * int(height) *
                int(width), int(channels)
            ],
            dtype=torch.float)
        output = zeros.to(im.get_device())
        output = output.scatter_add(dim=0, index=idx_a.long().unsqueeze(1).repeat(1,channels), src=im_flat * wa)
        output = output.scatter_add(dim=0, index=idx_b.long().unsqueeze(1).repeat(1,channels), src=im_flat * wb)
        output = output.scatter_add(dim=0, index=idx_c.long().unsqueeze(1).repeat(1,channels), src=im_flat * wc)
        output = output.scatter_add(dim=0, index=idx_d.long().unsqueeze(1).repeat(1,channels), src=im_flat * wd)

        return output

    def _meshgrid(height, width):
        # This should be equivalent to:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                                 np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        return torch.from_numpy(x_t).float(), torch.from_numpy(y_t).float()

    def _transform(flo, input_dim, out_size):
        num_batch, height, width, num_channels = input_dim.shape[0:4]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        x_s, y_s = _meshgrid(out_height, out_width)
        x_s = x_s.to(flo.get_device()).unsqueeze(0)
        x_s = x_s.repeat([num_batch, 1, 1])

        y_s = y_s.to(flo.get_device()).unsqueeze(0)
        y_s =y_s.repeat([num_batch, 1, 1])

        x_t = x_s + flo[:, :, :, 0] / ((out_width - 1.0) / 2.0)
        y_t = y_s + flo[:, :, :, 1] / ((out_height - 1.0) / 2.0)

        x_t_flat = x_t.view([-1])
        y_t_flat = y_t.view([-1])

        input_transformed = _interpolate(input_dim, x_t_flat, y_t_flat,
                                            out_size)

        output = input_transformed.view([num_batch, out_height, out_width, num_channels])
        return output

    #out_size = int(out_size)
    output = _transform(flo, U, out_size)
    return output

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, padding=1)(x)
    mu_y = nn.AvgPool2d(3, 1, padding=1)(y)

    sigma_x = nn.AvgPool2d(3, 1, padding=1)(x**2) - mu_x**2
    sigma_y = nn.AvgPool2d(3, 1, padding=1)(y**2) - mu_y**2
    sigma_xy = nn.AvgPool2d(3, 1, padding=1)(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    return SSIM

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