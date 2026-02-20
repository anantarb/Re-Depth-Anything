import torch
import torch.nn as nn
import os
from PIL import Image
import cv2
import torch.nn.functional as F
import numpy as np


class DepthEnsemble(nn.Module):
    def __init__(self):
        super(DepthEnsemble, self).__init__()
        self.depth_maps = []

    def forward(self):

        depth_maps = torch.stack(self.depth_maps, dim=0)
        return depth_maps.mean(dim=0)

    def add_depth(self, depth_map):
        self.depth_maps.append(depth_map)

    def isEmpty(self):
        return len(self.depth_maps) == 0

    def len(self):
        return len(self.depth_maps)


def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        a1=a1,
        a2=a2,
        a3=a3,
        abs_rel=abs_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
        sq_rel=sq_rel,
    )


def srgb_to_linear(x):
    return x**2.2


def linear_to_srgb(x):
    return x ** (1 / 2.2)


def laplacian_smoothness(z):
    lap_k = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=z.dtype, device=z.device
    ).view(1, 1, 3, 3)
    lap_k = lap_k.repeat(z.shape[1], 1, 1, 1)
    lap = F.conv2d(z, lap_k, padding=1, groups=z.shape[1])

    loss = lap.abs()
    loss = loss.mean(dim=(1, 2, 3))

    return loss.mean()


def align_least_squares(gt_depth, model_output, mask=None):

    y = gt_depth.reshape(-1).to(dtype=torch.float32)
    x = model_output.reshape(-1).to(dtype=torch.float32)

    if mask is not None:
        m = mask.reshape(-1)
        if m.dtype != torch.bool:
            m = m != 0
        x = x[m]
        y = y[m]

    ones = torch.ones_like(x)
    A = torch.stack([x, ones], dim=1)

    sol = torch.linalg.lstsq(A, y).solution
    scale, shift = sol[0], sol[1]
    return scale, shift


def save_RGB(x, dir, filename):
    for i in range(x.shape[0]):
        Image.fromarray((x[i].permute(1, 2, 0).cpu() * 255).byte().numpy()).save(
            os.path.join(dir, f"{i}_{filename}.png")
        )


def save_mask(x, dir, filename):
    for i in range(x.shape[0]):
        Image.fromarray((x[i].cpu() * 255).byte().numpy()).save(
            os.path.join(dir, f"{i}_{filename}.png")
        )


def colorize_depth(x):
    color = torch.zeros(
        x.shape[0], 3, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device
    )
    for i in range(x.shape[0]):
        u8 = (x[i] * 255).byte().cpu().numpy()
        u8 = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        u8 = cv2.cvtColor(u8, cv2.COLOR_BGR2RGB) / 255.0
        color[i] = torch.from_numpy(u8).permute(2, 0, 1)
    return color
