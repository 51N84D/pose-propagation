import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import scipy.misc
from PIL import Image
import numpy as np
import re


def transform_topk(aff, frame1, k, h2=None, w2=None):
    """
    INPUTS:
        - aff: affinity matrix, b * N * N
        - frame1: reference frame
        - k: only aggregate top-k pixels with highest aff(j,i)
        - h2, w2, frame2's height & width
    OUTPUT:
        - frame2: propagated mask from frame1 to the next frame
    """
    b, c, h, w = frame1.shape
    # print("frame1: ", frame1.shape)
    b, N1, N2 = aff.shape
    # print("aff: ", aff.shape)
    # b * 20 * N
    tk_val, tk_idx = torch.topk(aff, dim=1, k=k)
    # b * N
    tk_val_min, _ = torch.min(tk_val, dim=1)
    tk_val_min = tk_val_min.view(b, 1, N2)
    aff[tk_val_min > aff] = 0
    frame1 = frame1.contiguous().view(b, c, -1)
    frame2 = torch.bmm(frame1, aff)
    if h2 is None:
        return frame2.view(b, c, h, w)
    else:
        return frame2.view(b, c, h2, w2)


class NLM_woSoft(nn.Module):
    """
    Non-local mean layer w/o softmax on affinity
    """

    def __init__(self):
        super(NLM_woSoft, self).__init__()

    def forward(self, in1, in2):
        n, c, h, w = in1.size()
        in1 = in1.view(n, c, -1)
        in2 = in2.view(n, c, -1)

        affinity = torch.bmm(in1.permute(0, 2, 1), in2)
        return affinity


class normalize(nn.Module):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std=(1.0, 1.0, 1.0)):
        super(normalize, self).__init__()
        self.mean = nn.Parameter(torch.FloatTensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.FloatTensor(std), requires_grad=False)

    def forward(self, frames):
        if len(frames.shape) < 4:
            frames = frames.unsqueeze(0)

        frames = frames.permute(0, 1, 3, 2)

        b, c, h, w = frames.shape

        frames = (
            frames - self.mean.view(1, 3, 1, 1).repeat(b, 1, h, w)
        ) / self.std.view(1, 3, 1, 1).repeat(b, 1, h, w)
        return frames


def to_tensor():

    resnet_transforms = transforms.Compose([transforms.ToTensor(),])
    return resnet_transforms


def norm_mask(mask):
    """
    INPUTS:
     - mask: segmentation mask
    """
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if mask_cnt.max() > 0:
            mask_cnt = mask_cnt - mask_cnt.min()
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt

    return mask


def color_normalize(x, mean, std):
    b, c, h, w = x.shape
    if c == 1:
        x = x.repeat(1, 3, 1, 1)
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)

    return x


def draw_labelmap_np(img, pt, sigma, type="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    if type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[g_y[0] : g_y[1], g_x[0] : g_x[1]]

    return img


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)
