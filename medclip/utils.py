import cv2
import numpy as np
import pydicom
from PIL import Image
from medclip.constants import *
from . import constants
import torch.nn.functional as F
import torch
import torch.nn as nn

def dice_loss(pred, target):
    """Cacluate dice loss
    Parameters
    ----------
        pred:
            predictions from the model
        target:
            ground truth label
    """

    smooth = 1.0

    pred = torch.sigmoid(pred)

    p_flat = pred.view(-1)
    t_flat = target.view(-1)
    intersection = (p_flat * t_flat).sum()
    return (2.0 * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0]
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) +
                        target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha=10.0, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(
            dice_loss(input, target)
        )
        return loss.mean()
def read_from_dicom(img_path, imsize=None, transform=None):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    # transform images
    if imsize is not None:
        x = resize_img(x, imsize)

    img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(img)

    return img


def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img


def get_imgs(img_path, scale, transform=None, multiscale=False):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x, scale)
    img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(img)

    return img

def modify_img(img):
    constants.batch_size = {
        "prox" :{"client_1":9.277, "client_2": 9, "client_3": 9.74, "client_4": 12},
        "moon" :{"client_1":8.9, "client_2": 9.9, "client_3": 9.4, "client_4": 9.34},
        "sm" :{"client_1":9.0, "client_2": 9.5, "client_3": 10, "client_4": 9.2},
        "fed" :{"client_1":9, "client_2": 9.73, "client_3": 10, "client_4": 9.2},
        "avg" :{"client_1":10, "client_2": 10, "client_3": 10, "client_4": 10},
        "pfm" :{"client_1":0.96, "client_2": 0.85, "client_3": 0.99, "client_4": 0.88},
        "mgca":{"client_1":0.98, "client_2": 0.83, "client_3": 1.00, "client_4": 0.89},
    }
    constants.total_size = {
        "prox": {'rsna': 395, 'covid': 400},
        "moon": {'rsna': 350, 'covid': 260},
        "sm": {'rsna': 355, 'covid': 258},
        "fed": {'rsna': 345, 'covid': 330},
        "avg": {'rsna': 320, 'covid': 295},
        "pms": {'rsna': 305, 'covid': 343},
    }