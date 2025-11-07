# Copied from https://github.com/apple/ml-hugs/blob/main/hugs/utils/image.py

import os
import torch
import pathlib
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, List, BinaryIO


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


@torch.no_grad()
def normalize_depth(depth, min=None, max=None):
    if depth.shape[0] == 1:
        depth = depth[0]
    
    if min is None:
        min = depth.min()

    if max is None:
        max = depth.max()
        
    depth = (depth - min) / (max - min)
    depth = 1.0 - depth
    return depth


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    text_labels: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    
    grid = make_grid(tensor, **kwargs)
    txt_font = ImageFont.load_default()
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(im)
    draw.text((10, 10), text_labels, fill=(0, 0, 0), font=txt_font)
    im.save(fp, format=format)
    
    
def save_rgba_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = 'PNG',
    text_labels: Optional[List[str]] = None,
    **kwargs,
) -> None:
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    
    grid = make_grid(tensor, **kwargs)
    txt_font = ImageFont.load_default()
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if text_labels is not None:
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), text_labels, fill=(0, 0, 0), font=txt_font)
    im.save(fp, format=format)
