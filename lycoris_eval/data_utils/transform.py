"""
Code adapted from
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py
"""


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

import torchvision.transforms as TF


class ResizeMaxSize(nn.Module):

    def __init__(self,
                 max_size,
                 interpolation=TF.InterpolationMode.BICUBIC,
                 fn='max',
                 fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1 or width != height:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img,
                        padding=[
                            pad_w // 2, pad_h // 2, pad_w - pad_w // 2,
                            pad_h - pad_h // 2
                        ],
                        fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
    image_size: int,
    mean: Optional[Tuple[float, ...]],
    std: Optional[Tuple[float, ...]],
    resize_mode: str = 'crop',
    fill_color: int = 0,
):

    accept_resize_modes = ['resize', 'crop', 'padding', 'resize_rectangle']
    assert resize_mode in accept_resize_modes

    if not isinstance(mean, (list, tuple)):
        mean = (mean, ) * 3

    if not isinstance(std, (list, tuple)):
        std = (std, ) * 3

    normalize = TF.Normalize(mean=mean, std=std)

    if resize_mode == 'crop':
        transforms = [
            TF.Resize(image_size, interpolation=TF.InterpolationMode.BICUBIC),
            TF.CenterCrop(image_size),
        ]
    elif resize_mode == 'padding':
        transforms = [ResizeMaxSize(image_size, fill=fill_color)]
    # Aspect ratio preserving resize, can give rectangle
    elif resize_mode == 'resize_rectangle':
        transforms = [
            TF.Resize(image_size, interpolation=TF.InterpolationMode.BICUBIC),
        ]
    else:
        transforms = [
            TF.Resize((image_size, image_size),
                      interpolation=TF.InterpolationMode.BICUBIC),
        ]
    transforms.extend([
        _convert_to_rgb,
        TF.ToTensor(),
        normalize,
    ])
    return TF.Compose(transforms)
