"""
Adapted from
https://github.com/zwx8981/LIQE/blob/main/LIQE.py
"""

import torch
import numpy as np
import clip
from itertools import product
import torch.nn.functional as F
import torch.nn as nn

dists = [
    'jpeg2000 compression', 'jpeg compression', 'white noise', 'gaussian blur',
    'fastfading', 'fnoise', 'contrast', 'lens', 'motion', 'diffusion',
    'shifting', 'color quantization', 'oversaturation', 'desaturation',
    'white with color', 'impulse', 'multiplicative',
    'white noise with denoise', 'brighten', 'darken', 'shifting the mean',
    'jitter', 'noneccentricity patch', 'pixelate', 'quantization',
    'color blocking', 'sharpness', 'realistic blur', 'realistic noise',
    'underexposure', 'overexposure', 'realistic contrast change',
    'other realistic'
]

scenes = [
    'animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant',
    'still_life', 'others'
]
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

type2label = {
    'jpeg2000 compression': 0,
    'jpeg compression': 1,
    'white noise': 2,
    'gaussian blur': 3,
    'fastfading': 4,
    'fnoise': 5,
    'contrast': 6,
    'lens': 7,
    'motion': 8,
    'diffusion': 9,
    'shifting': 10,
    'color quantization': 11,
    'oversaturation': 12,
    'desaturation': 13,
    'white with color': 14,
    'impulse': 15,
    'multiplicative': 16,
    'white noise with denoise': 17,
    'brighten': 18,
    'darken': 19,
    'shifting the mean': 20,
    'jitter': 21,
    'noneccentricity patch': 22,
    'pixelate': 23,
    'quantization': 24,
    'color blocking': 25,
    'sharpness': 26,
    'realistic blur': 27,
    'realistic noise': 28,
    'underexposure': 29,
    'overexposure': 30,
    'realistic contrast change': 31,
    'other realistic': 32
}

dist_map = {
    'jpeg2000 compression': 'jpeg2000 compression',
    'jpeg compression': 'jpeg compression',
    'white noise': 'noise',
    'gaussian blur': 'blur',
    'fastfading': 'jpeg2000 compression',
    'fnoise': 'noise',
    'contrast': 'contrast',
    'lens': 'blur',
    'motion': 'blur',
    'diffusion': 'color',
    'shifting': 'blur',
    'color quantization': 'quantization',
    'oversaturation': 'color',
    'desaturation': 'color',
    'white with color': 'noise',
    'impulse': 'noise',
    'multiplicative': 'noise',
    'white noise with denoise': 'noise',
    'brighten': 'overexposure',
    'darken': 'underexposure',
    'shifting the mean': 'other',
    'jitter': 'spatial',
    'noneccentricity patch': 'spatial',
    'pixelate': 'spatial',
    'quantization': 'quantization',
    'color blocking': 'spatial',
    'sharpness': 'contrast',
    'realistic blur': 'blur',
    'realistic noise': 'noise',
    'underexposure': 'underexposure',
    'overexposure': 'overexposure',
    'realistic contrast change': 'contrast',
    'other realistic': 'other'
}

map2label = {
    'jpeg2000 compression': 0,
    'jpeg compression': 1,
    'noise': 2,
    'blur': 3,
    'color': 4,
    'contrast': 5,
    'overexposure': 6,
    'underexposure': 7,
    'spatial': 8,
    'quantization': 9,
    'other': 10
}

dists_map = [
    'jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color',
    'contrast', 'overexposure', 'underexposure', 'spatial', 'quantization',
    'other'
]

scene2label = {
    'animal': 0,
    'cityscape': 1,
    'human': 2,
    'indoor': 3,
    'landscape': 4,
    'night': 5,
    'plant': 6,
    'still_life': 7,
    'others': 8
}


class LIQE(nn.Module):

    def __init__(self, device):

        super(LIQE, self).__init__()
        self.model, preprocess = clip.load("ViT-B/32",
                                           device=device,
                                           jit=False)
        joint_texts = torch.cat([
            clip.tokenize(
                f"a photo of a {c} with {d} artifacts, which is of {q} quality"
            ) for q, c, d in product(qualitys, scenes, dists_map)
        ]).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(joint_texts)
            self.text_features = self.text_features / self.text_features.norm(
                dim=1, keepdim=True)
        self.step = 32
        self.patch_size = 224

    def forward(self, x, n_patches):

        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size, self.step)
        x = x.unfold(3, self.patch_size, self.step)
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch_size, -1, 3, self.patch_size, self.patch_size)

        num_slices = x.size(1)
        selected = np.random.choice(num_slices,
                                    size=n_patches,
                                    replace=False)
        x = x[:, selected, ...]
        x = x.reshape(-1, 3, self.patch_size, self.patch_size)

        image_features = self.model.encode_image(x)

        # normalized features
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ self.text_features.t(
        )

        logits_per_image = logits_per_image.view(batch_size, n_patches, -1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = F.softmax(logits_per_image, dim=1)

        logits_per_image = logits_per_image.view(-1, len(qualitys),
                                                 len(scenes), len(dists_map))
        logits_quality = logits_per_image.sum(3).sum(2)

        quality = (1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] +
                   3 * logits_quality[:, 2] + 4 * logits_quality[:, 3] +
                   5 * logits_quality[:, 4])

        return quality
