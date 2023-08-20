import numpy as np

import torch
import torch.nn as nn

from lycoris_eval.data_utils.transform import image_transform
from lycoris_eval.evaluation.models.liqe import LIQE
from lycoris_eval.evaluation.models.maniqa import MANIQA
from lycoris_eval.evaluation.models.aesthetics_scorer import AestheticScorer


class Scorer(nn.Module):

    def __init__(self,
                 model_path,
                 image_size=512,
                 resize_mode='resize',
                 device='cuda'):
        """
        Note that patch_size depends on the model while image size
        can be chosen arbitrarily as long as it is larger than patch size
        (it needs to be large enough if we use multiple patches to
        compute the score)
        """

        super(Scorer, self).__init__()
        accept_resize_modes = ['resize', 'crop', 'padding']
        assert resize_mode in accept_resize_modes, \
            "invalid resize_mode, must be 'resize', 'crop', or 'padding'"

        self.device = device
        self.image_size = image_size
        self.setup_model()
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        accept_rectangle = self.get_accept_rectangle()

        if accept_rectangle:
            if resize_mode != 'resize':
                print(f"Warning, {resize_mode} is used while" +
                      " the model can accept rectangle input")
            else:
                resize_mode = 'resize_rectangle'

        self.transform = image_transform(image_size=image_size,
                                         mean=self.image_mean,
                                         std=self.image_std,
                                         resize_mode=resize_mode)
        self.step = 32

    def setup_model(self):
        raise NotImplementedError

    def get_accept_rectangle(self):
        return False

    def forward(self, x, n_patches=1):
        if self.image_size == self.patch_size:
            return self.model(x).reshape(-1)

        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size, self.step)
        x = x.unfold(3, self.patch_size, self.step)
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch_size, -1, 3, self.patch_size, self.patch_size)

        num_slices = x.size(1)
        selected = np.random.choice(num_slices, size=n_patches, replace=False)
        x = x[:, selected, ...]
        x = x.reshape(-1, 3, self.patch_size, self.patch_size)
        scores = self.model(x).reshape(batch_size, n_patches)
        scores = torch.mean(scores, axis=1)
        return scores


class LiqeScorer(Scorer):

    def setup_model(self):
        self.model = LIQE(self.device)
        self.patch_size = 224
        self.image_mean = (0.48145466, 0.4578275, 0.40821073)
        self.image_std = (0.26862954, 0.26130258, 0.27577711)

    def get_accept_rectangle(self):
        return True

    def forward(self, x, n_patches=15):
        return self.model(x, n_patches=n_patches)


class ManiqaScorer(Scorer):

    def setup_model(self):

        self.model = MANIQA(embed_dim=768,
                            num_outputs=1,
                            dim_mlp=768,
                            patch_size=8,
                            img_size=224,
                            window_size=4,
                            depths=[2, 2],
                            num_heads=[4, 4],
                            num_tab=2,
                            scale=0.8).to(self.device)

        self.patch_size = 224
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)

    def get_accept_rectangle(self):
        return True


class ArtifactScorer(Scorer):

    def setup_model(self):
        self.model = AestheticScorer(
            'laion/CLIP-ViT-L-14-laion2B-s32B-b82K').to(self.device)
        self.patch_size = 224
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)
