import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import timm
import open_clip

from transform import image_transform


class Encoder(object):

    def __init__(self, model_name, resize_mode, device):

        accept_resize_modes = ['resize', 'crop', 'padding']
        assert resize_mode in accept_resize_modes, \
            "invalid resize_mode, must be 'resize', 'crop', or 'padding'"

        model, image_size, image_mean, image_std = self.setup_model(model_name)
        self.model = model.eval().to(device)

        accept_rectangle = self.get_accept_rectangle()

        if accept_rectangle:
            if resize_mode != 'resize':
                print(f"Warning, {resize_mode} is used while" +
                      " the model can accept rectangle input")
            else:
                resize_mode = 'resize_rectangle'

        self.transform = image_transform(image_size=image_size,
                                         mean=image_mean,
                                         std=image_std,
                                         resize_mode=resize_mode)

    def setup_model(self):
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError

    def get_accept_rectangle(self):
        return False


class ClipL14(Encoder):

    def setup_model(self, model_name):
        model = open_clip.create_model(model_name,
                                       pretrained='datacomp_xl_s13b_b90k')
        image_size = model.visual.image_size[0]
        image_mean = model.visual.image_mean
        image_std = model.visual.image_std
        return model, image_size, image_mean, image_std

    def encode(self, images):
        return self.model.encode_image(images)


class TimmModel(Encoder):

    def setup_model(self, model_name):
        self.model_name = model_name
        model = timm.create_model(model_name, pretrained=True)
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        image_size = data_config['input_size'][1]
        image_mean = data_config['mean']
        image_std = data_config['std']
        return model, image_size, image_mean, image_std

    def encode(self, images):
        return self.model(images)

    def get_accept_rectangle(self):
        if 'convnext' in self.model_name:
            return True
        return False


class DINOv2Encoder(Encoder):

    def setup_model(self, model_name):
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        image_size = 224
        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)
        return model, image_size, image_mean, image_std

    def encode(self, images):
        return self.model(images)


class VGG(nn.Module):

    def __init__(self):
        # Use layers conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)
        return features


class VggGram(Encoder):

    def setup_model(self, model_name):
        model = VGG()
        image_size = 224
        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)
        return model, image_size, image_mean, image_std

    def encode(self, images):
        grams = []
        features = self.model(images)

        for feature in features:
            batch_size, channel, height, width = feature.shape
            feature_reshaped = feature.view(batch_size, channel,
                                            height * width)
            # Normalize using the number of element in each feature map
            feature_reshaped = feature_reshaped / np.sqrt(
                channel * height * width)
            gram_matrix = torch.bmm(feature_reshaped,
                                    feature_reshaped.transpose(1, 2))
            grams.append(gram_matrix.view(batch_size, -1))
        grams = torch.hstack(grams)
        return grams

    def get_accept_rectangle(self):
        return True
