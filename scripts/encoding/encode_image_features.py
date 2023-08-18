import os
import argparse
import warnings
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from encoders import ClipL14, TimmModel, DINOv2Encoder, VggGram

# import cProfile
# import pstats
# from pstats import SortKey

encoders = {
    'clip-L-14': (ClipL14, 'ViT-L-14'),
    'convnextv2-l': (TimmModel, 'convnextv2_large'),
    'convnextv2-h': (TimmModel, 'convnextv2_huge'),
    'dinov2-l-hf': (TimmModel, 'vit_large_patch14_dinov2'),
    'dinov2-l-fb': (DINOv2Encoder, 'dinov2_vitl14'),
    'vgg19-gram': (VggGram, None)
}


class ImageDataset(Dataset):

    def __init__(self, image_paths, transform, is_generated):
        self.image_paths = image_paths
        self.transform = transform
        self.is_generated = is_generated

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Custom warning handler
        def custom_showwarning(message,
                               category,
                               filename,
                               lineno,
                               file=None,
                               line=None):
            print(f"{image_path}: {message}")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            warnings.showwarning = custom_showwarning

            if self.is_generated:
                key = int(os.path.basename(image_path).split('-')[0])
            else:
                key = 0
            with Image.open(image_path) as image:
                image = self.transform(image)

        return key, image


def encode_image_features(image_paths,
                          encoder,
                          generated,
                          batch_size=16,
                          device='cuda'):

    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, encoder.transform, generated)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=8,
                            shuffle=False)

    all_keys = []
    all_features = []
    # all_images = []

    with torch.no_grad(), torch.autocast('cuda'):
        for keys, images in dataloader:
            images = images.to(device)
            features = encoder.encode(images)
            all_keys.extend(keys)
            all_features.append(features)
            # all_images.append(images)

    # Concatenate all the features
    all_features = torch.cat(all_features, dim=0)
    # all_images = torch.cat(all_images, dim=0)

    # Sort features based on keys
    sorted_indices = sorted(range(len(all_keys)), key=lambda k: all_keys[k])
    sorted_features = all_features[sorted_indices]
    # sorted_images = all_images[sorted_indices]

    return sorted_features.cpu().to(torch.float32)


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect all subdirectories that contain images
    subdirs_with_images = []
    for subdir, _, files in os.walk(args.src_dir):
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
               for file in files):
            subdirs_with_images.append(subdir)

    with tqdm(total=len(subdirs_with_images) * len(args.encoder_names),
              desc="Encoding") as pbar:
        for encoder_name in args.encoder_names:  # Loop over each encoder
            # Initialize encoder for the current encoder_name
            print(f"Set up encoder {encoder_name}...")
            encoder_class, model_name = encoders[encoder_name]
            encoder = encoder_class(model_name, args.resize_mode, device)
            feature_key = encoder_name + f"-{args.resize_mode}"

            for subdir in subdirs_with_images:

                if args.generated:
                    save_path = os.path.join(
                        os.path.dirname(subdir),
                        f"{os.path.basename(subdir)}-image-features.npz")
                else:
                    save_path = os.path.join(subdir, "image-features.npz")
                # This is to deal with a bug in felix_vallotton_stroke
                save_path = save_path.replace('captions', 'prompts')

                save_path = os.path.join(
                    args.dst_dir, os.path.relpath(save_path, args.src_dir))

                if os.path.exists(save_path):
                    image_features_all = dict(np.load(save_path).items())
                    if (feature_key in image_features_all
                            and not args.overwrite):
                        print(f"{feature_key} already in {save_path}, skip")
                        pbar.update(1)
                        continue
                else:
                    image_features_all = {}

                image_paths = [
                    os.path.join(subdir, file) for file in os.listdir(subdir)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg',
                                              '.webp'))
                ]
                image_features = encode_image_features(image_paths,
                                                       encoder,
                                                       args.generated,
                                                       args.batch_size,
                                                       device=device)
                image_features_all[feature_key] = image_features.numpy()

                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(save_path, **image_features_all)

                # if args.save_transformed_images:
                #     save_path = save_path.replace(
                #           'clip-image-features', 'images')
                #     torch.save(transformed_images, save_path)

                pbar.update(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Encode image features for subdirectories.")
    parser.add_argument("--src_dir",
                        type=str,
                        required=True,
                        help="Directory to process")
    parser.add_argument("--dst_dir",
                        type=str,
                        required=True,
                        help="Directory to save_features")
    parser.add_argument(
        "--encoder_name",
        type=str,
        default=None,
        help="Choice of encoder (overwritten by encoder_names)")
    parser.add_argument(
        "--encoder_names",
        type=str,
        default=None,
        help="Comma-separated list of encoders (e.g. encoder1, encoder2)")
    # parser.add_argument("--save_transformed_images",
    #                     action="store_true",
    #                     help="Whether to save transformed image or not")
    parser.add_argument("--resize_mode",
                        type=str,
                        help="Mode used to resize the images")
    parser.add_argument("--generated",
                        action="store_true",
                        help="Dealing with generated images")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Whether to overwrite or not")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size for feature encoding")
    args = parser.parse_args()

    if args.encoder_names is not None:
        args.encoder_names = args.encoder_names.split(',')
    elif args.encoder_name is not None:
        args.encoder_names = [args.encoder_name]
    else:
        raise ValueError(
            "Either --encoder_name or --encoder_names should be speciefied")

    main(args)

    # Profiler = cProfile.Profile()
    # Profiler.enable()
    # main(args)  # or whatever method you want to profile
    # Profiler.disable()
    # Sortby = SortKey.CUMULATIVE
    # Ps = pstats.Stats(profiler).sort_stats(sortby)
    # Ps.print_stats(20)
