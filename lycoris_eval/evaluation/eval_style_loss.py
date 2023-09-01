import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from utils import get_all_subdirectories, get_relevant_directories
from utils import load_image_features, load_image_features_in_out
from utils import update_results

from lycoris_eval.encoding.encode_image_features import encode_image_features
from lycoris_eval.encoding.encoders import VggGram


def get_images(directory):
    file_list = os.listdir(directory)
    file_list = list(map(lambda f: os.path.join(directory, f), file_list))
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    imgs = list(
        filter(lambda f: f.lower().endswith(image_extensions), file_list))
    return imgs


def get_vgg_features(directory, batch_size, mode):

    if not os.path.exists(directory):
        return None

    assert mode in ['dataset', 'style_prompts', 'all']
    files = os.listdir(directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == 'dataset' or mode == 'style_prompts':
        vgg_features = None
        if mode == 'dataset':
            feature_name = 'image-features.npz'
            style_dir = directory
            generated = False
        else:
            feature_name = 'style_prompts-image-features.npz'
            style_dir = os.path.join(directory, 'style_prompts')
            # style_dir = os.path.join(directory, 'in_dist_prompts')
            generated = True
        if feature_name in files:
            feature_path = os.path.join(directory, feature_name)
            vgg_features = load_image_features(feature_path,
                                               'vgg19-gram')['resize']
            vgg_features = torch.tensor(vgg_features).to(device)
        elif os.path.isdir(style_dir):
            image_paths = get_images(style_dir)
            if image_paths:
                encoder = VggGram(None, 'resize', device)
                vgg_features = encode_image_features(image_paths,
                                                     encoder,
                                                     generated,
                                                     batch_size,
                                                     device=device,
                                                     return_cpu=False)
        return vgg_features

    if mode == 'all':
        vgg_features_dict = load_image_features_in_out(directory, 'vgg19-gram')
        for key in vgg_features_dict:
            vgg_features_dict[key] = torch.tensor(
                vgg_features_dict[key]).to(device)
        if not vgg_features_dict:
            folder_names = {
                'in': 'in_dist_prompts',
                'out': 'out_dist_prompts',
                'trigger': 'triggeronly'
            }
            for prompt_type in folder_names:
                image_dir = os.path.join(directory, folder_names[prompt_type])
                if os.path.exists(image_dir):
                    image_paths = get_images(image_dir)
                    encoder = VggGram(None, 'resize', device)
                    vgg_features = encode_image_features(image_paths,
                                                         encoder,
                                                         True,
                                                         batch_size,
                                                         device=device,
                                                         return_cpu=False)
                    vgg_features_dict[prompt_type] = vgg_features

        return vgg_features_dict


def get_style_loss(eval_dir, ref_dir, batch_size=8, save_dir=None):
    """
    Comparing with generated style images with same seed
    """

    ref_features = get_vgg_features(ref_dir,
                                    batch_size=batch_size,
                                    mode='style_prompts')
    if ref_features is None:
        return {}

    eval_features = get_vgg_features(eval_dir,
                                     batch_size=batch_size,
                                     mode='style_prompts')
    if eval_features is None:
        return {}

    metrics = {}
    ref_features = ref_features.to(torch.float32)
    eval_features = eval_features.to(torch.float32)
    losses = torch.sum((ref_features - eval_features)**2, axis=1)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'style-losses.npy'), losses.cpu())
    loss = torch.mean(losses)
    loss = loss.cpu().item()
    metrics['Style Loss (base model)'] = loss

    return metrics


def allpair_msd(X1, X2):
    X1 = X1.to(torch.float32)
    X2 = X2.to(torch.float32)
    avg1 = torch.mean(X1, axis=0, keepdims=True)
    avg2 = torch.mean(X2, axis=0, keepdims=True)
    var1 = torch.sum((X1 - avg1)**2) / X1.shape[0]
    var2 = torch.sum((X2 - avg2)**2) / X2.shape[0]
    cent_dis = torch.sum((avg1 - avg2)**2)
    return (cent_dis + var1 + var2).cpu().item()


def get_style_loss_allpairs(eval_dir, ref_dir, batch_size=8):
    """
    Comparaision with real style dataset
    """
    ref_features = get_vgg_features(ref_dir,
                                    batch_size=batch_size,
                                    mode='dataset')
    if ref_features is None:
        return {}

    eval_features_dict = get_vgg_features(eval_dir,
                                          batch_size=batch_size,
                                          mode='all')
    if len(eval_features_dict) == 0:
        return {}

    metrics = {}
    for prompt_type in eval_features_dict:
        loss = allpair_msd(ref_features, eval_features_dict[prompt_type])
        if np.isinf(loss):
            raise ValueError(f'{ref_dir}: inf loss detectetd')
        metrics[f'Style Loss ({prompt_type})'] = loss
    return metrics


def main(args):
    """Evaluate cosine similarity for tensor files in given directories."""

    # Check if CSV exists and if so, load its data
    if os.path.exists(args.metric_csv):
        try:
            existing_df = pd.read_csv(args.metric_csv)
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()
            # Or handle it in some other way, like logging a message, etc.
            print(f"Warning: The file {args.metric_csv} is empty!")
    else:
        existing_df = pd.DataFrame()

    results = []
    n_updated = 0
    eval_subdirs = get_all_subdirectories(args.eval_dir)

    for subdir in tqdm(eval_subdirs, desc="Evaluating"):
        eval_subdir, ref_subdir, key_path = get_relevant_directories(
            subdir, args.eval_dir, args.ref_dir, args.extra_level,
            args.extra_descr)

        if args.compare_with_dataset:
            metrics = get_style_loss_allpairs(eval_subdir, ref_subdir,
                                              args.batch_size)
        else:
            if args.save_dir is not None:
                save_dir = os.path.join(args.save_dir, key_path)
            else:
                save_dir = eval_subdir
            metrics = get_style_loss(eval_subdir,
                                     ref_subdir,
                                     args.batch_size,
                                     save_dir=save_dir)

        update_results(key_path, metrics, existing_df, results)

        if len(metrics) > 0:
            n_updated += 1

        if n_updated >= args.write_every:

            new_df = pd.DataFrame(results)
            existing_df = pd.concat([existing_df, new_df])
            existing_df.to_csv(args.metric_csv, index=False)
            results = []
            n_updated = 0

    new_df = pd.DataFrame(results)
    combined_df = pd.concat([existing_df, new_df])
    combined_df.to_csv(args.metric_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Text-Image Similarity using CLIP Features")
    parser.add_argument("--ref_dir",
                        type=str,
                        required=True,
                        help="Directory with ref image features")
    parser.add_argument("--eval_dir",
                        type=str,
                        required=True,
                        help="Directory with eval image features")
    parser.add_argument("--save_dir",
                        type=str,
                        default=None,
                        help="Directory to save per image style loss")
    parser.add_argument(
        "--compare_with_dataset",
        action='store_true',
        help="Compare with dataset instead of images from base model")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for feature encoding")
    parser.add_argument("--extra_level",
                        type=int,
                        default=1,
                        help="Additional level on top of the dataset path")
    parser.add_argument(
        "--extra_descr",
        type=str,
        default=None,
        help="Additional description to add at the beginning for csv")
    parser.add_argument("--metric_csv",
                        type=str,
                        default="style_losses.csv",
                        help="Name of the output CSV file")
    parser.add_argument("--write_every",
                        type=int,
                        default=10,
                        help="Write to csv frequency")

    args = parser.parse_args()
    main(args)
