import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils import get_relevant_directories
from utils import update_results
from metrics.scorers import LiqeScorer, ManiqaScorer, ArtifactScorer

from lycoris_eval.data_utils.image_dataset import ImageDataset

scorers = {
    'liqe': {
        'model': LiqeScorer,
        'path': './checkpoints/LIQE.pt',
        'n_patches': 10,
        'image_size': 512,
    },
    'maniqa': {
        'model': ManiqaScorer,
        'path': './checkpoints/ckpt_koniq10k.pt',
        'n_patches': 10,
        'image_size': 512,
    },
    'artifact': {
        'model': ArtifactScorer,
        'path':
        './checkpoints/aesthetics_scorer_artifacts_openclip_vit_l_14.pth',
        'n_patches': 1,
        'image_size': 224,
    },
}


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_image_scores(image_paths,
                         scorer,
                         generated,
                         n_patches=1,
                         batch_size=8,
                         autocast=True,
                         device='cuda'):

    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, scorer.transform, generated)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=16,
                            shuffle=False)

    all_keys = []
    all_scores = []

    with torch.no_grad(), torch.autocast(device_type=device, enabled=autocast):
        for keys, images in dataloader:
            images = images.to(device)
            scores = scorer(images, n_patches=n_patches)
            if not torch.all(torch.isfinite(scores)):
                raise ValueError('Invalid scores detected')
            all_keys.extend(keys)
            all_scores.append(scores)

    all_scores = torch.cat(all_scores, dim=0)

    # Sort features based on keys
    sorted_indices = sorted(range(len(all_keys)), key=lambda k: all_keys[k])
    sorted_scores = all_scores[sorted_indices]

    return sorted_scores.cpu()


def main(args):

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect all subdirectories that contain images
    subdirs_with_images = []
    for subdir, _, files in os.walk(args.eval_dir):
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
               for file in files):
            subdirs_with_images.append(subdir)

    with tqdm(total=len(subdirs_with_images) * len(args.scorer_names),
              desc="Scoring") as pbar:
        for scorer_name in args.scorer_names:  # Loop over each scorer
            # Initialize scorer for the current scorer_name
            print(f"Set up scorer {scorer_name}...")
            scorer_config = scorers[scorer_name]
            scorer = scorer_config['model'](
                model_path=scorer_config['path'],
                image_size=scorer_config['image_size'],
                resize_mode=args.resize_mode,
                device=device)
            n_patches = scorer_config['n_patches']
            score_key = scorer_name + f"-{args.resize_mode}"

            for subdir in subdirs_with_images:

                if args.generated:
                    save_path = os.path.join(
                        os.path.dirname(subdir),
                        f"{os.path.basename(subdir)}-image-scores.npz")
                else:
                    save_path = os.path.join(subdir, 'image-scores.npz')
                # This is to deal with a bug in felix_vallotton_stroke
                save_path = save_path.replace('captions', 'prompts')

                save_path = os.path.join(
                    args.dst_dir, os.path.relpath(save_path, args.eval_dir))

                if os.path.exists(save_path):
                    npz = np.load(save_path)
                    image_scores_all = dict(npz.items())
                else:
                    image_scores_all = {}

                if score_key in image_scores_all.keys() and not args.overwrite:
                    print(f"{score_key} already in {save_path}, skip")

                else:
                    image_paths = [
                        os.path.join(subdir, file)
                        for file in os.listdir(subdir)
                        if file.lower().endswith(('.png', '.jpg', '.jpeg',
                                                  '.webp'))
                    ]
                    image_scores = compute_image_scores(image_paths,
                                                        scorer,
                                                        args.generated,
                                                        n_patches,
                                                        args.batch_size,
                                                        (not args.no_autocast),
                                                        device=device)
                    image_scores_all[score_key] = image_scores.numpy()

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(save_path, **image_scores_all)

                eval_subdir, _, key_path = get_relevant_directories(
                    subdir, args.eval_dir, '', args.extra_level,
                    args.extra_descr)

                score_dict = {
                    score_key.capitalize():
                    np.mean(image_scores_all[score_key])
                }

                update_results(key_path, score_dict, existing_df, results)

                pbar.update(1)

            new_df = pd.DataFrame(results)
            existing_df = pd.concat([existing_df, new_df])
            existing_df.to_csv(args.metric_csv, index=False)
            results = []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Encode image features for subdirectories.")
    parser.add_argument("--eval_dir",
                        type=str,
                        required=True,
                        help="Directory to process")
    parser.add_argument("--dst_dir",
                        type=str,
                        required=True,
                        help="Directory to save_features")
    parser.add_argument("--scorer_name",
                        type=str,
                        default=None,
                        help="Choice of scorer (overwritten by scorer_names)")
    parser.add_argument(
        "--scorer_names",
        type=str,
        default=None,
        help="Comma-separated list of scorers (e.g. scorer1, scorer2)")
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
                        default=8,
                        help="Batch size for feature encoding")
    parser.add_argument("--no_autocast",
                        action="store_true",
                        help="Turn off autocast")
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
                        default="quality_results.csv",
                        help="Name of the output CSV file")

    args = parser.parse_args()

    if args.scorer_names is not None:
        args.scorer_names = args.scorer_names.split(',')
    elif args.scorer_name is not None:
        args.scorer_names = [args.scorer_name]
    else:
        raise ValueError(
            "Either --scorer_name or --scorer_names should be speciefied")

    main(args)
