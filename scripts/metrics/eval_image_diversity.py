import os
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

from utils import get_all_subdirectories, get_relevant_directories
from utils import load_image_features_in_out
from utils import should_compute_metric
from utils import update_results

from vendi import compute_vendi_score
from vendi import compute_per_prompt_vendi_scores


def get_im_diversity_metrics(eval_dir,
                             encoder_name,
                             n_images_per_out_prompt,
                             key_path=None,
                             existing_df=None,
                             overwrite=False):

    eval_features_dict = load_image_features_in_out(eval_dir, encoder_name)
    metrics = {}

    if 'in' in eval_features_dict:
        in_features = eval_features_dict['in']
        metric_name = f"vendi ({encoder_name}) (in)"
        if should_compute_metric(key_path, metric_name, existing_df,
                                 overwrite):
            vendi_score = compute_vendi_score(in_features)
            metrics[metric_name] = vendi_score
        else:
            print(f"{key_path}: {metric_name} exists, skip")

    if 'out' in eval_features_dict:
        out_features = eval_features_dict['out']
        metric_name = f"vendi ({encoder_name}) (out)"
        if should_compute_metric(key_path, metric_name, existing_df,
                                 overwrite):
            vendi_per_prompt = compute_per_prompt_vendi_scores(
                out_features, n_images_per_out_prompt)
            metrics[metric_name] = np.mean(vendi_per_prompt)
        else:
            print(f"{key_path}: {metric_name} exists, skip")

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
            print(f"Warning: The file {args.csv} is empty!")
    else:
        existing_df = pd.DataFrame()

    results = []
    eval_subdirs = get_all_subdirectories(args.eval_dir)

    for subdir in tqdm(eval_subdirs, desc="Evaluating"):
        eval_subdir, _, key_path = get_relevant_directories(
            subdir, args.eval_dir, '', args.extra_level, args.extra_descr)

        metrics = get_im_diversity_metrics(eval_subdir, args.encoder_name,
                                           args.n_images_per_out_prompt,
                                           key_path, existing_df,
                                           args.overwrite)

        update_results(key_path, metrics, existing_df, results)

    new_df = pd.DataFrame(results)

    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df])
    else:
        combined_df = new_df

    combined_df.to_csv(args.metric_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Text-Image Similarity using CLIP Features")
    parser.add_argument("--eval_dir",
                        type=str,
                        required=True,
                        help="Directory with eval image features")
    parser.add_argument("--encoder_name",
                        type=str,
                        default='clip-L-14',
                        help="Choice of encoder")
    parser.add_argument("--n_images_per_out_prompt",
                        type=int,
                        default=10,
                        help="Number of images for each out dist prompt")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite even the metric already exists in csv")
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
                        default="similarity_results.csv",
                        help="Name of the output CSV file")

    args = parser.parse_args()
    main(args)
