import os
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

from utils import get_all_subdirectories, get_relevant_directories
from utils import load_image_features, load_image_features_in_out
from utils import update_results


def compute_cosine_similarity_and_cent_dist2(arr_a, arr_b):
    """Compute average cosine similarity over all the cross-dataset pairs.
    Compute also distance of the centroid.
    """
    arr_a = arr_a / np.linalg.norm(arr_a, axis=1, keepdims=True)
    arr_b = arr_b / np.linalg.norm(arr_b, axis=1, keepdims=True)
    similarities = arr_a @ arr_b.T
    avg_score = similarities.mean()
    mean_a = np.mean(arr_a, axis=0)
    mean_b = np.mean(arr_b, axis=0)
    cent_dist2 = np.sum((mean_a - mean_b)**2)
    return avg_score, cent_dist2


def get_im_similarity_metrics(eval_dir, ref_dir, encoder_name):

    ref_features_dict = {}

    ref_feature_path = os.path.join(ref_dir, "image-features.npz")
    if os.path.exists(ref_feature_path):
        ref_features_dict = load_image_features(ref_feature_path, encoder_name)
    else:
        # Completely ignore it if reference image feature file does not exist
        return {}

    if len(ref_features_dict) == 0:
        print(f"Warning: Relevant features for {ref_dir} unfound")

    eval_features_dict = load_image_features_in_out(eval_dir, encoder_name)

    if len(eval_features_dict) == 0:
        print(f"Warning: Relevant features for {eval_dir} unfound")

    metrics = {}

    for ref_key in ref_features_dict:
        for eval_key in eval_features_dict:
            sim_results = compute_cosine_similarity_and_cent_dist2(
                ref_features_dict[ref_key], eval_features_dict[eval_key])
            names = ['Image Similarity', 'Squared Centroid Distance']
            for name, score in zip(names, sim_results):
                metric_name = (
                    f"{name} ({encoder_name}-{ref_key}) ({eval_key})")
                metrics[metric_name] = score

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

        metrics = get_im_similarity_metrics(eval_subdir, ref_subdir,
                                            args.encoder_name)

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
    parser.add_argument("--encoder_name",
                        type=str,
                        default='clip-L-14',
                        help="Choice of encoder")
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
    parser.add_argument("--write_every",
                        type=int,
                        default=1000,
                        help="Write to csv frequency")

    args = parser.parse_args()
    main(args)
