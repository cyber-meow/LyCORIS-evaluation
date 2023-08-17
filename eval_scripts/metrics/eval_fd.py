import os
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

from utils import get_all_subdirectories
from utils import should_compute_metric
from utils import update_results

from fd import compute_efficient_FD_with_reps


def get_fd_scores(ref_features_npz,
                  eval_features_npz,
                  key_path=None,
                  existing_df=None,
                  overwrite=False):
    metrics = {}

    for key_name in ref_features_npz.files:
        if key_name.endswith('-padding'):
            encoder_name = key_name.rstrip('-padding')
        else:
            encoder_name = key_name
        if encoder_name in eval_features_npz.files:
            ref_features = ref_features_npz[key_name]
            eval_features = eval_features_npz[encoder_name]

            metric_name = f"FD ({key_name})"

            # Check if we should compute this specific metric
            if should_compute_metric(key_path, metric_name, existing_df,
                                     overwrite):
                fd_score = compute_efficient_FD_with_reps(
                    ref_features, eval_features)
                metrics[metric_name] = fd_score
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

    ref_features_npz = np.load(args.ref_features)

    results = []
    eval_subdirs = get_all_subdirectories(args.eval_dir,
                                          target_file='fd-image-features.npz')

    for subdir in tqdm(eval_subdirs, desc="Evaluating"):

        eval_file = os.path.join(subdir, 'fd-image-features.npz')
        eval_features_npz = np.load(eval_file)

        key_path = os.path.relpath(subdir, args.eval_dir)
        metrics = get_fd_scores(ref_features_npz, eval_features_npz, key_path,
                                existing_df, args.overwrite)
        if args.extra_descr is not None:
            key_path = os.path.join(args.extra_descr, key_path)
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
    parser.add_argument("--ref_features",
                        type=str,
                        required=True,
                        help="File with ref image features")
    parser.add_argument("--eval_dir",
                        type=str,
                        required=True,
                        help="Directory with eval image features")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite even the metric already exists in csv")
    parser.add_argument(
        "--extra_descr",
        type=str,
        default=None,
        help="Additional description to add at the beginning for csv")
    parser.add_argument("--metric_csv",
                        type=str,
                        default="fd.csv",
                        help="Name of the output CSV file")

    args = parser.parse_args()
    main(args)
