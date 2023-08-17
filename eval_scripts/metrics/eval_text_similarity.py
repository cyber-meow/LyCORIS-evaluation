import os
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from utils import get_all_subdirectories, get_relevant_directories
from utils import load_image_features_in_out
from utils import update_results


def compute_cosine_similarity(arr_a, arr_b):
    """Compute average cosine similarity between two tensors.
    The pairs are in 1-1 correspondance.
    """
    tensor_a, tensor_b = torch.tensor(arr_a), torch.tensor(arr_b)
    similarities = cosine_similarity(tensor_a, tensor_b)
    return similarities.mean().item()


def get_text_similarity_metrics(eval_dir, ref_dir):

    metrics = {}

    eval_features_dict = load_image_features_in_out(eval_dir, 'clip-L-14')

    text_feature_path_in = os.path.join(
        ref_dir, "in_dist_prompts-clip-text-features.npy")
    text_feature_path_out = os.path.join(
        ref_dir, "out_dist_prompts-clip-text-features.npy")

    if os.path.exists(text_feature_path_in):
        if 'in' in eval_features_dict:
            text_features = np.load(text_feature_path_in)
            image_features = eval_features_dict['in']
            similarity = compute_cosine_similarity(text_features,
                                                   image_features)
            metrics['Text Similarity (in)'] = similarity
        else:
            print(f"Warning: in prompt features unfound in {eval_dir}")

    if os.path.exists(text_feature_path_out):
        if 'out' in eval_features_dict:
            text_features = np.load(text_feature_path_out)
            image_features = eval_features_dict['out']
            similarity = compute_cosine_similarity(text_features,
                                                   image_features)
            metrics['Text Similarity (out)'] = similarity
        else:
            print(f"Warning: out prompt features unfound in {eval_dir}")

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
            print("Warning: The file is empty!")
    else:
        existing_df = pd.DataFrame()

    results = []
    eval_subdirs = get_all_subdirectories(args.eval_dir)

    for subdir in tqdm(eval_subdirs, desc="Evaluating"):
        eval_subdir, ref_subdir, key_path = get_relevant_directories(
            subdir, args.eval_dir, args.ref_dir, args.extra_level,
            args.extra_descr)

        metrics = get_text_similarity_metrics(eval_subdir, ref_subdir)

        update_results(key_path, metrics, existing_df, results)

    new_df = pd.DataFrame(results)

    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df]).fillna(0)
    else:
        combined_df = new_df

    combined_df.to_csv(args.metric_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Text-Image Similarity using CLIP Features")
    parser.add_argument("--ref_dir",
                        type=str,
                        required=True,
                        help="Directory with text features")
    parser.add_argument("--eval_dir",
                        type=str,
                        required=True,
                        help="Directory with image features")
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
