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


def update_text_similarity_metrics(
        text_feature_path, image_feature_dict,
        prompt_type, metrics, eval_dir):
    if os.path.exists(text_feature_path):
        if prompt_type in image_feature_dict:
            text_features = np.load(text_feature_path)
            image_features = image_feature_dict[prompt_type]
            similarity = compute_cosine_similarity(text_features,
                                                   image_features)
            metrics[f'Text Similarity ({prompt_type})'] = similarity
        else:
            print(f"Warning: in prompt features unfound in {eval_dir}")


def get_text_similarity_metrics(eval_dir, ref_dir):

    metrics = {}

    image_feacture_dict = load_image_features_in_out(eval_dir, 'clip-L-14')

    text_feature_paths = [
        os.path.join(
            ref_dir, "in_dist_prompts-clip-text-features.npy"),
        os.path.join(
            ref_dir, "out_dist_prompts-clip-text-features.npy"),
        os.path.join(
            ref_dir, "triggeronly-clip-text-features.npy")
    ]
    prompt_types = ['in', 'out', 'trigger']

    comb = zip(text_feature_paths, prompt_types)
    for text_feature_path, prompt_type in comb:
        update_text_similarity_metrics(
            text_feature_path, image_feacture_dict,
            prompt_type, metrics, eval_dir)

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
    n_updated = 0
    eval_subdirs = get_all_subdirectories(args.eval_dir)

    for subdir in tqdm(eval_subdirs, desc="Evaluating"):
        eval_subdir, ref_subdir, key_path = get_relevant_directories(
            subdir, args.eval_dir, args.ref_dir, args.extra_level,
            args.extra_descr)

        metrics = get_text_similarity_metrics(eval_subdir, ref_subdir)

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
    parser.add_argument("--write_every",
                        type=int,
                        default=1000,
                        help="Write to csv frequency")

    args = parser.parse_args()
    main(args)
