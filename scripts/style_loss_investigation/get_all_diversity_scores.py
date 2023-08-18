import os
import argparse
import gc
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.nn.functional import normalize


def compute_vendi_and_dissim(X):
    X = normalize(X, dim=1)
    n = X.shape[0]
    S = X @ X.T
    S = S.to(torch.float32)
    # print('similarity matrix of shape {}'.format(S.shape))
    w = torch.linalg.eigvalsh(S / n)
    vendi = torch.exp(entropy_q(w))
    dissim = 1 - torch.mean(S)
    return vendi.cpu().item(), dissim.cpu().item()


def entropy_q(p):
    p_ = p[p > 0]
    return -(p_ * torch.log(p_)).sum()


def compute_std(X):
    mean = torch.mean(X, axis=0, keepdims=True)
    std = torch.sqrt(torch.sum((X - mean)**2) / (X.shape[0] - 1))
    return std.cpu().item()


def get_diversity_score_aux(features):
    """
    Compute the diversity score for a set of features.

    Args:
        - features (np.ndarray): Features extracted from an encoder.

    Returns:
        - scores (dict): Dictionary of diversity scores for the features.
    """
    vendi, dissim = compute_vendi_and_dissim(features)
    torch.cuda.empty_cache()
    std = compute_std(features)
    torch.cuda.empty_cache()
    scores = {"Vendi": vendi, "Dissimilarity": dissim, "Std": std}
    return scores


def get_diversity_scores(features_dict):
    """
    Compute diversity scores for a dictionary of features.

    Args:
        - features_dict (dict):
            Dictionary containing features from different encoders.

    Returns:
        - combined_scores (dict): Aggregated diversity scores.
    """
    combined_scores = {}

    for encoder_name, features in features_dict.items():
        encoder_scores = get_diversity_score_aux(features)

        for metric_name, score in encoder_scores.items():
            combined_key = f"{metric_name} ({encoder_name})"
            combined_scores[combined_key] = score

    return combined_scores


def compute_diversity_scores(features, n_max, n_eval_samples):
    """
    Compute diversity scores for a given set of features.

    Args:
        - features (dict): Features extracted from different encoders.
        - n_max (int): Maximum number of samples used to compute scores.
        - n_eval_samples (list):
            List of sample sizes for which scores should be computed.

    Returns:
        - results (list): List of tuples (n, scores_dict).
    """
    results = []

    encoder_names = list(features.keys())
    num_features = len(features[encoder_names[0]])
    print(num_features)
    if num_features <= n_max:
        diversity_scores = get_diversity_scores(features)
        results.append((num_features, diversity_scores))

    # Compute scores for subsets of samples
    for n in n_eval_samples:
        if num_features > n:
            subset_features = {}
            for encoder in encoder_names:
                if num_features != features[encoder].shape[0]:
                    print('Error')
                    exit(1)
                subset_features[encoder] = features[encoder][np.random.choice(
                    num_features, n, replace=False)]
            subset_scores = get_diversity_scores(subset_features)
            results.append((n, subset_scores))

    return results


def concatenate_and_resample(features_dicts,
                             n_cache_max=None,
                             proportions=None):
    concatenated_features_dict = {}

    # Initial concatenation
    for features_dict in features_dicts:
        for key, value in features_dict.items():
            if key not in concatenated_features_dict:
                concatenated_features_dict[key] = [value]
            else:
                concatenated_features_dict[key].append(value)

    # Convert lists to tensors
    for key in concatenated_features_dict:
        concatenated_features_dict[key] = torch.vstack(
            concatenated_features_dict[key])

    # If n_cache_max is provided, resample based on proportions
    if n_cache_max:
        expanded_weights = []
        for proportion, features_dict in zip(proportions, features_dicts):
            n_features = len(next(iter(features_dict.values())))
            per_feature_weight = proportion / n_features
            expanded_weights.extend([per_feature_weight] * n_features)
        expanded_weights = expanded_weights / np.sum(expanded_weights)

        for key in concatenated_features_dict:
            feature_matrix = concatenated_features_dict[key]
            sample_indices = np.random.choice(len(feature_matrix),
                                              size=n_cache_max,
                                              replace=False,
                                              p=expanded_weights)
            concatenated_features_dict[key] = feature_matrix[sample_indices]

    # Empty CUDA cache
    torch.cuda.empty_cache()
    gc.collect()

    return concatenated_features_dict


def update_results(key_path, count, metrics, existing_df, results):

    # Update existing CSV or add new data
    if len(metrics) == 0:
        return

    if existing_df.empty:
        results.append({"Folder": key_path, "Count": count, **metrics})
    else:
        mask = (existing_df["Folder"] == key_path) & (existing_df["Count"]
                                                      == count)
        if mask.any():
            for metric_name in metrics:
                similarity = metrics[metric_name]
                existing_df.loc[mask, metric_name] = similarity
        else:
            results.append({"Folder": key_path, "Count": count, **metrics})


def load_npz_as_torch(file_path, n_cache_max=None):
    """
    Load the contents of an .npz file, convert each numpy array
    to a PyTorch tensor with float16 precision.

    Args:
    - file_path (str): Path to the .npz file.
    - n_cache_max (int, optional): Maximum number of items to retain.

    Returns:
    - dict: A dictionary containing the same keys as in the .npz file,
        but with values as torch tensors.
    """

    # Load numpy arrays from .npz file
    numpy_data = dict(np.load(file_path, allow_pickle=True).items())

    # Check if sampling based on n_cache_max is required
    if n_cache_max is not None:
        total_size = len(next(iter(numpy_data.values())))
        if total_size > n_cache_max:
            numpy_data = {
                k: v[np.random.choice(v.shape[0], n_cache_max, replace=False)]
                for k, v in numpy_data.items()
            }

    # Convert numpy arrays to torch tensors
    torch_data = {k: torch.tensor(v).cuda() for k, v in numpy_data.items()}

    # Empty CUDA cache
    torch.cuda.empty_cache()
    gc.collect()

    return torch_data


def dfs_diversity(folder_path,
                  root_dir,
                  n_min,
                  n_max,
                  n_cache_max,
                  n_eval_samples,
                  existing_df,
                  results,
                  pbar=None):
    """
    Depth-first search for folder traversal and computing diversity scores.

    Args:
        - folder_path (str): Path of the current folder.
        - root_dir (str): Path to the root directory.
        - n_min (int): Minimum number of samples required to compute scores.
        - n_max (int): Maximum number of samples used to compute scores.
        - n_eval_samples (list):
            List of sample sizes for which scores should be computed.

    Returns:
        - concatenated_features (dict):
            Features concatenated across all subdirectories.
    """
    total_images = 0
    features_from_subdirs = []
    proportions = []

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            sub_features, n_images = dfs_diversity(item_path, root_dir, n_min,
                                                   n_max, n_cache_max,
                                                   n_eval_samples, existing_df,
                                                   results, pbar)
            features_from_subdirs.append(sub_features)
            proportions.append(n_images)
        else:
            n_images = 0

        # If we are at a directory with an image-feature.npz file
        if 'image-features.npz' in os.listdir(folder_path):
            file_path = os.path.join(folder_path, 'image-features.npz')
            file_features = load_npz_as_torch(file_path, n_cache_max)
            n_images += len(next(iter(file_features.values())))
            features_from_subdirs.append(file_features)

        proportions.append(n_images)
        total_images += n_images

        # If we exceed n_cache_max, we perform resampling
        if n_cache_max is not None and total_images > n_cache_max:
            features_from_subdirs = concatenate_and_resample(
                features_from_subdirs, n_cache_max, proportions)
            features_from_subdirs = [features_from_subdirs]
            proportions = [total_images]

    # Assume there are always feaures
    if len(features_from_subdirs) > 1:
        concatenated_features = concatenate_and_resample(features_from_subdirs)
    else:
        concatenated_features = features_from_subdirs[0]

    # Compute diversity scores
    encoder_names = list(concatenated_features.keys())
    if len(encoder_names) == 0:
        if pbar:
            pbar.update(1)
        return concatenated_features
    num_features = len(concatenated_features[encoder_names[0]])

    # Compute scores if there are enough features
    if num_features >= n_min:
        print(folder_path)
        scores_list = compute_diversity_scores(concatenated_features, n_max,
                                               n_eval_samples)
        # Update the results DataFrame
        for n, scores in scores_list:
            key_path = os.path.relpath(folder_path, root_dir)
            update_results(key_path, n, scores, existing_df, results)
    if pbar:
        pbar.update(1)
    return concatenated_features, total_images


def main(args):
    # Check if the CSV already exists
    if os.path.exists(args.output_csv):
        try:
            existing_df = pd.read_csv(args.output_csv)
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()
            print(f"Warning: The file {args.output_csv} is empty!")
    else:
        existing_df = pd.DataFrame()

    results = []

    total_dirs = 1
    for _, dirnames, _ in os.walk(args.root_dir):
        total_dirs += len(dirnames)

    with tqdm(total=total_dirs, desc="Processing directories",
              unit="dir") as pbar:
        dfs_diversity(args.root_dir, args.root_dir, args.n_min, args.n_max,
                      args.n_cache_max, args.n_eval_samples, existing_df,
                      results, pbar)

    # After DFS, create and/or update the CSV
    new_df = pd.DataFrame(results)
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df])
    else:
        combined_df = new_df

    combined_df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute diversity scores")
    parser.add_argument('--root_dir',
                        type=str,
                        help="Root directory to start DFS from.")
    parser.add_argument(
        '--n_min',
        default=50,
        type=int,
        help="Minimum number of samples required to compute scores.")
    parser.add_argument(
        '--n_max',
        default=9999,
        type=int,
        help="Maximum  number of samples used to compute scores.")
    parser.add_argument(
        '--n_cache_max',
        default=None,
        type=int,
        help="Maximum  number of samples to cache during random sampling.")
    parser.add_argument(
        '--n_eval_samples',
        default=None,
        type=str,
        help="Comma-separated sample sizes for which scores are computed.")
    parser.add_argument('--output_csv',
                        type=str,
                        default='all_diversity.csv',
                        help="Path to save CSV with scores.")

    args = parser.parse_args()

    if args.n_eval_samples is not None:
        try:
            args.n_eval_samples = list(map(int,
                                           args.n_eval_samples.split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Input should be comma-separated integers")
    if args.n_cache_max is not None:
        assert args.n_cache_max >= args.n_max

    main(args)
