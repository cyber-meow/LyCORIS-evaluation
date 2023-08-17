import os
import argparse
from tqdm import tqdm

import numpy as np


def get_split(n_total, proportion):
    proportion = np.array(proportion)
    n_per_portion = n_total / np.sum(proportion)
    split = (proportion * n_per_portion).astype(int)
    remaining = n_total - np.sum(split)
    split[:remaining] += 1
    return split


def load_features_from_file(file_path):
    """Load features from a single .npz file and return as a dictionary."""
    with np.load(file_path) as data:
        return {key: data[key] for key in data.files}


def load_features_from_dir(directory):
    """Load all features from the directory."""
    features_dicts = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == 'image-features.npz':
                file_path = os.path.join(root, file)
                features_dicts.append(load_features_from_file(file_path))
    return concatenate_features(features_dicts)


def concatenate_features(features_dicts):
    """Concatenate a list of feature dictionaries into one dictionary."""
    concatenated_features_dict = {}
    for features_dict in features_dicts:
        for key, value in features_dict.items():
            if key not in concatenated_features_dict:
                concatenated_features_dict[key] = [value]
            else:
                concatenated_features_dict[key].append(value)

    # Convert lists to numpy arrays
    for key in concatenated_features_dict:
        concatenated_features_dict[key] = np.vstack(
            concatenated_features_dict[key])

    return concatenated_features_dict


def get_features_count(file_path):
    """Count the number of features in a .npz file."""
    with np.load(file_path) as data:
        # Assuming the first key is representative of all
        key = data.files[0]
        return data[key].shape[0]


def balance_single_dict_features(features_dict, max_count):
    """Balance the features of a single dictionary for each key."""
    balanced_dict = {}
    for key, features in features_dict.items():
        repeat_count = max_count // len(features)
        extra_samples = max_count % len(features)
        repeated_features = np.tile(features, (repeat_count, 1))
        extra_features = features[:extra_samples]
        balanced_dict[key] = np.vstack([repeated_features, extra_features])
    return balanced_dict


def load_and_balance_ref_features(ref_dir, n_eval_images_per_class):
    features_list = []
    max_count = 0
    eval_counts = {}

    # Traverse the second-level directories for classes
    for first_level_dir in os.listdir(ref_dir):
        first_level_path = os.path.join(ref_dir, first_level_dir)
        if os.path.isdir(first_level_path):
            for class_dir in os.listdir(first_level_path):
                class_path = os.path.join(first_level_path, class_dir)
                if not os.path.isdir(class_path):
                    continue

                # Load features for the class
                class_features_dict = load_features_from_dir(class_path)
                features_list.append(class_features_dict)

                # Number of evaluate images to use
                subfolder_counts = []
                for subfolder in os.listdir(class_path):
                    subfolder_path = os.path.join(class_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        count = get_features_count(
                            os.path.join(subfolder_path, 'image-features.npz'))
                        subfolder_counts.append(count)
                if subfolder_counts:
                    splits = get_split(n_eval_images_per_class,
                                       subfolder_counts)
                    for subfolder, split_count in zip(os.listdir(class_path),
                                                      splits):
                        eval_counts[os.path.join(class_dir,
                                                 subfolder)] = split_count
                else:
                    eval_counts[class_dir] = n_eval_images_per_class

    # Get max_count for balancing
    max_count = max(
        [arr.shape[0] for dict_ in features_list for arr in dict_.values()])

    # Balance features for each dictionary in the list
    for idx, features_dict in enumerate(features_list):
        features_list[idx] = balance_single_dict_features(
            features_dict, max_count)

    # Concatenate all features into one dictionary
    final_features_dict = concatenate_features(features_list)

    return final_features_dict, eval_counts


def get_target_folders(eval_dir, extra_level):
    """Determine the folders to process based on extra_level."""
    target_folders = [eval_dir]
    for i in range(extra_level):
        next_level_folders = []
        for folder in target_folders:
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if os.path.isdir(subfolder_path):
                    next_level_folders.append(subfolder_path)
        target_folders = next_level_folders
    return target_folders


def load_and_balance_eval_features(eval_dir, extra_level, eval_counts):

    eval_features_dict = {}
    target_folders = get_target_folders(eval_dir, extra_level)

    for folder in tqdm(target_folders):
        features_list = []
        for root, _, files in os.walk(folder):
            if 'in_dist_prompts-image-features.npz' in files:
                file_path = os.path.join(root,
                                         'in_dist_prompts-image-features.npz')
                features_dict = load_features_from_file(file_path)
                relative_path = os.path.relpath(root, eval_dir)
                relative_path = os.path.join(
                    *relative_path.split(os.path.sep)[extra_level+1:])

                if relative_path not in eval_counts:
                    raise ValueError(f"Relative path '{relative_path}' " +
                                     "not found in eval_counts.")

                desired_count = eval_counts[relative_path]
                balanced_features = balance_single_dict_features(
                    features_dict, desired_count)
                features_list.append(balanced_features)
        eval_features_dict[folder] = concatenate_features(features_list)

    return eval_features_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Balance features for classes")
    parser.add_argument(
        '--ref_dir',
        required=True,
        help='Reference directory containing class subdirectories')
    parser.add_argument('--eval_dir',
                        required=False,
                        help='Evaluation directory')
    parser.add_argument('--n_eval_images_per_class',
                        default=100,
                        type=int,
                        help='Number of evaluation images per class')
    parser.add_argument('--extra_level',
                        default=1,
                        help='Extra directory level for eval directory')

    args = parser.parse_args()

    print("Processing reference features...")
    balanced_features, eval_counts = load_and_balance_ref_features(
        args.ref_dir, args.n_eval_images_per_class)
    np.savez(os.path.join(args.ref_dir, 'fd-image-features.npz'),
             **balanced_features)

    print("Processing evaluation features...")
    eval_features_dict = load_and_balance_eval_features(
        args.eval_dir, args.extra_level, eval_counts)
    print("Saving evaluation features...")
    for folder_path in tqdm(eval_features_dict):
        np.savez(os.path.join(folder_path, 'fd-image-features.npz'),
                 **eval_features_dict[folder_path])
