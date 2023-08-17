import os
import pandas as pd
import numpy as np


def get_all_subdirectories(root_dir, target_file=None):
    """
    Return a list of subfolders containing the specified file.

    Parameters:
    - root_dir: the directory from where the search should start.
    - target_file: the name of the file to search for.

    Returns:
    - A list of subfolders containing the target file.
    """
    subfolders_with_target = []

    for root, dirs, files in os.walk(root_dir):
        if target_file is None or target_file in files:
            subfolders_with_target.append(root)

    return subfolders_with_target


def get_relevant_directories(eval_subdir,
                             eval_dir,
                             ref_dir,
                             extra_level,
                             extra_descr=None):

    eval_relpath_full = os.path.relpath(eval_subdir, eval_dir)

    eval_relpath = eval_relpath_full.split(os.path.sep)[extra_level:]
    eval_relpath = os.path.sep.join(eval_relpath)

    ref_subdir = os.path.join(ref_dir, eval_relpath)

    if extra_descr is not None:
        key_path = os.path.join(extra_descr, eval_relpath_full)
    else:
        key_path = eval_relpath_full

    return eval_subdir, ref_subdir, key_path


def load_image_features(path, encoder_name, padding=False):
    features_dict = {}
    if os.path.exists(path):
        feature_npz = np.load(path)
        if encoder_name in feature_npz.files:
            features_dict['crop'] = feature_npz[encoder_name]
        if padding and (encoder_name + '-padding') in feature_npz.files:
            features_dict['padding'] = feature_npz[encoder_name + '-padding']
    return features_dict


def load_image_features_in_out(eval_dir, encoder_name):

    in_dist_features = "in_dist_prompts-image-features.npz"
    out_dist_features = "out_dist_prompts-image-features.npz"

    eval_features_dict = {}

    eval_features_in_dict = load_image_features(
        os.path.join(eval_dir, in_dist_features), encoder_name)
    if 'crop' in eval_features_in_dict:
        eval_features_dict['in'] = eval_features_in_dict['crop']

    eval_features_out_dict = load_image_features(
        os.path.join(eval_dir, out_dist_features), encoder_name)
    if 'crop' in eval_features_out_dict:
        eval_features_dict['out'] = eval_features_out_dict['crop']

    return eval_features_dict


def update_results(key_path, metrics, existing_df, results):

    # Update existing CSV or add new data
    if len(metrics) == 0:
        return

    if (not existing_df.empty and key_path in existing_df["Folder"].values):
        for metric_name in metrics:
            similarity = metrics[metric_name]
            existing_df.loc[existing_df["Folder"] == key_path,
                            metric_name] = similarity
    else:
        results.append({"Folder": key_path, **metrics})


def should_compute_metric(key_path, metric_name, existing_df, overwrite):

    # If overwrite flag is set, we should compute
    if (key_path is None or existing_df is None or existing_df.empty
            or overwrite):
        return True

    # If key_path is not present, we should compute
    if key_path not in existing_df["Folder"].values:
        return True

    # Check if the specific metric field is missing or has NaN/0 value
    row = existing_df[existing_df["Folder"] == key_path].iloc[0]
    if metric_name not in row or pd.isna(
            row[metric_name]) or row[metric_name] == 0:
        return True

    # If none of the above conditions met, then we shouldn't compute
    return False
