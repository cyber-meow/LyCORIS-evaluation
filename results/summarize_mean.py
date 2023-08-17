import pandas as pd
import os
import argparse

from tqdm import tqdm


def average_up_tree(folder_name, df, cache, all_dirs):
    """Recursively compute average up the directory tree."""

    # If already computed, return cached result
    if folder_name in cache:
        return cache[folder_name]

    # Find subdirectories/siblings
    subdirs = [
        f for f in all_dirs
        if f.startswith(folder_name) and f != folder_name
    ]

    # If leaf node (no subdirectories)
    if not subdirs:
        scores = df[df["Folder"] == folder_name].iloc[0, 1:].to_dict()
        cache[folder_name] = scores
        return scores

    # If node has children, average their scores
    avg_scores = {}
    count = 0
    for subdir in subdirs:
        # Ensure not to double count
        if subdir.count(os.path.sep) == folder_name.count(os.path.sep) + 1:
            count += 1
            subdir_scores = average_up_tree(subdir, df, cache, all_dirs)
            for key, val in subdir_scores.items():
                avg_scores[key] = avg_scores.get(key, 0) + val

    for key in avg_scores:
        avg_scores[key] /= count

    cache[folder_name] = avg_scores
    return avg_scores


def get_all_unique_dirs(folders):
    """Get a list of all unique parent directories."""
    unique_dirs = set()
    for folder in folders:
        unique_dirs.add(folder)
        parent = os.path.dirname(folder)
        while parent:
            unique_dirs.add(parent)
            parent = os.path.dirname(parent)
    return list(unique_dirs)


def main(args):
    df = pd.read_csv(args.input_csv)
    unique_dirs = df["Folder"].values

    cache = {}
    results = []
    all_dirs = get_all_unique_dirs(unique_dirs)

    for folder in tqdm(all_dirs):
        average_up_tree(folder, df, cache, all_dirs)

    for folder, scores in cache.items():
        if folder not in unique_dirs:
            row = {"Folder": folder}
            row.update(scores)
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Folder")
    results_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average scores over directory tree structure")
    parser.add_argument("--input_csv",
                        type=str,
                        required=True,
                        help="Input CSV file with folder names and scores")
    parser.add_argument("--output_csv",
                        type=str,
                        default="averaged_scores.csv",
                        help="Output CSV file for averaged scores")

    args = parser.parse_args()
    main(args)
