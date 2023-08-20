import os
import numpy as np
import argparse


def combine_npz(dir1, dir2, output_dir):
    # Get the set of all npz files from both directories recursively
    npz_files_dir1 = {
        os.path.relpath(os.path.join(dp, f), dir1)
        for dp, dn, filenames in os.walk(dir1)
        for f in filenames if f.endswith('.npz')
    }
    npz_files_dir2 = {
        os.path.relpath(os.path.join(dp, f), dir2)
        for dp, dn, filenames in os.walk(dir2)
        for f in filenames if f.endswith('.npz')
    }

    # Combine both sets
    all_npz_files = npz_files_dir1.union(npz_files_dir2)

    for rel_path in all_npz_files:
        npz_path1 = os.path.join(dir1, rel_path)
        npz_path2 = os.path.join(dir2, rel_path)
        output_path = os.path.join(output_dir, rel_path)

        # If the npz exists only in one of the directories,
        # copy it to the output
        if os.path.exists(npz_path1) and not os.path.exists(npz_path2):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, **dict(np.load(npz_path1,
                                                 allow_pickle=True)))

        elif os.path.exists(npz_path2) and not os.path.exists(npz_path1):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, **dict(np.load(npz_path2,
                                                 allow_pickle=True)))

        # If it exists in both, combine the contents
        else:
            data1 = dict(np.load(npz_path1, allow_pickle=True))
            data2 = dict(np.load(npz_path2, allow_pickle=True))

            # Merge dictionaries.
            # If there are conflicts, data from dir2 overwrites data from dir1
            combined_data = {**data1, **data2}

            # Ensure the output directory for this file exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save the combined data
            np.savez(output_path, **combined_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine .npz files from two' +
        ' directories with relative paths.')
    parser.add_argument(
        '--dir1',
        type=str,
        required=True,
        help='Path to the first directory containing .npz files.')
    parser.add_argument(
        '--dir2',
        type=str,
        required=True,
        help='Path to the second directory containing .npz files.')
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Path to the output directory.')

    args = parser.parse_args()
    combine_npz(args.dir1, args.dir2, args.output_dir)
