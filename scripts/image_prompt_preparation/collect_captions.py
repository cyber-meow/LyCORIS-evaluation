"""
This is used to generate the file of in_dist_prompts.txt
"""


import os
import argparse

from pathlib import Path


def collect_captions(input_dir):
    # Define the extensions of interest
    image_extensions = [".jpg", ".png", ".jpeg", ".webp", ".gif"]

    all_captions = {}

    # First, walk through the directory and subdirectories to collect captions
    for root, _, files in os.walk(input_dir):
        captions = []
        for filename in files:
            # Check if the file has one of the desired extensions
            if Path(filename).suffix in image_extensions:
                # Construct the .txt filename
                txt_filename = os.path.join(root, Path(filename).stem + ".txt")

                # If the .txt file exists, read its content
                if os.path.exists(txt_filename):
                    with open(txt_filename, "r") as f:
                        captions.append(f.read().strip())

        # Store the captions in the all_captions dictionary
        if captions:
            all_captions[root] = captions

    # Then, write captions to the all_captions.txt files and include in parent directories
    for dir_path, captions in all_captions.items():
        current_dir = dir_path
        while current_dir != input_dir:
            with open(os.path.join(current_dir, "all_captions.txt"), "a") as f:
                f.write("\n".join(captions) + "\n")
            current_dir = os.path.dirname(current_dir)
        with open(os.path.join(current_dir, "all_captions.txt"), "a") as f:
            f.write("\n".join(captions) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect captions from .txt files corresponding to images in directories and subdirectories.")
    parser.add_argument(
        "directory", help="Root directory to start searching from.")
    args = parser.parse_args()

    collect_captions(args.directory)
