import csv
import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
import open_clip


def load_replacements(filename):
    """Load replacements from a CSV file."""
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        return {rows[0]: rows[1] for rows in reader}


def process_line(line, replacements):
    """Process a single line based on replacements and additional rules."""
    # Replace words from CSV
    for key, value in replacements.items():
        line = line.replace(key, value)

    # Remove leading and trailing whitespace, commas, and "of"
    line = line.strip(" ,")

    # Remove leading and trailing "of"
    if line.startswith("of "):
        line = line[3:]

    # Remove consecutive whitespaces
    line = ' '.join(line.split())

    # Replace consecutive commas separated by whitespaces
    line = line.replace(", , ", ", ")

    # note that new line does not affect the output of clip
    return line + "\n"


def main(args):

    replacements = load_replacements(args.eval_replace_file)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "cpu")  # Setting up the device

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # First, gather a list of all valid files we'll process
    valid_files = []
    for subdir, _, files in os.walk(args.src_dir):
        for file in files:
            if file.endswith('.txt') and 'template' not in file:
                valid_files.append(os.path.join(subdir, file))

    # Now we know the total number of valid files
    with tqdm(total=len(valid_files), desc="Processing") as pbar:
        for filepath in valid_files:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            new_lines = [process_line(line, replacements) for line in lines]
            n_prompts = len(new_lines)
            repeat_per_prompt = args.n_images // n_prompts

            new_lines_update = []
            for i, prompt in enumerate(new_lines):
                if i >= args.n_images:
                    break
                if i < args.n_images % n_prompts:
                    repeat = repeat_per_prompt + 1
                else:
                    repeat = repeat_per_prompt
                new_lines_update.extend([prompt] * repeat)

            with open(filepath + '-eval', 'w') as f:
                f.writelines(new_lines_update)

            text = tokenizer(new_lines_update).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = model.encode_text(text).cpu().numpy()
            save_path = (os.path.splitext(filepath)[0] +
                         '-clip-text-features.npy')
            save_path = os.path.join(args.dst_dir,
                                     os.path.relpath(save_path, args.src_dir))

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, text_features)

            pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Modify text files based on replacement CSV.")
    parser.add_argument("--eval_replace_file",
                        type=str,
                        help="Path to the eval_replace CSV file")
    parser.add_argument("--src_dir",
                        type=str,
                        help="Directory containing text files to encode")
    parser.add_argument("--dst_dir",
                        type=str,
                        help="Directory to save encoded features")
    parser.add_argument('--n_images',
                        type=int,
                        help='Number of images to generate per prompt file.')

    args = parser.parse_args()

    main(args)
