"""
This is used to generate the file of out_dist_prompts.txt
"""

import os
import argparse
import csv


def read_token_mapping(csv_path):
    mapping = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            mapping[row[0]] = row[1]
    return mapping


def construct_trigger_word(path_components, token_mapping):
    trigger = token_mapping.get(path_components[0], path_components[0])

    if len(path_components) > 1 and path_components[1] != "none":
        trigger += f", {path_components[1]}"

    return f"{trigger}"


def create_out_dist_prompts(directory, token_mapping):
    template_data = {}
    template_tags_data = {}

    # Collect all template data
    for root, _, files in os.walk(directory):
        if 'template.txt' in files:
            with open(os.path.join(root, 'template.txt'), 'r') as f:
                template_data[root] = f.readlines()
        if 'template_tags.txt' in files:
            with open(os.path.join(root, 'template_tags.txt'), 'r') as f:
                template_tags_data[root] = f.readlines()

    for root, dirs, _ in os.walk(directory):
        # Only process deepest directories
        if not dirs:
            path_components = root.split(os.path.sep)[2:]
            trigger_word = construct_trigger_word(
                path_components, token_mapping)

            # Find the closest template
            current_root = root
            while current_root not in template_data and current_root != directory:
                current_root = os.path.dirname(current_root)

            # If we found a template, use it
            if current_root in template_data:
                out_file_path = os.path.join(root, 'out_dist_prompts.txt')
                with open(out_file_path, 'w') as f_out:
                    for line in template_data[current_root]:
                        f_out.write(line.replace("{}", trigger_word))

            # Find the closest tag template
            current_root = root
            while current_root not in template_tags_data and current_root != directory:
                current_root = os.path.dirname(current_root)

            # If we found a template, use it
            if current_root in template_tags_data:
                out_file_path = os.path.join(root, 'out_dist_prompts_tags.txt')
                with open(out_file_path, 'w') as f_out:
                    for line in template_tags_data[current_root]:
                        trigger_word_split = trigger_word.split(',')
                        to_write = ','.join(
                            trigger_word_split + line.split(',')[len(trigger_word_split):])
                        f_out.write(to_write)


def main(args):
    token_mapping = read_token_mapping(args.token_mapping)
    create_out_dist_prompts(args.directory, token_mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process directory and generate out_dist_prompts.txt files.")
    parser.add_argument("--directory", help="Target directory")
    parser.add_argument("--token_mapping", help="CSV file for token mapping")
    args = parser.parse_args()
    main(args)
