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


def create_out_dist_prompts(directory, token_mapping,
                            template_file, save_name, tag_format):
    template_data = {}

    # Collect all template data
    for root, _, files in os.walk(directory):
        if template_file in files:
            with open(os.path.join(root, template_file), 'r') as f:
                template_data[root] = f.readlines()
    print(template_data)

    for root, dirs, _ in os.walk(directory):
        # Only process deepest directories
        if not dirs:
            root_rel = os.path.relpath(root, directory)
            # print(root_rel)
            path_components = root_rel.split(os.path.sep)[1:]
            trigger_word = construct_trigger_word(
                path_components, token_mapping)

            # Find the closest tag template
            current_root = root
            while (current_root not in template_data
                    and current_root != directory.rstrip('/')):
                current_root = os.path.dirname(current_root)

            if current_root in template_data:

                # If we found a template, use it
                out_file_path = os.path.join(root, save_name)
                with open(out_file_path, 'w') as f_out:
                    if tag_format:
                        for line in template_data[current_root]:
                            trigger_word_split = trigger_word.split(',')
                            to_write = ','.join(
                                trigger_word_split
                                + line.split(',')[len(trigger_word_split):])
                            f_out.write(to_write)
                    else:
                        for line in template_data[current_root]:
                            f_out.write(line.replace("{}", trigger_word))


def main(args):
    token_mapping = read_token_mapping(args.token_mapping)
    create_out_dist_prompts(
        args.directory, token_mapping,
        args.template_name, args.save_name, args.tag_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process template files.")
    parser.add_argument("--directory", help="Target directory")
    parser.add_argument("--token_mapping", help="CSV file for token mapping")
    parser.add_argument("--template_name",
                        type=str,
                        default='template.txt',
                        help="name of the template file")
    parser.add_argument("--save_name",
                        type=str,
                        default='out_dist_prompts.txt',
                        help="name of the template file")
    parser.add_argument("--tag_format",
                        action='store_true',
                        help="whether the template has tag format")
    args = parser.parse_args()
    main(args)
