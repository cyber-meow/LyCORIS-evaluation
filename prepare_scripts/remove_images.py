import os
import argparse


def remove_files(directory, img_exts=['.jpg', '.png', '.jpeg', '.webp', '.gif']):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file is an image and has a corresponding .txt
            if any(filename.endswith(ext) for ext in img_exts) and os.path.exists(os.path.join(root, filename.rsplit('.', 1)[0] + '.txt')):
                img_path = os.path.join(root, filename)
                txt_path = os.path.join(
                    root, filename.rsplit('.', 1)[0] + '.txt')

                os.remove(img_path)
                os.remove(txt_path)

            # Rename 'all_captions.txt' to 'in_dist_prompts.txt'
            if filename == "all_captions.txt":
                os.rename(os.path.join(root, filename),
                          os.path.join(root, "in_dist_prompts.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove images and their corresponding txt files, and rename 'all_captions.txt' to 'in_dist_prompts.txt'.")
    parser.add_argument("directory", help="Directory to process")
    args = parser.parse_args()

    remove_files(args.directory)
