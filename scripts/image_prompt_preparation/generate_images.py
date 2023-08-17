import os
import argparse
import csv
import toml
import hashlib

import webuiapi


def generate_images(api,
                    configs,
                    prompts,
                    n_images,
                    dst_dir,
                    seed=1000,
                    lora_name=None):

    n_prompts = len(prompts)
    repeat_per_prompt = n_images // n_prompts
    num_digits = len(str(n_images))
    if 'seed' in configs:
        seed = configs['seed']
    new_configs = {
        k: v
        for k, v in configs.items() if k not in ["seed", "lora_weight"]
    }

    count = 0
    for i, prompt in enumerate(prompts):
        if i >= n_images:
            break
        if i < n_images % n_prompts:
            repeat = repeat_per_prompt + 1
        else:
            repeat = repeat_per_prompt
        if lora_name is not None:
            lora_weight = configs.get('lora_weight', 1)
            prompt += f' <lora:{lora_name}:{lora_weight}>'
        results = api.txt2img(prompt=prompt,
                              batch_size=repeat,
                              seed=seed,
                              **new_configs)
        infotexts = results.info['infotexts']
        for image, infotext in zip(results.images, infotexts):
            md5hash = hashlib.md5(image.tobytes()).hexdigest()
            formatted_value = "{:0{width}d}".format(count, width=num_digits)
            save_name = os.path.join(dst_dir, f"{formatted_value}-{md5hash}")
            image.save(save_name + '.webp')
            with open(save_name + '.txt', 'w') as f:
                f.write(infotext)
            count += 1
        seed += repeat


def generate_images_from_dir(api,
                             configs,
                             n_images,
                             eval_prompt_dir,
                             dst_dir,
                             lora_name=None):

    # Traverse through all the files in src_dir
    for subdir, _, files in os.walk(eval_prompt_dir):
        for file in files:
            # Check if the file is a txt file and
            # doesn't contain 'template' in its name
            if file.endswith('.txt') and 'template' not in file:
                # Create the full path for the file
                full_file_path = os.path.join(subdir, file)

                # Read the prompts from the file
                with open(full_file_path, 'r') as f:
                    prompts = [line.strip() for line in f.readlines()]

                # Create a new directory in dst_dir
                relative_path = os.path.relpath(
                    subdir, eval_prompt_dir)  # Get the relative path
                new_dst_dir = os.path.join(dst_dir, relative_path,
                                           os.path.splitext(file)[0])
                os.makedirs(new_dst_dir, exist_ok=True)

                # Call generate_images with the new directory
                generate_images(api,
                                configs,
                                prompts,
                                n_images,
                                new_dst_dir,
                                lora_name=lora_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate images based on prompts.')
    parser.add_argument('--dst_dir',
                        type=str,
                        help='Destination directory to save generated images.')
    parser.add_argument('--path_prepend',
                        type=str,
                        default=None,
                        help='Description to prepend folder name.')
    parser.add_argument('--n_images',
                        type=int,
                        help='Number of images to generate per prompt file.')
    parser.add_argument('--eval_prompt_dir',
                        type=str,
                        help='Directory containing prompt files.')
    parser.add_argument('--sampling_configs',
                        type=str,
                        help='TOML configuration file path.')
    parser.add_argument('--network_csv',
                        type=str,
                        help='CSV file with base model and lora details.')

    args = parser.parse_args()

    # Load configs from TOML file
    with open(args.sampling_configs, 'r') as file:
        configs = toml.load(file)

    api = webuiapi.WebUIApi()
    base_model = ''

    # Read the CSV file and iterate through each line
    with open(args.network_csv, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            base_model_new, lora = row

            if base_model_new != base_model:
                base_model = base_model_new
                # Set the model using api
                api.util_set_model(base_model)

            # Create a new dst_dir with base_model name and lora_name
            base_model_name = os.path.basename(base_model).split('.')[0]

            if args.path_prepend is not None:
                base_model_name = f"{args.path_prepend}-{base_model_name}"

            if lora:  # if lora is not an empty string
                new_dst_dir = os.path.join(args.dst_dir,
                                           f"{base_model_name}-{lora}")
            else:
                new_dst_dir = os.path.join(args.dst_dir, base_model_name)

            os.makedirs(new_dst_dir, exist_ok=True)
            save_config_file = os.path.join(new_dst_dir,
                                            'sampling_config.toml')
            with open(save_config_file, 'w') as toml_file:
                toml.dump(configs, toml_file)

            if lora:  # if lora is not an empty string
                generate_images_from_dir(api,
                                         configs,
                                         args.n_images,
                                         args.eval_prompt_dir,
                                         new_dst_dir,
                                         lora_name=lora)
            else:
                generate_images_from_dir(api, configs, args.n_images,
                                         args.eval_prompt_dir, new_dst_dir)
