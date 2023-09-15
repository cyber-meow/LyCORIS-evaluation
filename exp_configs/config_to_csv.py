import os
import argparse
import toml
import csv


def parse_toml(toml_file):
    data = toml.load(toml_file)

    config_name = os.path.splitext(os.path.basename(toml_file))[0]
    parts = config_name.split('-')
    if parts[-1] in ['a', 'b', 'c']:
        config_name = '-'.join(parts[:-1])

    lycoris_data = data.get('LyCORIS', {})
    network_args = lycoris_data.get('network_args', [])
    algo = next((item.split('=')[1]
                 for item in network_args if "algo=" in item), "lora")
    preset = next((item.split('=')[1]
                  for item in network_args if "preset=" in item), "full")
    factor = next((item.split('=')[1]
                  for item in network_args if "factor=" in item), "N/A")
    if factor == '-1':
        factor = 108

    optimizer_data = data.get('Optimizer', {})
    lr = optimizer_data.get(
        'unet_lr', optimizer_data.get('learning_rate', 'N/A'))

    network_setup = data.get('Network_setup', {})
    dim = network_setup.get('network_dim', 'N/A')
    alpha = network_setup.get('network_alpha', 'N/A')

    caption_setup = data.get('Captions', {})
    caption_ext = caption_setup.get('caption_extension', '.txt')

    return [config_name, algo, preset, lr, dim, alpha, factor, caption_ext]


def main(directory, csv_name):
    with open(csv_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Config", "Algo", "Preset", "Lr", "Dim",
             "Alpha", "Factor", "Caption"])
        seen_configs = set()

        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.toml'):
                    try:
                        row_data = parse_toml(os.path.join(root, filename))
                        if row_data[0] not in seen_configs:
                            seen_configs.add(row_data[0])
                            csv_writer.writerow(row_data)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parse TOML files and save data to CSV")
    parser.add_argument('--directory', type=str,
                        help="Directory containing the TOML files")
    parser.add_argument('--output_csv', type=str,
                        help="Csv file to save config correspondances")
    args = parser.parse_args()

    main(args.directory, args.output_csv)
