import os
import argparse
import toml
import csv


def parse_toml(toml_file):
    data = toml.load(toml_file)

    config_name = os.path.splitext(os.path.basename(toml_file))[0]

    if "nt" in config_name:
        algo = "db"
        preset = "full"
        factor = "N/A"
    else:
        lycoris_data = data.get('LyCORIS', {})
        network_args = lycoris_data.get('network_args', [])
        algo = next((item.split('=')[1]
                     for item in network_args if "algo=" in item), "N/A")
        preset = next((item.split('=')[1]
                      for item in network_args if "preset=" in item), "N/A")
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

    return [config_name, algo, preset, lr, dim, alpha, factor]


def main(directory, csv_name):
    with open(csv_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["config_name", "algo", "preset", "lr", "dim", "alpha", "factor"])

        for filename in os.listdir(directory):
            if filename.endswith('.toml'):
                try:
                    row_data = parse_toml(os.path.join(directory, filename))
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
