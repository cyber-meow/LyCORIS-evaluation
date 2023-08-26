import pandas as pd
import argparse

def separate_csv(input_csv, output_csv1, output_csv2):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv)
    
    # Separate the DataFrame into two based on the "Folder" column
    df1 = df[df['Folder'].str.contains('/', na=False)]
    df2 = df[~df['Folder'].str.contains('/', na=False)]

    # Drop the last three columns for df1, keep only "Folder" and the last three columns for df2
    df1 = df1.iloc[:, :-3]
    df2 = df2[["Folder"] + list(df.columns[-3:])]

    # Save the new DataFrames to CSV files
    df1.to_csv(output_csv1, index=False)
    df2.to_csv(output_csv2, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Separate a CSV file into two.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_csv1', type=str, required=True, help='Path to save the first output CSV file.')
    parser.add_argument('--output_csv2', type=str, required=True, help='Path to save the second output CSV file.')
    args = parser.parse_args()

    separate_csv(args.input_csv, args.output_csv1, args.output_csv2)

