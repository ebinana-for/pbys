"""This script splits a CSV file containing labels into training and testing sets.
It reads the input CSV file, splits the data into training and testing sets, and saves them as new CSV files."""
import os

import pandas as pd
from fire import Fire
from sklearn.model_selection import train_test_split

LABEL_FILE = "/data/public_datasets/Lagenda/lagenda_annotation_sanity_check1.csv"

def run(label_file=LABEL_FILE):
    """
    Splits the input CSV file into training and testing sets and saves them as new CSV files.
    Args:
        label_file (str): Path to the input CSV file containing labels.
    Returns:
        None
    This function performs the following steps:
    1. Reads the input CSV file into a DataFrame.
    2. Drops the 'Unnamed: 0' column from the DataFrame.
    3. Splits the DataFrame into training and testing sets (90% training, 10% testing).
    4. Resets the index of both DataFrames.
    5. Saves the training set as a new CSV file with '_train' suffix.
    6. Saves the testing set as a new CSV file with '_test' suffix.
    """
    label_df = pd.read_csv(label_file)
    label_df.drop('Unnamed: 0', axis=1, inplace=True)
    basename = os.path.basename(label_file)
    dirname =  os.path.dirname(label_file)

    basename_train = basename.replace('annotation', 'annotation_train')
    basename_test = basename.replace('annotation', 'annotation_test')
    train_df, test_df = train_test_split(label_df, test_size=0.1)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(os.path.join(dirname, basename_train))
    test_df.to_csv(os.path.join(dirname, basename_test))
    print("File {} created successfully".format(basename_train))
    print("File {} created successfully".format(basename_test))

if __name__ == "__main__":
    Fire(run)
