import pandas as pd
import numpy as np
import torch

def evaluate_model(model, data, predictions, ground_truth):
    """This function evaluates the model on a test split"""

    predictions = model.predict(data, return_predictions=True)

    mean = (predictions == ground_truth).float().mean()
    recalls = {}

    for c in torch.unique(ground_truth):
        tp = ((predictions == c) & (ground_truth == c)).sum().float()
        fn = ((predictions != c) & (ground_truth == c)).sum().float()
        recalls[int(c)] = tp / (tp + fn)

    return mean, recalls[0], recalls[1], recalls[2]

def mask_labels(dataframe, column=None, mask_probability=0.8):
    """This function turns a certain percentage of each label into NaNs"""

    # Unlses otherwise stated, the labels are expected to be in the last column
    if column is None:
        column = len(dataframe.columns) - 1

    # Get column name for later use
    col_name = dataframe.columns[column]

    # Find out what the individual labels are
    unique_values = dataframe.iloc[:, column].unique()

    # Separate the labels
    filtered = {}

    for unique_value in unique_values:
        filtered[unique_value] = dataframe[
            dataframe.iloc[:, column] == unique_value
        ].copy()

    # Mask a percentage of each
    for key, df in filtered.items():
        mask_idx = df.sample(frac=mask_probability).index

        df.loc[mask_idx, col_name] = np.nan

        filtered[key] = df

    # Join the tables back together and shuffle
    dataframe = pd.concat(filtered.values(), ignore_index=True)

    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    return dataframe


def extract_equal_proportion(dataframe, proportion, column=None):
    """
    This function extracts a certain proportion of datapoints into a new dataframe.
    It works per label to guarantee representation.
    """

    # Unlses otherwise stated, the labels are expected to be in the last column
    if column is None:
        column = len(dataframe.columns) - 1

    # Find out what the individual labels are
    unique_values = dataframe.iloc[:, column].unique()

    # Separate the labels
    filtered_old = {}

    # > np.nan == np.nan
    # > FALSE
    # hence, this if statement
    for unique_value in unique_values:
        if pd.isna(unique_value):
            filtered_old[unique_value] = dataframe[
                dataframe.iloc[:, column].isna()
            ].copy()
        else:
            filtered_old[unique_value] = dataframe[
                dataframe.iloc[:, column] == unique_value
            ].copy()

    # Extract the proportion
    filtered_new = {}

    for key, df in filtered_old.items():
        if pd.isna(key):
            continue
        extracted = df.sample(frac=proportion)
        filtered_new[key] = extracted
        filtered_old[key] = df.drop(extracted.index)

    # Join the tables back together and shuffle
    dataframe_old = pd.concat(filtered_old.values(), ignore_index=True)
    dataframe_old = dataframe_old.sample(frac=1).reset_index(drop=True)

    new_split = pd.concat(filtered_new.values(), ignore_index=True)
    new_split = new_split.sample(frac=1).reset_index(drop=True)

    return dataframe_old, new_split


