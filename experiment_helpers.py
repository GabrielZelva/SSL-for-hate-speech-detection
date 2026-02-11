import pandas as pd
import numpy as np
import torch


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


def promotion_mechanism(
    unlabeled_data, labeled_data, probability_matrix, threshold=0.9
):
    """
    This function takes the labeled & unlabeled data and the probabilities of the labels.
    Then, it tests if any of the probabilities have surpassed the threshold and if so,
    assigns the corresponding label to the predictors which produced it.
    Finally, it transfers the newly labeled data to the labeled_data dataframe.
    It could be summed up as 'transfer data from unlabeled to labeled
    if a pseudolabel can be assigned'
    """

    new_labels = []

    for row in range(len(probability_matrix)):
        was_there_a_promotion = False

        for column in [0, 1, 2]:
            value = probability_matrix[row, column]
            if value > threshold:
                new_labels.append(column)  # Culumn id = label
                was_there_a_promotion = True

        if not was_there_a_promotion:
            new_labels.append(np.nan)


    if not all(np.isnan(new_labels)): # If there were no promotions, don't change anything

        unlabeled_data.iloc[:, -1] = new_labels

        unlabeled_data, newly_labeled = extract_equal_proportion(
            unlabeled_data, proportion=1
        )

        labeled_data = pd.concat([labeled_data, newly_labeled], ignore_index=True)

        labeled_data = labeled_data.sample(frac=1).reset_index(drop=True)

    return unlabeled_data, labeled_data
