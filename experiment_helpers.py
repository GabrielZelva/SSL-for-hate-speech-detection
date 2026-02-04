import pandas as pd
import numpy as np

def mask_labels(dataframe, column = None, mask_probability = 0.8):

    # Unlses otherwise stated, the labels are expected to be in the last column
    if column is None:
        column = len(dataframe.columns) -1 

     # Get column name for later use
    col_name = dataframe.columns[column]

    # Find out what the individual labels are
    unique_values = dataframe.iloc[:, column].unique()
   
    # Separate the labels
    filtered = {}

    for unique_value in unique_values:

        filtered[unique_value] = dataframe[dataframe.iloc[:, column] == unique_value].copy()
    
    # Mask a percentage of each
    for key, df in filtered.items():

        mask_idx = df.sample(frac=mask_probability).index

        df.loc[mask_idx, col_name] = np.nan
        
        filtered[key] = df

    # Join the tables back together and shuffle
    dataframe = pd.concat(filtered.values(), ignore_index=True)
    
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    return dataframe
