"""Module providing functions to build the datasets."""
import os
from typing import Literal, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd


def _get_labels_list(labels_row: pd.Series) -> pd.Series:
    """Get a list of labels.

    Parameters
    ----------
    labels_row : Series
        the current row of labels.

    Returns
    -------
    Series
        The list of active labels in the row.
    """
    return [index for index, value in labels_row.items() if value == 1]

def get_dataframes(
    data_folder: str, df_type: Literal['training', 'validation']
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the arguments and labels dataframes.

    Parameters
    ----------
    data_folder : str
        The data folder path.
    df_type : 'training' | 'validation'
        The dataframe type.

    Returns
    -------
    (DataFrame, DataFrame)
        The arguments and labels dataframes.
    """
    # Assert the dataframe type is correct
    assert df_type in ['training', 'validation'], \
        'Specify the `df_type` as either training or validation'

    # Get the arguments and labels dataframes
    arguments_df = pd.read_csv(
        os.path.join(data_folder,f'arguments-{df_type}.tsv'),
        sep='\t', header=0)
    labels_df = pd.read_csv(
        os.path.join(data_folder, f'labels-{df_type}.tsv'),
        sep='\t', header=0)

    # Add list of labels to the arguments dataframe
    arguments_df['Labels'] = labels_df.apply(_get_labels_list, axis=1)

    # Remove the `Argument ID` column from the dataframes
    arguments_df.drop('Argument ID', axis=1, inplace=True)
    labels_df.drop('Argument ID', axis=1, inplace=True)

    return arguments_df, labels_df

def split_dataframes(
    arguments_df: pd.DataFrame, labels_df: pd.DataFrame, seed: int = 42,
    test_size: int = .2) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame],
                                  Tuple[pd.DataFrame, pd.DataFrame]]:
    """Split the arguments and label dataframes into train and validation

    Parameters
    ----------
    arguments_df : pd.DataFrame
        The arguments pandas dataframe to split
    labels_df : pd.DataFrame
        The labels pandas dataframe to split
    seed : int, optional
        The seed to use, by default 42
    test_size : int, optional
        The test size, by default .2

    Returns
    -------
    ((DataFrame, DataFrame), (DataFrame, DataFrame))
        A tuple containing:
        - The split arguments train dataframe and the respective labels
        dataframe;
        - The split arguments validation dataframe and the respective
        labels dataframe.
    """
    # Split the dataframes in train and validation
    arguments_df_1, arguments_df_2, labels_df_1, labels_df_2 = \
        train_test_split(arguments_df, labels_df, test_size=test_size,
                         random_state=seed)

    # Reset the indices
    arguments_df_1.reset_index(drop=True, inplace=True)
    labels_df_1.reset_index(drop=True, inplace=True)
    arguments_df_2.reset_index(drop=True, inplace=True)
    labels_df_2.reset_index(drop=True, inplace=True)

    return (arguments_df_1, labels_df_1), (arguments_df_2, labels_df_2)
