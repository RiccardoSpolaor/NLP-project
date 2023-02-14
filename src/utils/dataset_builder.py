import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def _get_labels_list(row: pd.Series) -> pd.Series:
    return [index for index, value in row.items() if value == 1]

def get_dataframes(data_folder: str, df_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert df_type in ['training', 'validation', 'test'], 'Specify the `df_type` as either training, validation or test'
    
    arguments_df = pd.read_csv(os.path.join(data_folder, f'arguments-{df_type}.tsv'), sep='\t', header=0)
    labels_df = pd.read_csv(os.path.join(data_folder, f'labels-{df_type}.tsv'), sep='\t', header=0)
    
    arguments_df['Labels'] = labels_df.apply(lambda x: _get_labels_list(x), axis=1)
    
    arguments_df.drop('Argument ID', axis=1, inplace=True)
    labels_df.drop('Argument ID', axis=1, inplace=True)
    
    return arguments_df, labels_df

def split_dataframes(arguments_df: pd.DataFrame, labels_df: pd.DataFrame, seed: int = 42, 
                     test_size: int = .2) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
    arguments_df_1, arguments_df_2, labels_df_1, labels_df_2 = train_test_split(
        arguments_df, labels_df, test_size=test_size, random_state=seed)

    arguments_df_1.reset_index(drop=True, inplace=True)
    labels_df_1.reset_index(drop=True, inplace=True)
    arguments_df_2.reset_index(drop=True, inplace=True)
    labels_df_2.reset_index(drop=True, inplace=True)
    
    return (arguments_df_1, labels_df_1), (arguments_df_2, labels_df_2)
