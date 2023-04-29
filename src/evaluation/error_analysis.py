"""Module providing functions to perform error analysis."""
from copy import deepcopy
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _get_accuracy(y_true: np.ndarray, y_preds: np.ndarray) -> np.ndarray:
    """Get the accuracy of each predicted instance.

    Parameters
    ----------
    y_true : ndarray
        The target labels.
    y_preds : ndarray
        The predicted labels.

    Returns
    -------
    float
        The accuracy of each predicted instance.
    """
    return (y_preds == y_true).astype(np.float32).mean(axis=1)

def get_k_worst_predicted_instances(
        arguments_test_df: pd.DataFrame, labels_test_df: pd.DataFrame,
        y_true: np.ndarray, y_preds: np.ndarray,
        n_worst_instances: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the k worst predicted instances based on accuracy.

    Parameters
    ----------
    arguments_test_df : DataFrame
        The pandas dataframe containing the test set arguments.
    labels_test_df : DataFrame
        The pandas dataframe containing the test set labels.
    y_true : ndarray
        The target labels.
    y_preds : ndarray
        The predicted labels.
    n_worst_instances : int, optional
        The k worst instances to return, by default 50.

    Returns
    -------
    DataFrame
        The arguments dataframe of the k worst instances.
    DataFrame
        The labels dataframe of the k worst instances.
    """
    accuracies = _get_accuracy(y_true, y_preds)
    argsort_acc = np.argsort(accuracies)

    worst_predicted_instances = arguments_test_df.iloc[
        argsort_acc[:n_worst_instances]]
    worst_predicted_instances_targets = labels_test_df.iloc[
        argsort_acc[:n_worst_instances]]

    return worst_predicted_instances, worst_predicted_instances_targets

def print_k_worst_predicted_instances(
    worst_predicted_instances: pd.DataFrame, y_true: np.ndarray,
    y_pred: np.ndarray, targets: List[str],
    n_worst_instances: int = 5) -> None:
    """Print the worst k predicted instances.

    Parameters
    ----------
    worst_predicted_instances : DataFrame
        The pandas dataframe containing the worst predicted instances.
    y_true : ndarray
        The target labels.
    y_preds : ndarray
        The predicted labels.
    targets : list of str
        The sentiment target names.
    n_worst_instances : int, optional
        The k worst instances to print, by default 5.
    """
    accuracies = _get_accuracy(y_true, y_pred)
    argsort_acc = np.argsort(accuracies)

    for i in range(n_worst_instances):
        row = worst_predicted_instances.iloc[i]
        sorted_y_pred = y_pred[argsort_acc][i]
        print(f'Worst instance {i + 1}:')
        print('-----------------')

        print(f'Premise: "{row.Premise}"')
        print(f'Stance: "{row.Stance}"')
        print(f'Conclusion: "{row.Conclusion}"')

        print(f'True targets: "{"; ".join(row.Labels)}"')

        predicted_targets_str = '; '.join(
            [t for idx, t in zip(sorted_y_pred, targets) if idx == 1])

        print(f'Predicted targets: "{predicted_targets_str}"', end='\n\n')
