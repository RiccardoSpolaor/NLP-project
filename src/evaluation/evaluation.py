"""Module defining functions to evaluate the model predictions."""
from typing import Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_dataset_prediction_scores(
    model: nn.Module, dataloader: DataLoader, device: str
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Get the prediction scores of the model from a dataloader.

    Parameters
    ----------
    model : Module
        The model that is used to predict the scores.
    dataloader : DataLoader
        The dataloader loading the dataset to predict.
    device : str
        The device onto which attach the prediction process.

    Returns
    -------
    ndarray
        The prediction scores on the dataset instances.
    ndarray    
        The true labels of the dataset instances.
    """
    scores = []
    true_labels = []

    for _, data in enumerate(dataloader, 0):
        with torch.no_grad():
            # Get the data
            ids = data['ids'].to(device)

            # Add batch predictions to the full list
            if 'mask' in data.keys():
                mask = data['mask'].to(device)
                outputs = model(ids, mask)
            else:
                outputs = model(ids)

            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            outputs = outputs.cpu().numpy()
            scores.extend(outputs)

            # Add true targests to the full list
            targets = data['labels'].cpu().numpy()
            true_labels.extend(targets)

    return np.array(scores), np.array(true_labels).astype(np.uint8)

def get_dataset_predictions(
    model: nn.Module, dataloader: DataLoader, device: str,
    thresholds: Union[float, np.ndarray[float]] = 0.
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Get the prediction labels of the model from a dataloader.

    Parameters
    ----------
    model : Module
        The model that is used to predict the labels.
    dataloader : DataLoader
        The dataloader loading the dataset to predict.
    device : str
        The device onto which attach the prediction process.
    thresholds : float | ndarray of float
        The thresholds used for the predictions, by default 0.

    Returns
    -------
    ndarray
        The predicted labels of the dataset instances.
    ndarray    
        The true labels of the dataset instances.
    """
    # Get the prediction scores.
    y_scores, y_true = get_dataset_prediction_scores(model, dataloader, device)

    # Get the predictions.
    y_preds = y_scores > thresholds
    y_preds = y_preds.astype(np.uint8)

    return y_preds, y_true

def get_cumulative_precision_recall_and_f1(
    y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the cumulative precision, recall and F1 macro score.
    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_scores : ndarray
        The prediction scores.

    Returns
    -------
    ndarray
        The sorted input logits.
    ndarray
        The cumulative precision.
    ndarray
        The cumulative recall.
    ndarray
        The cumulative F1 macro scores.
    """
    y_scores = torch.tensor(y_scores)
    y_true = torch.tensor(y_true)

    # Argsort elements by non decreasing values of each column.
    idx = torch.argsort(y_scores, 0)

    # Sort predictions and true labels based on argsort.
    s_scores = torch.gather(y_scores, 0, idx)
    s_true = torch.gather(y_true, 0, idx)

    # Cumulative sum of total positive instances.
    c_positive = torch.cumsum(s_true, 0)

    # Cumulative sum of true positive elements for each target.
    c_true_positive = c_positive[-1:] - c_positive

    # Vector of instances range.
    r = (torch.arange(len(c_true_positive)) + 1)[:, None]

    # Constant to handle zero division.
    CONST = 1e-7

    # Compute the cumulative precision.
    c_precision = c_true_positive / (r[-1] - r + 1) + CONST
    # Compute the cumulative recall.
    c_t_positives_f_negatives = (c_true_positive[:1] + CONST) + CONST
    c_recall = c_true_positive / c_t_positives_f_negatives
    # Compute the cumulative F1 macro score.
    c_f1 = 2 * (c_precision * c_recall) / (c_precision + c_recall)
    return s_scores, c_precision, c_recall, c_f1

def get_best_thresholds(
    y_true: np.ndarray, y_scores: np.ndarray) -> np.ndarray:
    """Get the best thresholds for each label that maximize the F1 macro
    score.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_scores : ndarray
        The prediction scores.

    Returns
    -------
    ndarray
        The best thresholds for each label.
    """
    s_scores, _, _, c_f1 = get_cumulative_precision_recall_and_f1(
        y_true, y_scores)

    # Get the index of maximum F1 macro score for each target.
    idx_max = c_f1.argmax(0)
    # Get the best threshold for each target.
    return np.array([s_scores[idx, i] for i, idx in enumerate(idx_max)])
