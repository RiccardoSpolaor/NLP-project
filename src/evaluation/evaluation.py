"""Module defining functions to evaluate the model predictions."""
from typing import Tuple, Union
import numpy as np
from sklearn.metrics import precision_recall_curve
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


def get_dataset_prediction_scores(
    model: AutoModelForSequenceClassification, dataloader: DataLoader,
    device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get the prediction scores of the model from a dataloader.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        The model that is used to predict the scores.
    dataloader : DataLoader
        The dataloader loading the dataset to predict.
    device : str
        The device onto which attach the prediction process.

    Returns
    -------
    (ndarray, ndarray)
        Tuple containing the prediction scores and the true labels
        of the dataset.
    """
    scores = []
    true_labels = []

    for _, data in enumerate(dataloader, 0):
        with torch.no_grad():
            # Get the data
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)

            # Add batch predictions to the full list
            outputs = model(ids, mask)
            preds = outputs.logits
            preds = preds.cpu().numpy()
            scores.extend(preds)

            # Add true targests to the full list
            targets = data['labels'].cpu().numpy()
            true_labels.extend(targets)

    return np.array(scores), np.array(true_labels).astype(np.uint8)

def get_dataset_predictions(
    model: AutoModelForSequenceClassification, dataloader: DataLoader,
    device: str, thresholds: Union[float, np.ndarray[float]] = 0.
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Get the prediction labels of the model from a dataloader.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        The model that is used to predict the labels.
    dataloader : DataLoader
        The dataloader loading the dataset to predict.
    device : str
        The device onto which attach the prediction process.
    thresholds : float | ndarray of float
        The thresholds used for the predictions, by default 0.

    Returns
    -------
    (ndarray, ndarray)
        Tuple containing the predicted labels and the true labels
        of the dataset.
    """
    # Get the prediction scores.
    y_scores, y_true = get_dataset_prediction_scores(model, dataloader, device)

    # Get the predictions.
    y_preds = y_scores > thresholds
    y_preds = y_preds.astype(np.uint8)

    return y_preds, y_true

def get_best_thresholds(y_true: np.ndarray, y_preds: np.ndarray) -> np.ndarray:
    """Get the best thresholds for each label that maximize the F1 macro
    Score.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_preds : ndarray
        The prediction scores.

    Returns
    -------
    ndarray
        The best thresholds for each label.
    """
    precision, recall, thresholds = {}, {}, {}
    f1_scores, idx_max, best_thresholds = {}, {}, {}

    for i in range(y_true.shape[1]):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(
            y_true[:,i], y_preds[:,i])
        f1_scores[i] = np.array([2 * (p * r) / (p + r) if p + r != 0 else 0.
                                 for p, r in zip(precision[i], recall[i])])
        idx_max[i] = f1_scores[i].argmax()
        best_thresholds[i] = thresholds[i][idx_max[i]]

    return np.array(list(best_thresholds.values()))
