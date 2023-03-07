import numpy as np
from sklearn.metrics import precision_recall_curve
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from typing import Optional, Tuple, Union

def get_dataset_predictions(model: AutoModelForSequenceClassification, dataloader: DataLoader,
                            device: str) -> Tuple[np.ndarray, np.ndarray]:
    predictions = []
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
            predictions.extend(preds)

            # Add true targests to the full list
            targets = data['labels'].cpu().numpy()
            true_labels.extend(targets)

    return np.array(predictions), np.array(true_labels).astype(np.uint8)

def predict(model: AutoModelForSequenceClassification, dataloader: DataLoader, device: str,
            threshold: Optional[Union[float, np.ndarray[float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    preds, y_true = get_dataset_predictions(model, dataloader, device)

    if threshold is None:
        threshold = 0.

    preds = preds > threshold

    preds = preds.astype(np.uint8)

    return preds, y_true

def get_best_thresholds(y_true, y_preds):
    precision, recall, thresholds, f1_scores, idx_max, best_thresholds = dict(), dict(), dict(), dict(), dict(), dict()

    for i in range(y_true.shape[1]):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_true[:,i], y_preds[:,i])
        f1_scores[i] = np.array([2 * (p * r) / (p + r) if p + r != 0 else 0. for p, r in zip(precision[i], recall[i])])
        idx_max[i] = f1_scores[i].argmax()
        best_thresholds[i] = thresholds[i][idx_max[i]]
    return np.array(list(best_thresholds.values()))