import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from typing import Tuple

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
            threshold: int = None) -> Tuple[np.ndarray, np.ndarray]:
    preds, y_true = get_dataset_predictions(model, dataloader, device)

    if threshold is None:
        threshold = 0

    preds = preds > threshold

    preds = preds.astype(np.uint8)

    return preds, y_true
