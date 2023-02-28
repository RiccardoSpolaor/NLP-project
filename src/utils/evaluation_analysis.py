from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, classification_report, ConfusionMatrixDisplay, multilabel_confusion_matrix, 
                             roc_curve, RocCurveDisplay)

from typing import Dict, List, OrderedDict

def get_classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str]) -> Dict:
    return classification_report(y_true=y_true, y_pred=y_pred, target_names=targets,
                                 zero_division=0, output_dict=True)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str],
                                dataset_name) -> None:


    #class_report_val = classification_report(y_true=y_true, y_pred=y_pred, target_names=targets,
    #                                        zero_division=0, output_dict=True)

    print(f'Classification report for the {dataset_name} set:')
    print(classification_report(y_true=y_true, y_pred=y_pred,
                                target_names=targets, zero_division=0))
    
def plot_targets_classification_statistics(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str], dataset_name: str) -> None:
    classification_report_dict = get_classification_report_dict(y_true, y_pred, targets)
    
    class_report_sorted = sorted([(classification_report_dict[label]['f1-score'], label) 
                                  for label in classification_report_dict.keys() if label in targets ])

    class_report_sorted = OrderedDict({label: classification_report_dict[label] for (_, label) in class_report_sorted})

    plt.figure(figsize=(12,6))
    x_axis = 2*np.arange(len(class_report_sorted))
    plt.bar(x_axis-0.5, [class_report_sorted[tag]['precision'] for tag in class_report_sorted], label='precision', width=0.5)
    b = plt.bar(x_axis, [class_report_sorted[tag]['recall'] for tag in class_report_sorted], label='recall', width=0.5)
    plt.bar(x_axis+0.5, [class_report_sorted[tag]['f1-score'] for tag in class_report_sorted], label='f1-score', width=0.5)
    plt.bar_label(b, labels=[class_report_sorted[tag]['support'] for tag in class_report_sorted], label_type='center')
    plt.xticks(x_axis, list(class_report_sorted.keys()))
    plt.grid(axis='y')
    plt.legend()
    plt.xticks(rotation=90)
    plt.title(f'Target classification statistics on the {dataset_name} set')
    plt.xlabel('labels')
    plt.show()

def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str], dataset_name: str,
                            normalize_by: str = 'none') -> None:
    assert normalize_by in ['recall', 'precision', 'accuracy', 'none'], \
        'Select one normalization metric among: recall, precision, accuracy or none'
    
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred).astype(np.float32)
    
    confusion_matrices_ = deepcopy(confusion_matrices)

    
    for i in range(confusion_matrices.shape[0]):
        if normalize_by == 'accuracy':
            confusion_matrices[i] = confusion_matrices_[i] / confusion_matrices_[i].sum()
        elif normalize_by == 'precision':
            confusion_matrices[i] = confusion_matrices_[i] / confusion_matrices_[i].sum(axis=0)
        elif normalize_by == 'recall':
            confusion_matrices[i] = confusion_matrices_[i] / confusion_matrices_[i].sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(5, 4, figsize=(15, 15))

    for i, (matrix, label) in enumerate(zip(confusion_matrices, targets)):
        ax = plt.subplot(5, 4, i + 1)
        cf = ConfusionMatrixDisplay(matrix, display_labels=['Current', 'Others'])
        cf.plot(cmap=plt.cm.Blues, values_format=".2f", ax=ax, colorbar=False)
        ax.set_title(label)
        
    fig.tight_layout()

    plt.suptitle(f'Confusion matrices of the {dataset_name} set for each target class normalized by {normalize_by}', y=1.025)

    plt.show()

def _plot_roc_subplot(y_true: np.ndarray, pred_scores: np.ndarray, targets: List[str], index: int) -> None:
    ax = plt.subplot(2, 2, index)

    for class_id in range(index * 4, index * 4 + 4):
        RocCurveDisplay.from_predictions(
            y_true[:, class_id],
            pred_scores[:, class_id],
            name=f"{targets[class_id]}",
            ax=ax
        )

def plot_roc_curves(y_true: np.ndarray, pred_scores: np.ndarray, targets: List[str], dataset_name: str) -> None:
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(len(targets)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], pred_scores[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(len(targets)):
        # Linear interpolation
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= len(targets)

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")
    
    
    plt.figure(figsize=(15,10))
    plt.subplot(2, 2, 1)
    plt.suptitle(f'Receiver Operating Characteristic One-vs-Rest curves on the {dataset_name} set')

    for i in range(4):
        _plot_roc_subplot(y_true, pred_scores, targets, i + 1)
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average (AUC = {roc_auc['macro']:.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )
        plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.axis("square")
    plt.tight_layout()
    plt.show()