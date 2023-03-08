"""Module providing functions to perform analysis on the model evaluation."""
from copy import deepcopy
from typing import Dict, List, Literal, OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc, classification_report, ConfusionMatrixDisplay,
    multilabel_confusion_matrix, precision_recall_curve,
    average_precision_score, RocCurveDisplay, PrecisionRecallDisplay)
import numpy as np


def _get_classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray,
                                   targets: List[str]) -> Dict[str, float]:
    """Get the classification report dictionary regarding the predictions
    of the model.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    targets : list of str
        The target labels.

    Returns
    -------
    { str: float }
        The classification report dictionary.
    """
    return classification_report(
        y_true=y_true, y_pred=y_pred, target_names=targets, zero_division=0,
        output_dict=True)

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                targets: List[str], dataset_name: str) -> None:
    """Print the classification report regarding the predictions of the
    model.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    targets : list of str
        The target labels.
    dataset_name : str
        The name of the dataset used to compute the classification report.
    """
    print(f'Classification report for the {dataset_name} set:')
    print(classification_report(y_true=y_true, y_pred=y_pred,
                                target_names=targets, zero_division=0))
    
def plot_targets_classification_statistics(
    y_true: np.ndarray, y_pred: np.ndarray, targets: List[str],
    dataset_name: str) -> None:
    """Plot the targets classification statistics in term of recall,
    precision and F1 macro score.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    targets : list of str
        The target labels.
    dataset_name : str
        The name of the dataset used to compute the statistics.
    """
    classification_report_dict = _get_classification_report_dict(y_true, y_pred, targets)

    class_report_sorted = sorted([(classification_report_dict[label]['f1-score'], label) 
                                  for label in classification_report_dict.keys() if label in targets ])

    class_report_sorted = OrderedDict({label: classification_report_dict[label] for (_, label) in class_report_sorted})

    plt.figure(figsize=(12,6))
    x_axis = 2*np.arange(len(class_report_sorted))
    plt.bar(x_axis-0.5, [class_report_sorted[tag]['precision']
                         for tag in class_report_sorted],
            label='precision', width=0.5)
    b = plt.bar(x_axis, [class_report_sorted[tag]['recall']
                         for tag in class_report_sorted],
                label='recall', width=0.5)
    plt.bar(x_axis + 0.5, [class_report_sorted[tag]['f1-score']
                           for tag in class_report_sorted],
            label='f1-score', width=0.5)
    plt.bar_label(b, labels=[class_report_sorted[tag]['support']
                             for tag in class_report_sorted],
                  label_type='center')
    plt.xticks(x_axis, list(class_report_sorted.keys()))
    plt.grid(axis='y')
    plt.legend()
    plt.xticks(rotation=90)
    plt.title(f'Target classification statistics on the {dataset_name} set')
    plt.xlabel('labels')
    plt.show()

def plot_confusion_matrices(
    y_true: np.ndarray, y_pred: np.ndarray, targets: List[str],
    dataset_name: str, normalize_by:
        Literal['recall', 'precision', 'accuracy'] = 'recall') -> None:
    """Plot the One-vs-Rest confusion matrices of the predictions.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    targets : list of str
        The target labels.
    dataset_name : str
        The name of the dataset used to compute the confusion matrices.
    normalize_by : 'recall' | 'precision' | 'accuracy', optional
        The way in which the matrices are normalized, by default 'recall'.
    """
    assert normalize_by in ['recall', 'precision', 'accuracy'], \
        'Select one normalization metric among: recall, precision or accuracy.'
    
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    confusion_matrices = confusion_matrices.astype(np.float32)
    c_ms = deepcopy(confusion_matrices)

    for i in range(confusion_matrices.shape[0]):
        if normalize_by == 'accuracy':
            confusion_matrices[i] = c_ms[i] / c_ms[i].sum()
        elif normalize_by == 'precision':
            confusion_matrices[i] = c_ms[i] / c_ms[i].sum(axis=0)
        elif normalize_by == 'recall':
            confusion_matrices[i] = c_ms[i] / c_ms[i].sum(axis=1,
                                                          keepdims=True)

    fig, ax = plt.subplots(5, 4, figsize=(15, 15))

    for i, (matrix, label) in enumerate(zip(confusion_matrices, targets)):
        ax = plt.subplot(5, 4, i + 1)
        cf = ConfusionMatrixDisplay(matrix,
                                    display_labels=['Current', 'Others'])
        cf.plot(cmap=plt.cm.Blues, values_format=".2f", ax=ax, colorbar=False)
        ax.set_title(label)
        
    fig.tight_layout()

    plt.suptitle(f'Confusion matrices of the {dataset_name} set for each ' +
                 f'target class normalized by {normalize_by}', y=1.025)

    plt.show()

def get_precision_recall_statistics(y_true: np.ndarray, pred_scores: np.ndarray, targets: List[str]):
    precision, recall, average_precision = dict(), dict(), dict()
    for i in range(len(targets)):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], pred_scores[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], pred_scores[:, i])

    precision["macro"], recall["macro"], _ = precision_recall_curve(
        y_true.ravel(), pred_scores.ravel()
    )
    average_precision["macro"] = average_precision_score(y_true, pred_scores, average="macro")
    return precision, recall, average_precision

def plot_precision_recall_curves(precision, recall, average_precision, targets: List[str], dataset_name: str) -> None:
    plt.figure(figsize=(15,10))
    plt.subplot(2, 2, 1)
    plt.suptitle(f'Precision-Recall One-vs-Rest curves on the {dataset_name} set')

    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        for class_id in range(i * 4, i * 4 + 4):
            display = PrecisionRecallDisplay(
                recall=recall[class_id],
                precision=precision[class_id],
                average_precision=average_precision[class_id]
            )
            display.plot(ax=ax, name=f"{targets[class_id]}")
        display = PrecisionRecallDisplay(
            recall=recall["macro"],
            precision=precision["macro"],
            average_precision=average_precision["macro"]
        )
        display.plot(ax=ax, name="Macro-average precision-recall", color="navy", linestyle=":", linewidth=4)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        #ax.set_xticks(np.arange(0., 1.1, .2))
        #ax.set_yticks(np.arange(0., 1.1, .2))
        #display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)
        #plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")

        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.legend(loc='upper right')
        #plt.axis("square")
    plt.tight_layout()
    plt.show()
