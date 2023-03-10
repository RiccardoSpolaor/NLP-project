"""Module providing functions to perform analysis on the model evaluation."""
from copy import deepcopy
from typing import Dict, List, Literal, OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay, multilabel_confusion_matrix,
    roc_curve, RocCurveDisplay, auc)
import numpy as np
from .evaluation import get_cumulative_precision_recall_and_f1


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
        The target label names.

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
        The target label names.
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
        The target label names.
    dataset_name : str
        The name of the dataset used to compute the statistics.
    """
    classification_report_dict = _get_classification_report_dict(y_true, y_pred, targets)

    class_report_sorted = sorted(
        [(classification_report_dict[label]['f1-score'], label) 
         for label in classification_report_dict.keys() if label in targets ])

    class_report_sorted = OrderedDict(
        {label: classification_report_dict[label]
         for (_, label) in class_report_sorted})

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
        The target label names.
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

def _plot_roc_subplot(y_true: np.ndarray, y_scores: np.ndarray,
                      targets: List[str], index: int) -> None:
    """Plot the ROC curve for selected target classes.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels scores.
    targets : list of str
        The target label names.
    index : int
        The current subplot index.
    """
    ax = plt.subplot(2, 2, index)

    for class_id in range(index * 4, index * 4 + 4):
        RocCurveDisplay.from_predictions(
            y_true[:, class_id],
            y_scores[:, class_id],
            name=f"{targets[class_id]}",
            ax=ax
        )

def plot_roc_curves(y_true: np.ndarray, y_scores: np.ndarray,
                    targets: List[str], dataset_name: str) -> None:
    """Plot the Receiving Operator Curves for the predictions.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels scores.
    targets : list of str
        The target label names.
    dataset_name : str
        The name of the dataset.
    """
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(len(targets)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_scores[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points.
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(len(targets)):
        # Linear interpolation.
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

    # Average it and compute AUC.
    mean_tpr /= len(targets)

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")


    plt.figure(figsize=(15,10))
    plt.subplot(2, 2, 1)
    plt.suptitle('Receiver Operating Characteristic One-vs-Rest curves on the'
                 f' {dataset_name} set')

    for i in range(4):
        _plot_roc_subplot(y_true, y_scores, targets, i + 1)
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

def plot_precision_recall_f1_macro_curves(
    y_true: np.ndarray, y_scores: np.ndarray, targets: List[str],
    dataset_name: str) -> None:
    """Plot the Receiving Operator Curves for the predictions.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels scores.
    targets : list of str
        The target label names.
    dataset_name : str
        The name of the dataset.
    """
    s_preds, c_precision, c_recall, c_f1 = \
        get_cumulative_precision_recall_and_f1(y_true, y_scores)
    plt.figure(figsize=(20, 15))
    plt.suptitle(f'Precision, Recall and F1 macro score on the {dataset_name} '
                 'set according to a threshold per class\n', size=14)
    for i, l in enumerate(targets):
        ax = plt.subplot(4, 5, i + 1)
        ax.set_title(l)
        ax.plot(s_preds[:,i], c_precision[:,i], label='precision')
        ax.plot(s_preds[:,i], c_recall[:,i], label='recall')
        ax.plot(s_preds[:,i], c_f1[:,i], label='F1 macro')
        ax.set_xlabel('threshold')
        ax.set_ylabel('score')
        ax.legend()
    plt.tight_layout()
    plt.show()
