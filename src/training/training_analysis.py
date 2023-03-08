"""Module providing functions to plot the training procedure."""
import matplotlib.pyplot as plt
import numpy as np

def _plot_loss_subplot(
    train_loss_history: np.ndarray, val_loss_history: np.ndarray,
    subplot_index: int, n_batches: int, use_log_scale: bool = False) -> None:
    """Plot the loss subplot.

    Parameters
    ----------
    train_loss_history : ndarray
        The train loss history.
    val_loss_history : ndarray
        The validation loss history.
    subplot_index : int
        The index of the subplot.
    n_batches : int
        The number of batches used for aggregation.
    use_log_scale : bool, optional
        Whether to plot the results in log scale or not, by default False.
    """
    plt.subplot(2, 2, subplot_index)

    plt.title(f'Loss history{" using log scale" if use_log_scale else ""}')

    plt.plot(train_loss_history, label='Training loss')

    averaged_train_history = np.convolve(
        train_loss_history, np.ones(n_batches)/n_batches, mode='valid')

    plt.plot(np.linspace(0, len(train_loss_history), 
                         len(averaged_train_history)), 
             averaged_train_history,
             label=f'Training loss averaged on {n_batches} batches')

    # Plot validation history if present.
    plt.plot(val_loss_history[:,0], val_loss_history[:,1], 'r*',
             label='Validation loss')

    plt.xlabel('iterations')

    # Use log scale if specified.
    if use_log_scale:
        plt.yscale('log')
        plt.ylabel('loss (log)')
    else:
        plt.ylabel('loss')

    plt.legend()

def _plot_f1_subplot(validation_f1_macro_history: np.ndarray,
                     subplot_index: int) -> None:
    """Plot the F1 macro score subplot on the validation set.

    Parameters
    ----------
    validation_f1_history : ndarray
        The validation F1 macro score history.
    subplot_index : int
        The index of the subplot.
    """
    plt.subplot(2, 1, subplot_index)

    plt.title('F1 macro history on the validation set')

    plt.plot(validation_f1_macro_history[:,0],
             validation_f1_macro_history[:,1], 'r')

    plt.xlabel('iterations')

    plt.ylabel('F1 macro')

def plot_training_history(
    train_loss_history: np.ndarray, val_loss_history: np.ndarray,
    val_f1_macro_history: np.ndarray, n_batches: int = 50) -> None:
    """Plot the training history.

    Parameters
    ----------
    train_loss_history : ndarray
        The train loss history.
    val_loss_history : ndarray
        The validation loss history.
    val_f1_macro_history : ndarray
        The validation F1 macro score history.
    n_batches: int, optional
        The number of batches used for aggregation, by default 50.
    """
    plt.figure(figsize=(15,12))
    plt.subplot(2, 2, 1)
    plt.suptitle('Training procedure analysis')

    # Plot loss history.
    _plot_loss_subplot(train_loss_history, val_loss_history, 1, n_batches,
                       use_log_scale=False)

    # Plot log loss history.
    _plot_loss_subplot(train_loss_history, val_loss_history, 2, n_batches,
                       use_log_scale=True)

    # Plot validation F1 macro history.
    _plot_f1_subplot(val_f1_macro_history, 2)

    plt.tight_layout()
    plt.show()
