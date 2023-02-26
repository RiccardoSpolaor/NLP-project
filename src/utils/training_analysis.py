import matplotlib.pyplot as plt
import numpy as np

def _plot_loss_subplot(train_loss_history: np.ndarray, val_loss_history: np.ndarray, subplot_index: int, n_batches: int,
                       use_log_scale: bool = False) -> None:
    plt.subplot(2, 2, subplot_index)

    plt.title(f'Loss history{" using log scale" if use_log_scale else ""}')

    plt.plot(train_loss_history, label='Training loss')
    
    averaged_train_history = np.convolve(train_loss_history, np.ones(n_batches)/n_batches, mode='valid')
    
    plt.plot(np.linspace(0, len(train_loss_history), len(averaged_train_history)), averaged_train_history,
             label=f'Training loss averaged on {n_batches} batches')

    # Plot validation history if present
    plt.plot(val_loss_history[:,0], val_loss_history[:,1], 'r*', label=f'Validation loss')
    
    plt.xlabel('iterations')
    
    # Use log scale if specified
    if use_log_scale:
        plt.yscale('log')
        plt.ylabel(f'loss (log)')
    else:
        plt.ylabel('loss')

    plt.legend()
    
def _plot_f1_subplot(validation_f1_history: np.ndarray, subplot_index: int) -> None:
    plt.subplot(2, 1, subplot_index)

    plt.title(f'F1 macro history on the validation set')

    #plt.plot(val_f1_macro_history)
    plt.plot(validation_f1_history[:,0], validation_f1_history[:,1], 'r')
    
    # averaged_train_history = np.convolve(train_history, np.ones(n_batches)/n_batches, mode='valid')
    
    #plt.plot(np.linspace(0, len(train_history), len(averaged_train_history)), averaged_train_history,
    #         label=f'Training {metric} averaged on {n_batches} batches')
    
    #plt.plot(np.convolve(training_loss_history[:,loss_index], np.ones(n_batches)/n_batches, mode='valid'), 
    #         label=f'Training loss averaged on {n_batches} batches')

    # Plot validation history if present
    #plt.plot(validation_history[:,0], validation_history[:,1], 'r*', label=f'Validation {metric}')
    
    plt.xlabel('iterations')
    
    # Use log scale if specified
    #if use_log_scale:
    #    plt.yscale('log')
    #    plt.ylabel(f'{metric} (log)')
    #else:
    plt.ylabel('F1 macro')

    #plt.legend()


def plot_training_history(train_loss_history: np.ndarray, val_loss_history: np.ndarray,
                          val_f1_macro_history: np.ndarray) -> None:
    n_batches = 50

    plt.figure(figsize=(15,12))
    plt.subplot(2, 2, 1)
    plt.suptitle('Training procedure analysis')
    
    # Plot loss history
    _plot_loss_subplot(train_loss_history, val_loss_history, 1, n_batches, use_log_scale=False)
    
    # Plot log loss history
    _plot_loss_subplot(train_loss_history, val_loss_history, 2, n_batches, use_log_scale=True)

    # Plot validation F1 history
    _plot_f1_subplot(val_f1_macro_history, 2)

    # Plot loss history of the Seq2seq module in log scale
    #_plot_loss_subplot(train_accuracy_history, validation_accuracy_history, 4, n_batches, use_log_scale=True)
    plt.tight_layout()
    plt.show()
