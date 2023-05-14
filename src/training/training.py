
"""Module providing the function to perform the training."""
from time import time
from typing import Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

from .training_utils import Checkpoint, EarlyStopping
from ..evaluation.evaluation import (get_dataset_prediction_scores,
                                     get_best_thresholds)


def _get_validation_loss_and_f1(
    model: nn.Module, val_dataloader: DataLoader,
    loss_function: nn.Module, device: str, use_threshold_selection: bool,
    print_result: bool = True) -> Tuple[float, float]:
    """Function to compute the validation loss and F1 macro score.

    Parameters
    ----------
    model : Module
        The model used to compute the metrics.
    val_dataloader : DataLoader
        The dataloader used to iterate through the validation dataset.
    loss_function : Module
        The loss function to consider.
    device : str
        The device onto which attach the validation process.
    use_threshold_selection : bool
        Whether to use threshold selection or not.
    print_result : bool, optional
        Whether to print the resuklts or not, by default True.

    Returns
    -------
    (float, float)
        Tuple containing the validation loss and F1 macro score.
    """
    print()
    start_time = time()

    torch.cuda.empty_cache()

    y_scores, y_true = get_dataset_prediction_scores(model, val_dataloader, device)

    loss = loss_function(torch.Tensor(y_scores).to(device, dtype=torch.float32),
                         torch.Tensor(y_true).to(device, dtype=torch.float32))

    if use_threshold_selection:
        thresholds_per_target = get_best_thresholds(y_true, y_scores)
        y_pred = y_scores > thresholds_per_target
    else:
        y_pred = y_scores > 0.

    y_pred = y_pred.astype(np.uint8)

    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    final_time = time() - start_time

    if print_result:
        print(
            'validate:',
            f'{final_time:.0f}s,',
            f'validation loss: {loss:.3g},',
            f'validation f1 macro: {f1_macro * 100:.3g} %',
            '               ')

    return loss, f1_macro

DIV_STR =     '---------------------------------------------------------------'
BIG_DIV_STR = '==============================================================='

def train(
    train_dataloader: DataLoader, val_dataloader: DataLoader,
    model: nn.Module, optimizer: torch.optim.Optimizer,
    loss_function: nn.Module, device: str, epochs: int = 5,
    steps_validate: int = 100, checkpoint: Optional[Checkpoint] = None,
    early_stopping: Optional[EarlyStopping] = None,
    reload_best_weights: bool = True,
    use_threshold_selection: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train the model and get the training history.

    Parameters
    ----------
    train_dataloader : DataLoader
        The dataloader used to iterate through the train dataset.
    val_dataloader : DataLoader
        The dataloader used to iterate through the validation dataset.
    model : Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer.
    loss_function : torch.nn.Module
        The loss function to use.
    device : str
        The device onto which attach the training process.
    epochs : int, optional
        The number of training epochs, by default 5.
    steps_validate : int, optional
        The number of steps after which validation is performed, by
        default 100.
    checkpoint : Checkpoint, optional
        The checkpoint monitor callback to use, by default None.
    early_stopping : EarlyStopping, optional
        The Early Stopping monitor callback to use, by default None.
    reload_best_weights : bool, optional
        Whether to reload the best weights of the model after the training
        proceure. If `checkpoint` is None, the best weights cannot be
        reloaded, by default True.
    use_threshold_selection : bool, optional
        Whether to use threshold selection or not, by default True

    Returns
    -------
    ndarray
        The train loss history.
    ndarray
        The validation loss history.
    ndarray
        The validation F1 macro score history.
    """
    train_loss_history = []
    val_loss_history = []
    val_f1_macro_history = []

    # Number of step already done.
    n_steps = 0

    # Set the model in train mode.
    model.train()

    # Iterate across the epochs
    for epoch in range(epochs):
        # Remove unused tensors from gpu memory.
        torch.cuda.empty_cache()

        # Initialize running loss.
        running_loss = 0.0

        optimizer.zero_grad()

        start_time = time()

        # Number of batches for the current update step.
        batch_steps = 0

        for batch_idx, data in enumerate(train_dataloader, 0):
            # Increment the number of batch steps.
            batch_steps += 1

            # Get the data.
            ids = data['ids'].to(device)
            targets = data['labels'].to(device, dtype = torch.float32)

            # Compute output.
            if 'mask' in data.keys():
                mask = data['mask'].to(device)
                outputs = model(ids, mask)
            else:
                outputs = model(ids)

            # Loss.
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            loss = loss_function(outputs, targets)
            running_loss += loss.item()

            # Zero the gradients and backpropagate.
            optimizer.zero_grad()
            loss.backward()

            # Do the optimization step.
            optimizer.step()

            # Evaluate on validation set.
            if batch_idx % steps_validate == steps_validate - 1:
                # Set the model in eval mode.
                model.eval()

                # Remove unused tensors from gpu memory.
                torch.cuda.empty_cache()

                # Compute the validation loss and the F1 macro score on
                # the validation set.
                val_loss, val_f1_macro = _get_validation_loss_and_f1(
                    model, val_dataloader, loss_function, device,
                    use_threshold_selection)

                # Update validation loss and F1 score macro history.
                val_loss_history.append([n_steps, val_loss.item()])
                val_f1_macro_history.append([n_steps, val_f1_macro])

                # Remove unused tensors from gpu memory.
                torch.cuda.empty_cache()

                # Save the checpoints and check the early stopping criteria.
                if checkpoint is not None:
                    checkpoint.save_best(
                        val_f1_macro, train_loss_history=train_loss_history,
                        val_loss_history=val_loss_history,
                        val_f1_macro_history=val_f1_macro_history)
                if early_stopping is not None:
                    early_stopping.update(val_f1_macro)
                    if early_stopping.is_stop_condition_met():
                        print('Early stopping')
                        return (train_loss_history, val_loss_history,
                                val_f1_macro_history)
                # Set the model in train mode.
                model.train()

            # Update training history and print.
            train_loss_history.append(loss.detach().cpu())

            epoch_time = time() - start_time
            batch_time = epoch_time / (batch_idx + 1)

            print(
                f'epoch: {epoch + 1}/{epochs},',
                f'{batch_idx + 1}/{len(train_dataloader)},',
                f'{epoch_time:.0f}s {batch_time * 1e3:.0f}ms/step,',
                f'loss: {running_loss / batch_steps:.3g}',
                '               ',
                end='\r')

            n_steps += 1

        # Set the model in eval mode.
        model.eval()

        # Remove unused tensors from gpu memory.
        torch.cuda.empty_cache()

        # Compute the validation loss and the F1 macro score on the
        # validation set.
        val_loss, val_f1_macro = _get_validation_loss_and_f1(
            model, val_dataloader, loss_function, device,
            use_threshold_selection, print_result=False)

        # Update validation loss history
        val_loss_history.append([n_steps, val_loss.item()])
        val_f1_macro_history.append([n_steps, val_f1_macro])

        # Remove unused tensors from gpu memory.
        torch.cuda.empty_cache()

        # Print the results.
        print(DIV_STR)
        print(
            f'epoch: {epoch + 1}/{epochs},',
            f'{epoch_time:.0f}s,',
            f'loss: {running_loss / batch_steps:.3g},',
            f'val loss:, {val_loss.mean():.3g},',
            f'val f1 macro: {val_f1_macro * 100:.3g} %')
        print(BIG_DIV_STR)

        # Save the checpoints and check the early stopping criteria.
        if checkpoint is not None:
            checkpoint.save_best(
                val_f1_macro, train_loss_history=train_loss_history,
                val_loss_history=val_loss_history,
                val_f1_macro_history=val_f1_macro_history)
        if early_stopping is not None:
            early_stopping.update(val_f1_macro)
            if early_stopping.is_stop_condition_met():
                print('Early stopping')
                return (train_loss_history, val_loss_history,
                        val_f1_macro_history)

        # Set the model in train mode.
        model.train()

    checkpoint.save_last( train_loss_history = train_loss_history,
                          val_loss_history = val_loss_history,
                          val_f1_macro_history = val_f1_macro_history)

    # Reload the best weights if specified.
    if checkpoint is not None and reload_best_weights:
        _ = checkpoint.load_best()

    # Set the model in eval mode.
    model.eval()

    return (np.array(train_loss_history), np.array(val_loss_history),
            np.array(val_f1_macro_history))
