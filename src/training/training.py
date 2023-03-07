
from time import time
import torch
from transformers import AutoModelForSequenceClassification
from typing import Optional
from sklearn.metrics import f1_score

import numpy as np
from torch.utils.data import DataLoader

from .training_utils import Checkpoint, EarlyStopping
from ..evaluation.evaluation import get_dataset_prediction_scores, get_best_thresholds

def _loss_validate(model: AutoModelForSequenceClassification, val_dataloader: DataLoader, loss_function: torch.nn.Module,
                   device: str, use_threshold_selection: bool, print_result: bool = True):
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

def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: AutoModelForSequenceClassification, 
          optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module, device: str, epochs: int = 5,
          steps_validate: int = 100, checkpoint: Optional[Checkpoint] = None, early_stopping: Optional[EarlyStopping] = None, 
          reload_best_weights: bool = True, use_threshold_selection: bool = True) -> None:
    train_loss_history = []
    val_loss_history = []
    val_f1_macro_history = []

    # Total steps to perform
    # tot_steps = len(train_dataloader) * epochs
    # Number of step already done
    n_steps = 0
    
    model.train()

    # Iterate across the epochs
    for epoch in range(epochs):
        # Set up display element
        #disp = display('', display_id=True)

        # Remove unused tensors from gpu memory
        torch.cuda.empty_cache()

        # Initialize running losses
        running_loss = 0.0
        
        optimizer.zero_grad()

        start_time = time()

        # Number of batches for the current update step
        batch_steps = 0

        for batch_idx, data in enumerate(train_dataloader, 0):
            # Increment the number of batch steps
            batch_steps += 1

            # Get the data
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.float32)
        
            # Compute output
            outputs = model(ids, mask)
            
            # Loss
            loss = loss_function(outputs.logits, targets)
            running_loss += loss.item()

            optimizer.zero_grad()
            #nb_tr_steps += 1
            loss.backward()
            
            # When using GPU
            optimizer.step()

            # Evaluate on validation set
            if batch_idx % steps_validate == steps_validate - 1:
                model.eval()
                torch.cuda.empty_cache()

                # Compute both the token importances validation loss and the answer generation validation loss
                val_loss, val_f1_macro = _loss_validate(model, val_dataloader, loss_function, device,
                                                        use_threshold_selection)
                
                # Update validation loss history
                val_loss_history.append([n_steps, val_loss.item()])
                val_f1_macro_history.append([n_steps, val_f1_macro])

                torch.cuda.empty_cache()
                
                if checkpoint is not None:
                    checkpoint.save_best(val_f1_macro, train_loss_history=train_loss_history,
                                         val_loss_history=val_loss_history, val_f1_macro_history=val_f1_macro_history)
                if early_stopping is not None:
                    early_stopping.update(val_f1_macro)
                    if early_stopping.is_stop_condition_met():
                        print('Early stopping')
                        return train_loss_history, val_loss_history, val_f1_macro_history

                model.train()


            # Update training history and print           
            train_loss_history.append(loss.detach().cpu())
            
            epoch_time = time() - start_time
            batch_time = epoch_time / (batch_idx + 1)
            
            # TODO: function to print batch string
            print(
                f'epoch: {epoch + 1}/{epochs},',
                f'{batch_idx + 1}/{len(train_dataloader)},',
                f'{epoch_time:.0f}s {batch_time * 1e3:.0f}ms/step,',
                f'loss: {running_loss / batch_steps:.3g}',
                '               ',
                end='\r')

            n_steps += 1

        model.eval()
        torch.cuda.empty_cache()
        # Compute both the token importances validation loss and the answer generation validation loss
        val_loss, val_f1_macro = _loss_validate(model, val_dataloader, loss_function, device,
                                                use_threshold_selection, print_result=False)
        # Update validation loss history
        val_loss_history.append([n_steps, val_loss.item()])
        val_f1_macro_history.append([n_steps, val_f1_macro])

        torch.cuda.empty_cache()

        print(DIV_STR)
        print(
            f'epoch: {epoch + 1}/{epochs},',
            f'{epoch_time:.0f}s,',
            f'loss: {running_loss / batch_steps:.3g},',
            f'val loss:, {val_loss.mean():.3g},',
            f'val f1 macro: {val_f1_macro * 100:.3g} %')
        print(BIG_DIV_STR)
        
        if checkpoint is not None:
            checkpoint.save_best(val_f1_macro, train_loss_history=train_loss_history, val_loss_history=val_loss_history,
                            val_f1_macro_history=val_f1_macro_history)

        if early_stopping is not None:
            early_stopping.update(val_f1_macro)
            if early_stopping.is_stop_condition_met():
                print('Early stopping')
                return train_loss_history, val_loss_history, val_f1_macro_history
        model.train()

    if checkpoint is not None and reload_best_weights:
        _ = checkpoint.load_best()

    model.eval()
    return np.array(train_loss_history), np.array(val_loss_history), np.array(val_f1_macro_history)