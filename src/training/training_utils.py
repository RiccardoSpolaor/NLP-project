import numpy as np
import os
import torch
from transformers import AutoModelForSequenceClassification
from typing import Optional

class FocalLoss(torch.nn.Module):
    """Class implementing the focal loss function"""
    def __init__(self, alpha: np.ndarray, gamma: float = 1.):
        """Get the focal loss module. The focal loss functions downweights easy examples to classify and upweights hard examples.
        The loss score is computed accordingly.

        Parameters
        ----------
        alpha : ndarray
            Vector of frequencies of each target
        gamma : float, optional
            Parameter to tweak the focal loss, by default 1
        """
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Forward pass of the focal loss computing a score for a minibatch.
        Given the true targets `targets`. It computes the following:
        - The probabilities `p` as the sigmoid of the predicted scores.
        - The true probabilities `p_t`:
            - p_t[i] = p[i] if targets[i] = 1
            - p_t[i] = 1 - p[i] otherwise
        - the vector `alpha_t` which is based on the vector of frequencies of labels `alpha`
            - alpha_t[i] = alpha[i] if targets[i] = 1
            - alpha_t[i] = 1 - alpha[i] otherwise
        
        The loss is given as the mean of:
        - alpha_t * (1 - p_t)^gamma * log(p_t)

        Parameters
        ----------
        inputs : Tensor
            The predicted scores.
        targets : Tensor
            The true targets.

        Returns
        -------
        Tensor
            The averaged focal loss score among all the instances of the minibatch.
        """
        alpha = self.alpha.to(predictions.device)
        
        p = predictions.sigmoid()
        
        # Probability of the true class
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * (1 - alpha) + (1 - targets) * alpha
        
        loss = - alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)

        return loss.mean()
    
class Checkpoint():
    """Class to handle the checkpoints of a model"""
    def __init__(self, model: AutoModelForSequenceClassification, path: Optional[str] = None) -> None:
        """Initialize the checkpoint instance.

        Parameters
        ----------
        model : AutoModelForSequenceClassification
            The model from which the weights are saved in or loaded from the checkpoints.
        path : str, optional
            The checkpoint path, by default None.
        """
        self.model = model
        self.last_metric = .0
        self.path = path if path is not None else os.path.join('checkpoints', 'best_model.pth')

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def save_best(self, new_metric: float, **kwargs) -> None:
        """Possibly save the best model weights in the checkpoint according to the new value of the metric.
        Parameters defined in `kwargs` are also saved in the checkpoints as an ndarray. 

        Parameters
        ----------
        new_metric : float
            The new value of the metric. It is compared to the best obtained so far and the checkpoints are updated solely if it is better.
        kwargs : Any
            Named arguments saved in the checkpoints as ndarrays.
        """
        if new_metric > self.last_metric:
            checkpoint = { k: np.array(v) for k, v in kwargs.items()}
            checkpoint['model_state_dict'] = self.model.state_dict(),

            torch.save(checkpoint, self.path)

        self.last_metric = new_metric
    
    def load_best(self) -> AutoModelForSequenceClassification:
        """Load the best weights of the model

        Returns
        -------
        AutoModelForSequenceClassification
            The model for sequence classification with the weights loaded from the best checkpoint.
        """
        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model_state_dict'][0])
        return self.model

class EarlyStopping():
    """Class defining an early stopping monitor"""
    def __init__(self, patience=5, tolerance=.5) -> None:
        """Create an instance of an early stopping monitor.
        It saves the early stopping condition inside a boolean variable and updates it when asked by the user.

        Parameters
        ----------
        patience : int, optional
            Integer specifying how many times the result can get worse before the early stopping condition is triggered, by default 5
        tolerance : float, optional
            Float representing the tolerance considered for the comparison between the previous best result and the new one.
            If the difference between the best previous metric and the new considered one is greater than this value the result is considered worse and the counter of "failures" is incremented, by default 0.5
        """
        self.patience = patience
        self.tolerance = tolerance
        self.trigger_times = 0
        self.max_metric = 0.
        self.stop_condition = False

    def update(self, metric: float) -> None:
        """Update the state of the early stopping based on the new metric value.
        If the difference between the previous saved best metric and the new one is less or equal than the tolerance, the counter of "failures" resets.
        Moreover, if the new metric is better than the previous one the current value is saved as the best one.
        Otherwise, in the case the difference is greater than tolerance, the counter of "failures" is incremented and, if it reaches the patience value, the stop condition is triggered

        Parameters
        ----------
        metric : float
            The new metric value
        """
        if self.max_metric - metric <= self.tolerance:
            self.trigger_times = 0
            if metric > self.max_metric:
                self.max_metric = metric
        else:
            self.trigger_times += 1

            if self.trigger_times >= self.patience:
                self.stop_condition = True

    def is_stop_condition_met(self) -> bool:
        """Return whether the stop condition is met or not.

        Returns
        -------
        bool
            Whether the stop condition is met or not.
        """
        return self.stop_condition
