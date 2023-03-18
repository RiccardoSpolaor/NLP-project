"""Module containing a class to initialize a LSTM model for the 
identification of human values behind arguments"""
import torch
from torch import nn
import numpy as np


class LSTM(nn.Module):
    """Class implementing a LSTM model."""
    def __init__(self, device: str, embedding_matrix: np.ndarray, 
                 n_labels: int, hidden_size: int = 128,
                 n_lstm_layers: int = 2) -> None:
        """Get the LSTM model.

        Parameters
        ----------
        device : str
            The device onto which attach the prediction process.
        embedding_matrix : ndarray
            The rows are the word ids encodings, and each
            row represents the embedding vector of that corresponding word.
        n_labels : int
            The number of prediction labels.
        hidden_size : int, optional
            The hidden feature representation of the model, by default 128.
        n_lstm_layers : int, optional
            The number of LSTM layers, by default 2.
        """
        super().__init__()
        # Get the embedding matrix shape.
        n_embeddings, n_features = embedding_matrix.shape

        # Set the embedding layer.
        self.embedding_layer = nn.Embedding(n_embeddings, n_features)
        # Set the weights of the embedding layer as the ones of the
        # given embedding matrix.
        self.embedding_layer.weight = nn.Parameter(
            torch.tensor(embedding_matrix))

        # Set the LSTM layer.
        self.lstm_layer = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True)
        self.classification_layer = torch.nn.Linear(hidden_size, n_labels)

        # Load the model to the selected device.
        self.to(device)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """Get the LSTM classification logits results on an input.

        Parameters
        ----------
        x : LongTensor
            Input containing a batch of sentences encoded according to their
            ids.

        Returns
        -------
        FloatTensor
            The classification logits results for the given input.
        """
        # Get the embedding representation of the input.
        out = self.embedding_layer(x)
        # Get the last hidden states of the LSTM layer.
        _, (_, c_n) = self.lstm_layer(out)
        out = c_n[-1]
        # Get the classification results.
        out = self.classification_layer(out)
        return out
