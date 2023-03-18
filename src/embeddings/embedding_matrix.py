"""Module providing functions to handle the embedding matrix."""
import os
from typing import Dict, List
import numpy as np


def load_embedding_model(
    folder_path : str, embedding_dim : int = 50,
    extended_version : bool = False) -> Dict[str, np.ndarray]:
    """Load the GLOVE embeddings.

    Parameters
    ----------
    folder_path : str
        Path of the folder containing the GLOVE embeddings.
    embedding_dim : int, optional
        Embedding dimension, by default 50
    extended_version : bool, optional
        Whether to use the extendend GLOVE embeddings, covering also the OOV
        words of our dataset, or not, by default False.

    Returns
    -------
    GLOVE_embeddings : { str: ndarray }
        Dictionary mapping word types into np.array embedding vectors.
    """
    GLOVE_embeddings = []

    if not extended_version:
        file_path = os.path.join(folder_path,
                                 f'glove.6B.{embedding_dim}d.txt')
    else:
        file_path = os.path.join(folder_path,
                                 f'extended_glove.{embedding_dim}d.txt')

    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read().splitlines()
        GLOVE_embeddings += text

    GLOVE_embeddings = [ line.split() for line in GLOVE_embeddings
                        if len(line) > 0 ]
    GLOVE_embeddings = { line[0]: np.array(line[1:], dtype=np.float32)
                        for line in GLOVE_embeddings }

    return GLOVE_embeddings

def build_embedding_matrix(
    vocabulary : List[str], GLOVE_embeddings : Dict[str, np.ndarray],
    embedding_dimension : int = 50) -> np.ndarray:
    """Build the embedding matrix from the given vocabulary and the 
    GLOVE embeddings.

    Parameters
    ----------
    vocabulary : list of str
        List of strings, representing the vocabulary: mapping 
        integers -> word types.
    GLOVE_embeddings : { str: ndarray }
        Dictionary mapping word types into embedding vectors.
    embedding_dimension : int, optional
        Dimension of the embedding, by default 50.

    Returns
    -------
    embedding_matrix : ndarray
        The rows are the word integers (given by `vocabulary`), and each
        row represents the embedding vector of that corresponding word.
    """
    embedding_matrix = np.zeros((len(vocabulary), embedding_dimension),
                                dtype=np.float32)
    for idx, word in enumerate(vocabulary):
        try:
            embedding_vector = GLOVE_embeddings[word]
        except (KeyError, TypeError):
            embedding_vector = np.random.uniform(low=-0.05, high=0.05,
                                                 size=embedding_dimension)

        embedding_matrix[idx] = embedding_vector

    return embedding_matrix
