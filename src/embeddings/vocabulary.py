"""Module providing functions to build a vocabulary for the model."""
from collections import Counter
from string import punctuation
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def get_vocabulary(texts : List[List[str]], add_padding_token : bool = True
                   ) -> Tuple[np.ndarray, List[List[int]]]:
    """Create the vocabulary from  the given texts.

    Parameters
    ----------
    texts : list of list of str
        Each list is a sentence, represented as a list of strings (i.e. words).
    add_padding_token : bool, optional
        Whether to add the padding token '' into the vocabulary, by default
        True.

    Returns
    -------
    vocabulary : ndarray
        Array of strings, representing the mapping from integer ids to words. 
        The first entry, i.e. index 0, is reserved for the padding:
        mapping 0 -> ''.
        The entries in this vocabulary are sorted by frequence in descending
        order: the word with index 1 is the most 
        frequent word.
    tokenizer : {str: int}
        Dictionary containing for each word its corresponding id.
    """
    texts_flat = [word for text in texts for word in text]
    tokens = np.array(list(Counter(texts_flat).keys()))
    tokens_counts = list(Counter(texts_flat).values())
    tokens = tokens[np.argsort(tokens_counts)][::-1]

    if add_padding_token:
        vocabulary = np.array([''] +  list(tokens))
    else:
        vocabulary = tokens

    tokenizer = { word: id for id, word in enumerate(vocabulary) }

    return vocabulary, tokenizer

def get_texts_from_arguments_dataframes(stance_encoder: Dict[str, str],
    *dataframes: pd.DataFrame) -> List[List[str]]:
    """Get the list of sentences in the form 
    '<premise> [FAV]/[AGN] <conclusion>' from a series of arguments
    dataframes.

    Each sentence is preprocessed by eliminating trailing punctuation
    and by turning it into lowercase.
    
    Each sentence is split and returned as a list of single words.

    Parameters
    ----------
    stance_encoder : { str: str }
        Dictionary containing the encoding for each stance.
    *dataframes: (DataFrame, ...)
        The argument dataframes that are used to build the list of sentences.
    Returns
    -------
    list of list of str
        The list of preprocessed sentences.
    """
    texts = []

    for df in dataframes:
        for row in df.itertuples():
            stance = stance_encoder[row.Stance]
            split_premise = [w.strip(punctuation + ' ')
                             for w in row.Premise.split()]
            processed_premise = ' '.join(split_premise).lower()

            split_conclusion = [w.strip(punctuation + ' ')
                                for w in row.Conclusion.split()]
            processed_conclusion = ' '.join(split_conclusion).lower()

            texts.append(
                processed_premise + f' {stance} ' + processed_conclusion)

    return [t.split() for t in texts]
