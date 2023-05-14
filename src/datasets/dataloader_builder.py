"""Module providing a function to build the dataloader."""
from random import sample
from string import punctuation
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class HumanValueDataset(Dataset):
    """The Human Value Dataset. Extends the class Dataset of torch."""
    def __init__(self, arguments_df: pd.DataFrame, labels_df: pd.DataFrame,
                 stance_encoder: Dict[str, str]) -> None:
        """Create an instance of the Human Value Dataset from a certain 
        dataframe.

        Parameters
        ----------
        arguments_df : DataFrame
            The arguments dataframe.
        labels_df : DataFrame
            The labels dataframe.
        stance encoder : { str: str }
            Dictionary containing the encoding for each stance.
        """
        self.len = len(arguments_df)
        # The arguments dataframe
        self.arguments_data = arguments_df[[
            'Premise', 'Conclusion', 'Stance']].to_numpy()
        # The labels dataframe
        self.labels_data = labels_df.to_numpy()
        # Encoder of the stance string into the relative tokenizer tokens
        self.stance_encoder = stance_encoder

    def __getitem__(self, index: int) -> Tuple[str, str, str, 
                                               List[np.ndarray[int]]]:
        """Get the item at a certain index in the dataset.

        Parameters
        ----------
        index : int
            Index of the item to obtain.

        Returns
        -------
        str
            The current item encoded as '<premise>'.
        str
            The current item encoded as '<conclusion>'.
        str
            The current item encoded as '<premise> [FAV]/[AGN] <conclusion>'.
        ndarray of int
            The targets vector.
        """
        # Get the premise, conclusion and stance at the current index
        arguments_data = self.arguments_data[index]
        premise = arguments_data[0]
        conclusion = arguments_data[1]
        stance = arguments_data[2]

        # Encode the stance into `[FAV]` or `[AGN]`
        encoded_stance = self.stance_encoder[stance]

        # Get the targets vector
        targets_vector = self.labels_data[index]

        # Get the whole text as: '<premise> [FAV]/[AGN] <conclusion>'
        whole_text = premise + f' {encoded_stance} ' + conclusion

        return premise, conclusion, whole_text, targets_vector

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the datadet.
        """
        return self.len

def _collate_batch_transformer(
    batch: Tuple[Tuple[str, str, str, List[np.ndarray[int]]]],
    tokenizer: AutoTokenizer, use_all_instance: bool = False
    ) -> Dict[str, torch.Tensor]:
    """Function to transforms a minibatch of samples into a format useful for
    the training procedure of a transformer-based model.

    Parameters
    ----------
    batch : ((str, str, str, list of int), ...)
        The input minibatch.
    tokenizer : AutoTokenizer
        The autotokenizer to encode the input data.
    use_all_instance : bool, optional
        Whether to use all instance data or just the premise, by default False.

    Returns
    -------
    { 'ids': Tensor, 'mask': Tensor, 'labels': Tensor }
        Dictionary of tensors containing the encoded ids of the minibatch,
        their attention masks and the respective labels.
    """
    # Create a numpy matrix for the input texts and the labels
    input_texts = np.zeros(shape=(len(batch),), dtype=object)
    labels = np.zeros(shape=(len(batch), len(batch[0][3])))

    for i, (p, _, w, l) in enumerate(batch):
        # Get '<premise> [FAV]/[AGN] <conclusion>'
        if use_all_instance:
            result = w
        # Get '<premise>'
        else:
            result = p
        # Assign to the matrices at the given index the text and the labels
        input_texts[i] = result
        labels[i] = l

    # Encode the input text
    inputs = tokenizer(
        input_texts.tolist(),
        None,
        add_special_tokens=True,
        max_length=None,
        padding=True,
        truncation=True,
        return_tensors='pt')

    # Get input ids and atetntion mask from the encoded input texts.
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    # Get the results in a dictionary.
    return {
        'ids': ids.type(torch.long),
        'mask': mask.type(torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

def _collate_batch_lstm(
    batch: Tuple[Tuple[str, str, str, List[np.ndarray[int]]]],
    tokenizer: Dict[str, int], use_all_instance: bool = False
    ) -> Dict[str, torch.Tensor]:
    """Function to transforms a minibatch of samples into a format useful for
    the training procedure of the LSTM model.

    Parameters
    ----------
    batch : ((str, str, str, list of int), ...)
        The input minibatch.
    tokenizer : { str, int }
        The tokenizer to encode the input data.
    use_all_instance : bool, optional
        Whether to use all instance data or just the premise, by default False.

    Returns
    -------
    { 'ids': Tensor, 'labels': Tensor }
        Dictionary of tensors containing the encoded ids of the minibatch
        and the respective labels.
    """
    # Create a list for the input texts.
    inputs = []
    # Create a numpy array for the labels.
    labels = np.zeros(shape=(len(batch), len(batch[0][3])))

    for i, (p, _, w, l) in enumerate(batch):
        # Get '<premise> [FAV]/[AGN] <conclusion>'
        if use_all_instance:
            result = p
        # Get '<premise>'
        else:
            result = w
        # Pre-process by stripping punctuation and turn to lowercase.
        split_result = [w.strip(punctuation + ' ').lower() 
                        if not '[AGN]' in w and not '[FAV]' in w else w.strip()
                        for w in result.split()]
        result = ' '.join(split_result)

        # Encode the current sequence and appen it to the inputs array.
        inputs.append(np.array([tokenizer[word] for word in result.split()]))
        # Assign to the label matrix the current labels.
        labels[i] = l

    # Get the max length of the sentences of the current batch.
    max_len = np.max([len(sequence) for sequence in inputs])

    # Get a list of the sequences of the batch padded to the maximum length
    # with zeroes and concatenate them in one array.
    inputs = [
        torch.tensor(np.concatenate([t, np.zeros(max_len - len(t))])) 
        for t in inputs]
    ids = torch.stack(inputs)

    # Get the results in a dictionary.
    return {
        'ids': ids.type(torch.long),
        'labels': torch.tensor(labels, dtype=torch.float32)
    }

def get_dataloader(arguments_df: pd.DataFrame, labels_df: pd.DataFrame,
                   tokenizer: AutoTokenizer, stance_encoder: Dict[str, str],
                   is_transformer: bool, batch_size: int = 8,
                   shuffle: bool = True, use_all_instance: bool = False
                   ) -> DataLoader:
    """Get a dataloader from the arguments and labels dataframes.

    Parameters
    ----------
    arguments_df : DataFrame
        The arguments dataframe.
    labels_df : DataFrame
        The labels dataframe.
    tokenizer : AutoTokenizer
        The autotokenizer to encode the input data.
    stance encoder : { str: str }
        Dictionary containing the encoding for each stance.
    is_transformer : bool
        Whether the model to use with the data is a transformer model or not.
    batch_size : int, optional
        The batch size, by default 8.
    shuffle : bool, optional
        Whether or not to shuffle the data while creating the dataloader,
        by default True.
    use_all_instance : bool, optional
        Whether to use all instance data or just the premise, by default False.

    Returns
    -------
    DataLoader
        The dataloader.
    """
    # Get dataset.
    dataset = HumanValueDataset(arguments_df, labels_df, stance_encoder)
    # Get collate function.
    collate_fn = _collate_batch_transformer if is_transformer \
        else _collate_batch_lstm

    # Get dataloder.
    data_loader = DataLoader(
        dataset, num_workers=0, shuffle=shuffle, batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer,
                                        use_all_instance==use_all_instance))
    return data_loader
