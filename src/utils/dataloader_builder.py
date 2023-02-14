import numpy as np
import pandas as pd
from random import sample
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple

class HumanValueDataset(Dataset):
    """ The Human Value Dataset. Extends the class Dataset of torch."""
    def __init__(self, arguments_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
        """Create an instance of the Human Value Dataset from a certain dataframe.

        Parameters
        ----------
        arguments_df : DataFrame
            The arguments dataframe.
        labels_df : DataFrame
            The labels dataframe.
        """
        self.len = len(arguments_df)
        # The arguments dataframe
        self.arguments_data = arguments_df[['Premise', 'Conclusion', 'Stance']].to_numpy()
        # The labels dataframe
        self.labels_data = labels_df.to_numpy()
        # Encoder of the stance string into the relative tokenizer tokens
        self.stance_encoder = {'in favor of': '[FAV]', 'against': '[AGN]'}

    # Casual number between 1 and 3 and depending on that give premise conclusion or both.
    def __getitem__(self, index: int) -> Tuple[str, str, str, List[np.ndarray[int]]]:
        """Get the item at a certain index in the dataset.

        Parameters
        ----------
        index : int
            Index of the item to obtain.

        Returns
        -------
        (str, str, str, ndarray of int)
            The current item encoded as ('<premise>', '<conclusion>', '<premise> [FAV]/[AGN] <conclusion>', <targets vector>).
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

def _collate_batch(batch: Tuple[Tuple[str, str, str, List[np.ndarray[int]]]], tokenizer: AutoTokenizer, 
                   augment_data: bool = False) -> Dict[str, torch.Tensor]:
    """Function to transforms a minibatch of samples into a format useful for the training procedure.

    Parameters
    ----------
    batch : tuple of (str, str, str, list of int)
        The input minibatch.
    tokenizer : AutoTokenizer
        The autotokenizer to encode the input data.
    augment_data : bool, optional
        Whether to augment the data or not, by default False.

    Returns
    -------
    { 'ids': Tensor, 'mask': Tensor, 'labels': Tensor }
        Dictionary of tensors containing the encoded ids of the minibatch, their attention masks and the respective labels.
    """
    # Create a numpy matrix for the input texts and the labels
    input_texts = np.zeros(shape=(len(batch),), dtype=object)
    labels = np.zeros(shape=(len(batch), len(batch[0][3])))

    for i, (p, c, w, l) in enumerate(batch):
        # Get random text among <premise>, <conclusion> and '<premise> [FAV]/[AGN] <conclusion>'
        if augment_data:
            [result] = sample([p, c, w], 1)
        # If no data augmentation is required get '<premise> [FAV]/[AGN] <conclusion>'
        else:
            result = w
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
        'ids': ids,
        'mask': mask,
        'labels': torch.tensor(labels, dtype=torch.float32)
    }

def get_dataloader(arguments_df: pd.DataFrame, labels_df: pd.DataFrame, tokenizer: AutoTokenizer, batch_size: int = 8,
                   shuffle: bool = True, use_augmentation: bool = False) -> DataLoader:
    """Get a dataloader from the arguments and labels dataframes.

    Parameters
    ----------
    arguments_df : DataFrame
        The arguments dataframe.
    labels_df : DataFrame
        The labels dataframe.
    tokenizer : AutoTokenizer
        The autotokenizer to encode the input data.
    batch_size : int, optional
        The batch size, by default 8.
    shuffle : bool, optional
        Whether or not to shuffle the data while creating the dataloader, by default True.
    use_augmentation : bool, optional
        Whether to augment the data or not, by default False.

    Returns
    -------
    DataLoader
        The dataloader.
    """
    # Get dataset
    dataset = HumanValueDataset(arguments_df, labels_df)
    # Get dataloder
    data_loader = DataLoader(dataset, num_workers=0, shuffle=shuffle, batch_size=batch_size, 
                             collate_fn=lambda x: _collate_batch(x, tokenizer, augment_data=use_augmentation))
    return data_loader
