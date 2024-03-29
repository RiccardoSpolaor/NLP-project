"""Module providing a function to get a transformer model for the 
identification of human values behind arguments."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_transformer_model(name: str, num_labels: int, tokenizer: AutoTokenizer,
                          device: str) -> AutoModelForSequenceClassification:
    """Get a transformer-based model for the identification of human values
    behind arguments. 

    Parameters
    ----------
    name : str
        Name of the pre-trained model.
    num_labels : int
        Number of labels to consider for classification.
    tokenizer : AutoTokenizer
        The autotokenizer to encode the input data.
    device : str
        The device used to run the model.

    Returns
    -------
    AutoModelForSequenceClassification
        The transformer-based model
    """
    # Get the model.
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=num_labels, ignore_mismatched_sizes=True,
        problem_type='multi_label_classification')

    # Resize the embedding matrix to the tokenizer new length.
    model.resize_token_embeddings(len(tokenizer))

    # Assign the model to the specified device.
    model.to(device)

    return model
