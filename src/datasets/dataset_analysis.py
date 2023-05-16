"""Module providing functions to perform the analysis of the dataset."""
from collections import Counter, OrderedDict
import string
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from wordcloud import WordCloud, STOPWORDS


def plot_stance_distribution(arguments_df: pd.DataFrame,
                             title: str = 'Stance distribution') -> None:
    """Plot the distribution of stances across the dataframe

    Parameters
    ----------
    arguments_df : DataFrame
        A pandas arguments dataframe.
    title : str, optional
        The title of the plot, by default 'Stance distribution'
    """
    # Plot the subplots
    _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                         figsize=(15, 5))
    # Plot the counts of the stances
    arguments_df.Stance.value_counts().plot(ax=ax, kind='bar', width=.25)

    # Set the title
    plt.suptitle(title)

    # Set the axis labels
    ax.set_xlabel('stance')
    ax.set_ylabel('count')

    # Use a tight layout
    plt.tight_layout()

    plt.show()
    
def plot_stance_distributions(
    arguments_dfs: List[pd.DataFrame], labels: List[str],
    title: str = 'Stance distribution') -> None:
    """Plot the distribution of stances across the dataframe

    Parameters
    ----------
    arguments_df : DataFrame
        A pandas arguments dataframe.
    title : str, optional
        The title of the plot, by default 'Stance distribution'
    """
    # Get color map
    cmap = get_cmap('Set1')
    # Set the bars offset
    offsets = np.linspace(0., 1., len(arguments_dfs))
    # Plot the subplots
    _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                         figsize=(5, 4))
    # Plot the counts of the stances for each dataframe
    for i, (arguments_df, label) in enumerate(zip(arguments_dfs, labels)):
        arguments_df.Stance.value_counts(normalize=True).plot(
            ax=ax, kind='bar', width=.1, label=label, color=cmap(i),
            position=offsets[i])

    # Set the title
    plt.suptitle(title)

    # Set the axis labels
    ax.set_xlabel('stance')
    ax.set_ylabel('ratio')

    # Use a tight layout
    plt.tight_layout()
    plt.legend(loc='lower left')


    plt.show()

def plot_sentiment_distribution(
    labels_df: pd.DataFrame,
    title: str = 'Sentiment values distribution') -> None:
    """Plot the distribution of sentiment values across the dataframe

    Parameters
    ----------
    labels_df : DataFrame
        A pandas labels dataframe.
    title : str, optional
        The title of the plot, by default 'Sentiment values distribution'
    """
    # Plot the subplots
    _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                         figsize=(15, 6))
    # Plot the counts of the sentiments
    labels_df.sum().plot(ax=ax, kind='bar')

    # Set the title
    plt.suptitle(title)

    # Set the axis labels
    ax.set_xlabel('sentiment values')
    ax.set_ylabel('count')

    # Use a tight layout
    plt.tight_layout()

    plt.show()
    
def plot_sentiment_distributions(
    label_dfs: List[pd.DataFrame], labels: List[str],
    title: str = 'Sentiment values distribution') -> None:
    """Plot the distribution of sentiment values across the dataframe

    Parameters
    ----------
    labels_df : DataFrame
        A pandas labels dataframe.
    title : str, optional
        The title of the plot, by default 'Sentiment values distribution'
    """
    # Get color map
    cmap = get_cmap('Set1')
    # Set the bars offset
    offsets = np.linspace(0., 1., len(label_dfs))
    # Plot the subplots
    _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                         figsize=(15, 6))
    # Plot the counts of the sentiments
    for i, (label_df, label) in enumerate(zip(label_dfs, labels)):
        label_df.sum().plot(
            ax=ax, kind='bar', width=.2, label=label, color=cmap(i),
            position=offsets[i])

    # Set the title
    plt.suptitle(title)

    # Set the axis labels
    ax.set_xlabel('sentiment values')
    ax.set_ylabel('count')

    # Use a tight layout
    plt.tight_layout()
    
    # Set the legend
    plt.legend()

    plt.show()

def plot_sequence_length_analysis(
    arguments_df: pd.DataFrame, df_name: str = 'dataset',
    percentile: Optional[float] = None) -> None:
    """Analyse the length of the premises + stances + conclusions
    Parameters
    ----------
    arguments_df : DataFrame
        A pandas arguments dataframe.
    df_name : str, optional
        The name of the dataframe, by default 'dataset'.
    percentile : float, optional
        The percentile to plot, by default None. If None it is not plotted.
    """
    # Get the length of each training sentence
    train_sentences_lenghts = arguments_df.Premise.str.len() + \
        arguments_df.Conclusion.str.len() + 1

    # Get the histogram of the sentences length distribution
    hist, bin_edges = np.histogram(
        train_sentences_lenghts, density=True,
        bins=np.max(train_sentences_lenghts) + 1) 
    # Get the cumulative distribution of the sentences length
    C = np.cumsum(hist)*(bin_edges[1] - bin_edges[0])

    # Get the quantile if a percentile is provided
    if percentile is not None:
        quantile = (arguments_df.Premise.str.len() + \
            arguments_df.Conclusion.str.len() + 1).quantile(
                percentile, interpolation='nearest')
    else:
        quantile = None

    # Plot the figure
    plt.figure(figsize=(15, 5))
    # Define the first subplot
    plt.subplot(1, 2, 1)
    # Plot the histogram of sentences length
    plt.plot(bin_edges[1:], hist)
    # Plot the quantile if defined
    if quantile is not None:
        plt.axvline(x = quantile, color = 'r', linestyle='--',
                    label=f'Quantile at {percentile * 100}')
    # Define title, labels and plot the legend
    plt.title(f'Distribution of the sentence length across the {df_name}')
    plt.xlabel('Sentence length')
    plt.legend()

    # Define the second subplot
    plt.subplot(1, 2, 2)
    # Plot the cumulative distribution of the sentences length
    plt.plot(bin_edges[1:], C)
    # Plot the quantile if defined
    if quantile is not None:
        plt.axvline(x = quantile, color = 'r', linestyle='--',
                    label=f'Quantile at {percentile * 100}')
    # Define title, labels and plot the legend
    plt.title('Comulative distribution of the sentence length across the ' + \
        df_name)
    plt.xlabel('Sentence length')
    plt.legend()

    plt.show()

def plot_word_cloud(arguments_df: pd.DataFrame,
                    title: str = 'Wordcloud') -> None:
    """Plot a word cloud on the corpus of the given dataframe

    Parameters
    ----------
    arguments_df : DataFrame
        A pandas arguments dataframe.
    title : str, optional
        The title of the plot, by default 'Wordcloud'
    """
    # Get total corpus
    total_corpus = arguments_df.Conclusion.tolist() + \
        arguments_df.Premise.tolist()

    # Create word cloud
    word_cloud = WordCloud(
        width=3000, height=2000, collocations=False,
        stopwords=STOPWORDS).generate(' '.join(total_corpus))

    # Plot the word cloud
    plt.figure(figsize=(15, 10))
    plt.imshow(word_cloud)
    plt.suptitle(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_most_common_words_distribution(
    arguments_df: pd.DataFrame, n: int = 20,
    title: str = 'Distribution of the most common words') -> None:
    """Plot the distribution of the most common n words

    Parameters
    ----------
    arguments_df : DataFrame
        A pandas arguments dataframe.
    n : int, optional
        Number of top frequent words to plot, by default 20.
    title : str, optional
        The title of the plot, by default 'Distribution of the most
        common words'
    """

    # Get corpus in lowercase
    corpus = ' '.join(arguments_df.Conclusion.tolist() + \
        arguments_df.Premise.tolist()).lower()
    # Remove punctuation
    corpus = ''.join([c for c in corpus if c not in string.punctuation])
    # Remove stopwords
    corpus_list = [c for c in corpus.split(' ')
                   if c not in STOPWORDS and c != '']

    # Get sorted dictionary of word frequencies
    counter = OrderedDict(Counter(corpus_list).most_common(n=n))

    # Plot the results
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(counter)), list(counter.values()), align='center')
    plt.xticks(range(len(counter)), list(counter.keys()))

    # set title and axis labels
    plt.suptitle(title)
    plt.xlabel('words')
    plt.ylabel('count')

    plt.tight_layout()

    plt.show()
