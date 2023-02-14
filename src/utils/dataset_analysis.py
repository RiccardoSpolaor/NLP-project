from collections import Counter, OrderedDict
import matplotlib.pyplot as plt 
import pandas as pd
import string
from wordcloud import WordCloud, STOPWORDS

from typing import Optional

def plot_stance_distribution(arguments_df: pd.DataFrame, title: str = 'Stance distribution') -> None:
    _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(15, 5))
    arguments_df.Stance.value_counts().plot(ax=ax, kind='bar', width=.25)

    # set title and axis labels
    plt.suptitle(title)

    ax.set_xlabel('stance')
    ax.set_ylabel('count')

    plt.tight_layout()
    
    # Show just the y grid
    #plt.grid(axis='y')

    plt.show()
    
def plot_sentiment_distribution(labels_df: pd.DataFrame, title: str = 'Sentiment values distribution') -> None:
    _, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(15, 6))
    labels_df.sum().plot(ax=ax, kind='bar')


    # Show just the x grid
    #ax.grid(axis='x')

    # set title and axis labels
    plt.suptitle(title)

    ax.set_xlabel('sentiment values')
    ax.set_ylabel('count')

    plt.tight_layout()
    
    #plt.grid(axis='y')

    plt.show()
    
import numpy as np

def plot_sequence_length_analysis(arguments_df: pd.DataFrame, df_name: str = 'dataset',
                                  percentile: Optional[float] = None) -> None:
    """Analyse the length of the premises + stances + conclusions
    Parameters
    ----------
    df : DataFrame
        A pandas dataframe.
    df_name : str, optional
        The name of the dataframe, by default 'dataset'.
    percentile : float, optional
        The percentile to plot, by default None. If None it is not plotted.
    """
    # Length of each training sentence
    train_sentences_lenghts = arguments_df.Premise.str.len() + arguments_df.Conclusion.str.len() + 1

    # Histogram of the sentences length distribution
    hist, bin_edges = np.histogram(train_sentences_lenghts, bins=np.max(train_sentences_lenghts) + 1, density=True) 
    # Cumulative distribution of the sentences length
    C = np.cumsum(hist)*(bin_edges[1] - bin_edges[0])
    
    if percentile is not None:
        quantile = (arguments_df.Premise.str.len() + arguments_df.Conclusion.str.len() + 1).quantile(
            percentile, interpolation='nearest')
    else:
        quantile = None


    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(bin_edges[1:], hist)
    if quantile is not None:
        plt.axvline(x = quantile, color = 'r', linestyle='--', label=f'Quantile at {percentile * 100}')
    plt.title(f'Distribution of the sentence length across the {df_name}')
    plt.legend()
    plt.xlabel('Sentence length')
    #plt.grid(axis='y') 
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[1:], C)
    if quantile is not None:
        plt.axvline(x = quantile, color = 'r', linestyle='--', label=f'Quantile at {percentile * 100}')
    plt.title(f'Comulative distribution of the sentence length across the {df_name}')
    plt.legend()
    plt.xlabel('Sentence length')
    #plt.grid(axis='y') 
    plt.show()
    
def plot_word_cloud(arguments_df: pd.DataFrame, title: str = 'Wordcloud') -> None:
    total_corpus = arguments_df.Conclusion.tolist() + arguments_df.Premise.tolist()

    word_cloud = WordCloud(width=3000, height=2000, collocations=False,
                           stopwords=STOPWORDS).generate(' '.join(total_corpus))

    plt.figure(figsize=(15, 10))
    plt.imshow(word_cloud)
    plt.suptitle(title) 
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_most_common_words_distribution(arguments_df: pd.DataFrame, n: int = 20,
                                        title: str = 'Distribution of the most common words') -> None:

    corpus = ' '.join(arguments_df.Conclusion.tolist() + arguments_df.Premise.tolist()).lower()
    corpus = ''.join([c for c in corpus if c not in string.punctuation])
    corpus_list = [c for c in corpus.split(' ') if c not in STOPWORDS and c != '']

    counter = OrderedDict(Counter(corpus_list).most_common(n=n))
    
    plt.figure(figsize=(15, 5))
    # Show just the x grid
    plt.bar(range(len(counter)), list(counter.values()), align='center')
    plt.xticks(range(len(counter)), list(counter.keys()))

    # set title and axis labels
    plt.suptitle(title)
    plt.xlabel('words')
    plt.ylabel('count')

    #plt.grid(axis='y')    
    plt.tight_layout()

    plt.show()
