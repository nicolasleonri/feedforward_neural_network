import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # Flatten the list of sentences into a list of words
    words = [word for sentence in sentences for word in sentence.split()]

    # Count occurrences of each word
    word_counts = {word: words.count(word) for word in set(words)}

    # Identify words occurring at least 2 times
    frequent_words = {word for word, count in word_counts.items() if count >= 2}

    # Create the vocabulary including the token <UNK>
    vocabulary = list(frequent_words) + ['<UNK>']

    # Initialize the matrix with zeros
    matrix = np.zeros((len(sentences), len(vocabulary)))

    # Populate the matrix
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            # If word is found, we change the value to 1
            if word in vocabulary:
                matrix[i, vocabulary.index(word)] = 1
            # else, we identify that an unknown word was found
            else:
                matrix[i, vocabulary.index('<UNK>')] = 1

    return matrix
    #########################################################################


def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # Extract unique labels and sentences from the data tuple
    sentences, unique_labels = zip(*data)

    # Convert unique_labels to a list for indexing
    unique_labels = list(unique_labels)

    # Initialize the matrix with zeros
    matrix = np.zeros((len(sentences), len(unique_labels)))

    # Populate the matrix
    for idx, (sentence, label) in enumerate(data):
        matrix[idx, unique_labels.index(label)] = 1

    return matrix
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return None
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return None
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return None
    #########################################################################
