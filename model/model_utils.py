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
    frequent_words = {word for word,
                      count in word_counts.items() if count > 2}

    # Create the vocabulary including the token <UNK>
    vocabulary = list(frequent_words) + ['<UNK>']

    # Initialize the matrix with zeros
    matrix = np.zeros((len(vocabulary), len(sentences)))

    # print(matrix)

    # Populate the matrix
    for idx, sentence in enumerate(sentences):
        for word in sentence.split():
            # If word is found, we add by 1
            if word in vocabulary:
                matrix[vocabulary.index(word), idx] += 1
            # else, we identify that an unknown word was found
            else:
                matrix[vocabulary.index('<UNK>'), idx] += 1

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
    unique_labels = list(set(unique_labels))

    # Initialize the matrix with zeros
    matrix = np.zeros((len(unique_labels), len(sentences)))

    # Populate the matrix
    for idx, (sentence, label) in enumerate(data):
        matrix[unique_labels.index(label), idx] += 1

    #multiple_ones_in_column = np.sum(matrix, axis=0) > 1
    #print(multiple_ones_in_column)

    return matrix
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    softmax = exp_z / sum_exp_z

    return softmax
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE

    relu = np.maximum(0, z)

    return relu
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE

    relu_prime = np.where(z >= 0 , 1, 0)

    return relu_prime
    #########################################################################
