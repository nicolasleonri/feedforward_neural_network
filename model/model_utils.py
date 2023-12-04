import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into a V x M matrix, where V is the vocabulary size and
    M is the number of sentences.

    Args:
    - sentences (List[str]): A list of sentences in the dataset.

    Returns:
    - np.ndarray: The bag-of-words matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # Flatten the list of sentences into a list of words
    words = [word for sentence in sentences for word in sentence.split()]

    # Count occurrences of each word
    word_counts = {word: words.count(word) for word in set(words)}

    # Identify words occurring at least 3 times
    frequent_words = {word for word,
                      count in word_counts.items() if count > 2}

    # Create the vocabulary including the token <UNK>
    vocabulary = list(frequent_words) + ['<UNK>']

    # Initialize the matrix with zeros
    matrix = np.zeros((len(vocabulary), len(sentences)))

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
    Convert the dataset into a K x M matrix, where K is the number of unique labels
    and M is the number of sentences.

    Args:
    - data (Tuple[List[str], Set[str]]): A tuple containing a list of sentences
      and a set of unique labels associated with each sentence.

    Returns:
    - np.ndarray: The labels matrix.
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

    return matrix
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Apply the softmax function to the input array.

    Args:
    - z (np.ndarray): Input array.

    Returns:
    - np.ndarray: Output array after applying the softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    exp_z = np.exp(z - np.max(z))  # Log-sum-exp trick for numerical stability

    # Calculate the sum of exponentials
    sum_exp_z = np.sum(exp_z)

    # Compute the softmax probabilities
    softmax_result = exp_z / sum_exp_z

    return softmax_result
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Apply the Rectified Linear Unit (ReLU) function to the input array.

    Args:
    - z (np.ndarray): Input array.

    Returns:
    - np.ndarray: Output array after applying the ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    relu_result = np.maximum(0, z)

    return relu_result
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Compute the first derivative of the Rectified Linear Unit (ReLU) function.

    Args:
    - z (np.ndarray): Input array.

    Returns:
    - np.ndarray: Output array representing the derivative of the ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    relu_prime_result = np.where(z >= 0, 1, 0)

    return relu_prime_result
    #########################################################################
