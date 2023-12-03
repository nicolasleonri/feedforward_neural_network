import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple


class NeuralNetwork(object):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int, 
        num_classes: int,
        seed: int = 1
    ):
        """
        Initialize neural network's weights and biases.
        """
        ############################# STUDENT SOLUTION ####################
        # YOUR CODE HERE
        #     TODO:
        #         1) Set a seed so that your model is reproducible
        #         2) Initialize weight matrices and biases with uniform
        #         distribution in the range (-1, 1).
        np.random.seed(seed)

        # Create weight and bias matrix W (see Figure 7.10)
        self.W = np.random.uniform(-1, +1, size=(hidden_size, input_size+1))
        # Add bias as element x0 = 1
        self.W[:, 0] = 1

        # Create weight matrix U to store results (see Figure 7.10)
        self.U = np.random.uniform(-1, +1, size=(num_classes, hidden_size))

        pass
        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Forward pass with X as input matrix, returning the model prediction
        Y_hat.
        """
        ######################### STUDENT SOLUTION #########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform only a forward pass with X as input.

        # Add bias term to input
        a0 = np.insert(X, 0, 1, axis=0)

        # Forward pass through the hidden layer with ReLU activation
        z_hidden = np.dot(self.W, a0)
        a_hidden = relu(z_hidden)

        # Forward pass through the output layer with softmax activation
        z_output = np.dot(self.U, a_hidden)
        Y_hat = softmax(z_output)

        return Y_hat
        #####################################################################

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`
        """
        ######################### STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Create a prediction matrix of the intent data using
        #         `self.forward()` function. The shape of prediction matrix
        #         should be similar to label matrix produced with
        #         `labels_matrix()`

        # Perform a forward pass to get predictions
        Y_hat = self.forward(X)

        Y_predict = np.zeros_like(Y_hat)

        for col_idx in range(Y_hat.shape[1]):
            max_index = np.argmax(Y_hat[:, col_idx])
            Y_predict[max_index, col_idx] = 1
        
        #multiple_ones_in_column = np.sum(Y_predict, axis=0) > 1
        #print(multiple_ones_in_column)

        return Y_predict
        ######################################################################

    def backward(
        self, 
        X: npt.ArrayLike, 
        Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.
        """
        ########################## STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform forward pass, then backpropagation
        #         to get gradient for weight matrices and biases
        #         2) Return the gradient for weight matrices and biases
        pass
        #######################################################################


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.
    """
    ########################## STUDENT SOLUTION ###########################
    # YOUR CODE HERE
    #     TODO:
    #         1) Compute the cross entropy loss between your model prediction
    #         and the ground truth.

    # Get the number of samples
    num_samples = pred.shape[1]

    # Extract the correct class indices
    correct_class_indices = truth.argmax(axis=0)

    # Extract the raw scores (logits) for the correct class
    z_c = pred[correct_class_indices, np.arange(num_samples)]

    # Compute the cross-entropy loss using the specified formula
    loss = -np.sum(np.log(np.exp(z_c) / np.sum(np.exp(pred), axis=0))) / num_samples

    return loss
    #######################################################################