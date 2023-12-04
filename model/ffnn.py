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

        Args:
        - input_size (int): Number of input features.
        - hidden_size (int): Number of neurons in the hidden layer.
        - num_classes (int): Number of output classes.
        - seed (int): Seed for random number generation (for reproducibility).
        """
        ############################# STUDENT SOLUTION ####################
        # Set a seed for reproducibility
        np.random.seed(seed)

        # Initialize weight matrices and biases with uniform distribution in the range (-1, 1)
        # Reference: Figure 7.10
        self.W = np.random.uniform(-1, +1, size=(hidden_size, input_size+1))

        # Add bias as element x0 = 1
        self.W[:, 0] = 1

        # Create weight matrix U to store results (see Figure 7.10)
        self.U = np.random.uniform(-1, +1, size=(num_classes, hidden_size))
        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Perform a forward pass with X as the input matrix, returning the model prediction Y_hat.

        Args:
        - X (np.ndarray): Input matrix.

        Returns:
        - np.ndarray: Model prediction.
        """
        ######################### STUDENT SOLUTION #########################
        # Add bias term to input
        a0 = np.insert(X, 0, 1, axis=0)

        # Forward pass through the hidden layer with ReLU activation (see Equation 7.13)
        z_hidden = np.dot(self.W, a0)
        a_hidden = relu(z_hidden)

        # Forward pass through the output layer with softmax activation
        z_output = np.dot(self.U, a_hidden)
        Y_hat = softmax(z_output)

        return Y_hat
        #####################################################################

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix using the `self.forward()` function.

        Args:
        - X (np.ndarray): Input matrix.

        Returns:
        - np.ndarray: Prediction matrix.
        """
        ######################### STUDENT SOLUTION ###########################
        # Perform a forward pass to get predictions
        Y_hat = self.forward(X)

        # Initialize a matrix for predictions with zeros
        Y_predict = np.zeros_like(Y_hat)

        # Convert predictions to a matrix of zeros and ones
        for col_idx in range(Y_hat.shape[1]):
            # Find the index of the maximum value in each column
            max_index = np.argmax(Y_hat[:, col_idx])
            # Set the corresponding entry to 1
            Y_predict[max_index, col_idx] = 1

        return Y_predict
        ######################################################################

    def backward(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm to compute gradients for weight matrices and biases.

        Args:
        - X (npt.ArrayLike): Input data for a single example.
        - Y (npt.ArrayLike): Ground truth labels for a single example.

        Returns:
        Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
            A tuple containing the gradients for weight matrices and biases:
            - grad_U (npt.ArrayLike): Gradient for the output layer weights.
            - grad_W (npt.ArrayLike): Gradient for the hidden layer weights.
            - delta_output (npt.ArrayLike): Error term for the output layer.
            - delta_hidden (npt.ArrayLike): Error term for the hidden layer.
        """
        ########################## STUDENT SOLUTION ###########################
        # Add bias term to input
        a0 = np.insert(X, 0, 1, axis=0)

        # Forward pass through the hidden layer with ReLU activation (see Equation 7.13)
        z_hidden = np.dot(self.W, a0)
        a_hidden = relu(z_hidden)

        # Forward pass through the output layer with softmax activation
        z_output = np.dot(self.U, a_hidden)
        Y_hat = softmax(z_output)

        # Compute the cross-entropy loss gradient at the output layer
        delta_output = Y_hat - Y

        # Compute gradients for the output layer
        grad_U = np.dot(delta_output, Y_hat.T)

        # Compute delta for the hidden layer using the chain rule
        delta_hidden = np.dot(self.U.T, delta_output) * relu_prime(a_hidden)

        # Compute gradients for the hidden layer
        grad_W = np.dot(delta_hidden.reshape(-1, 1), a0.reshape(1, -1))
    
        return grad_U, grad_W, delta_output, delta_hidden
        #######################################################################


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross-entropy loss.

    Args:
    - pred (npt.ArrayLike): Model predictions.
    - truth (npt.ArrayLike): Ground truth labels.

    Returns:
    - float: Cross-entropy loss.
    """
    ########################## STUDENT SOLUTION ###########################
    # Ensure there are no zeros in pred to avoid log(0) issues
    pred = np.maximum(pred, 1e-15)
    
    # Calculate the negative log likelihood loss using equation 7.26
    loss = -np.sum(truth * np.log(pred))

    return loss
    #######################################################################
