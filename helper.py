from model.ffnn import NeuralNetwork, compute_loss
import numpy as np
import matplotlib.pyplot as plt

def batch_train(X, Y, model, train_flag=False):
    """
    Train the neural network using batch gradient descent and plot the loss.

    Args:
    - X (npt.ArrayLike): Input data.
    - Y (npt.ArrayLike): Ground truth labels.
    - model (NeuralNetwork): Neural network model.
    - train_flag (bool): If True, perform training and plot the loss.

    Returns:
    - None
    """
    def calculate_accuracy(Y_true, Y_predicted):
        """
        Calculate accuracy given true labels and predicted labels.

        Args:
        - Y_true (npt.ArrayLike): Ground truth labels.
        - Y_predicted (npt.ArrayLike): Predicted labels.

        Returns:
        - float: Accuracy as a percentage.
        """
        correct_predictions = np.sum(np.all(Y_true == Y_predicted, axis=0))
        total_samples = Y_true.shape[1]
        accuracy = correct_predictions / total_samples
        return accuracy

    # Predict before training
    Y_predicted_before_training = model.predict(X)
    print("Accuracy without training:", calculate_accuracy(Y, Y_predicted_before_training))

    # Train the neural network if train_flag is True
    if train_flag:
        epochs = 1000
        learning_rate = 0.005
        losses = []

        # Iterate through training epochs
        for epoch in range(epochs):
            # Initialize gradient accumulators for each epoch
            grad_W_accumulator = np.zeros_like(model.W)
            grad_U_accumulator = 0
            loss_accumulator = 0

            # Iterate through the training dataset
            for example in range(X.shape[1]):
                # Get a single example
                x_example = X[:, example]
                y_example = Y[:, example]

                # Perform forward pass
                Y_hat_example = model.forward(x_example)

                # Compute cross-entropy loss for the example
                loss_example = compute_loss(Y_hat_example, y_example)
                # Accumulate the loss
                loss_accumulator += loss_example

                # Perform backward pass to compute gradients
                grad_U_example, grad_W_example,_ ,_ = model.backward(x_example, y_example)

                # Accumulate the gradients
                grad_W_accumulator += grad_W_example
                grad_U_accumulator += grad_U_example

            # Average the gradients and loss over the entire dataset
            avg_grad_W = grad_W_accumulator / X.shape[1]
            avg_grad_U = grad_U_accumulator / X.shape[1]
            avg_loss = loss_accumulator / X.shape[1]

            # Update weights and biases using the averaged gradients
            model.W -= learning_rate * avg_grad_W
            model.U -= learning_rate * avg_grad_U
        
            # Append the average loss to the list for plotting
            losses.append(avg_loss)

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss}")

        # Predict after training          
        Y_predicted_after_training = model.predict(X)
        print("Accuracy with training:", calculate_accuracy(Y, Y_predicted_after_training))

        # Plot the cost function for each iteration
        plt.plot(range(epochs), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Progress')
        plt.show()

    return None
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.

    def calculate_accuracy(Y_true, Y_predicted):
        """
        Calculate accuracy given true labels and predicted labels.

        Args:
        - Y_true (npt.ArrayLike): Ground truth labels.
        - Y_predicted (npt.ArrayLike): Predicted labels.

        Returns:
        - float: Accuracy as a percentage.
        """
        correct_predictions = np.sum(np.all(Y_true == Y_predicted, axis=0))
        total_samples = Y_true.shape[1]
        accuracy = correct_predictions / total_samples
        return accuracy

    # Predict before training
    Y_predicted_before_training = model.predict(X)
    print("Accuracy without training:", calculate_accuracy(Y, Y_predicted_before_training))

    if train_flag:
        epochs = 1000
        learning_rate = 0.005
        losses = []
        batch_size = 64

        for epoch in range(epochs):
            grad_W_accumulator = np.zeros_like(model.W)
            grad_U_accumulator = 0
            loss_accumulator = 0

            # Shuffle the data for each epoch
            permutation = np.random.permutation(X.shape[1])
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            # Iterate through mini-batches
            for batch_start in range(0, X.shape[1], batch_size):
                batch_end = batch_start + batch_size
                X_batch = X_shuffled[:, batch_start:batch_end]
                Y_batch = Y_shuffled[:, batch_start:batch_end]

                # Initialize gradient accumulators for the mini-batch
                grad_W_batch = np.zeros_like(model.W)
                grad_U_batch = 0
                loss_batch = 0

                # Iterate through examples in the mini-batch
                for example in range(X_batch.shape[1]):
                    x_example = X_batch[:, example]
                    y_example = Y_batch[:, example]

                    Y_hat_example = model.forward(x_example)

                    loss_example = compute_loss(Y_hat_example, y_example)
                    loss_batch += loss_example

                    grad_U_example, grad_W_example, _, _ = model.backward(x_example, y_example)

                    grad_W_batch += grad_W_example
                    grad_U_batch += grad_U_example

                # Average over the mini-batch
                avg_grad_W_batch = grad_W_batch / X_batch.shape[1]
                avg_grad_U_batch = grad_U_batch / X_batch.shape[1]
                avg_loss_batch = loss_batch / X_batch.shape[1]

                # Accumulate the gradients and loss for the entire epoch
                grad_W_accumulator += avg_grad_W_batch
                grad_U_accumulator += avg_grad_U_batch
                loss_accumulator += avg_loss_batch

            # Average the gradients and loss over the entire dataset
            avg_grad_W = grad_W_accumulator / (X.shape[1] // batch_size)
            avg_grad_U = grad_U_accumulator / (X.shape[1] // batch_size)
            avg_loss = loss_accumulator / (X.shape[1] // batch_size)

            # Update weights and biases using the averaged gradients
            model.W -= learning_rate * avg_grad_W
            model.U -= learning_rate * avg_grad_U

            # Append the average loss to the list for plotting
            losses.append(avg_loss)

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss}")

        # Predict after training          
        Y_predicted_after_training = model.predict(X)
        print("Accuracy with training:", calculate_accuracy(Y, Y_predicted_after_training))

        # Plot the cost function for each iteration
        plt.plot(range(epochs), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Progress')
        plt.show()
    
    return None

    #########################################################################
