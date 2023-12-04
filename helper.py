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
    ################################# STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) Use your neural network to predict the intent
    #         (without any training) and calculate the accuracy 
    #         of the classifier. Should you be expecting high
    #         numbers yet?
    #         2) if train_flag is true, run the training for 1000 epochs using 
    #         learning rate = 0.005 and use this neural network to predict the 
    #         intent and calculate the accuracy of the classifier
    #         3) Then, plot the cost function for each iteration and
    #         compare the results after training with results before training

    def calculate_accuracy(Y_true, Y_predicted):
        """
        Calculate accuracy given true labels and predicted labels.
        """
        correct_predictions = np.sum(np.all(Y_true == Y_predicted, axis=0))
        total_samples = Y_true.shape[1]
        accuracy = correct_predictions / total_samples
        return accuracy

    # Use neural network to predict the intent (without any training)
    # and calculate the accuracy of the classifier
    Y_predicted_before_training = model.predict(X)

    print("Accuracy without training:", calculate_accuracy(Y, Y_predicted_before_training))
    print("Loss without training:", compute_loss(Y_predicted_before_training, Y))

    # if train_flag is true, run the training for 1000 epochs using 
    # learning rate = 0.005 
    if train_flag:
        epochs = 1000
        learning_rate = 0.005
        losses = []

        # Iterate through training epochs
        for epoch in range(epochs):
            print("Epoch:", epoch)
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
                grad_U_example, grad_W_example = model.backward(x_example, y_example)

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
        print(losses)

    # Plot the cost function for each iteration
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Progress')
    plt.show()
                
    Y_predicted_after_training = model.predict(X)

    print("Accuracy with training:", calculate_accuracy(Y, Y_predicted_after_training))
    print("Loss with training:", compute_loss(Y_predicted_after_training, Y))

    return None
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    pass
    if train_flag:
        pass
    #########################################################################
