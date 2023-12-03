from model.ffnn import NeuralNetwork, compute_loss
import numpy as np


def batch_train(X, Y, model, train_flag=False):
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

    # Use your neural network to predict the intent

    def calculate_accuracy(Y_true, Y_predicted):
        """
        Calculate accuracy given true labels and predicted labels.
        """
        correct_predictions = np.sum(np.all(Y_true == Y_predicted, axis=0))
        total_samples = Y_true.shape[1]
        accuracy = correct_predictions / total_samples
        return accuracy

    Y_predicted_before_training = model.predict(X)
    accuracy_before_training = calculate_accuracy(Y, Y_predicted_before_training)

    print(compute_loss(Y_predicted_before_training, Y))

    if train_flag:
        print("Test")
        pass
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
