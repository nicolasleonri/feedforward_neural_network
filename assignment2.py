import argparse
import numpy as np

from utils import load_dataset
from model.model_utils import bag_of_words_matrix, labels_matrix
from model.ffnn import NeuralNetwork
from helper import batch_train, minibatch_train

DATA_PATH = './data/dataset.csv'


def main():
    parser = argparse.ArgumentParser(
        description='Train feedforward neural network'
    )

    parser.add_argument(
        '--minibatch', dest='minibatch',
        help='Train feedforward neural network with mini-batch gradient descent/SGD',
        action='store_true'
    )

    parser.add_argument(
        '--train', dest='train',
        help='Turn on this flag when you are ready to train the model with backpropagation.',
        action='store_true'
    )

    args = parser.parse_args()

    sentences, intent, unique_intent = load_dataset(DATA_PATH)

    ############################ STUDENT SOLUTION ####################
    X = bag_of_words_matrix(sentences)
    Y = labels_matrix(tuple(zip(sentences, intent)))

    model = NeuralNetwork(X.shape[0], 150, len(unique_intent), 42)

    '''
    DISCUSSIONS TO EACH TASK:

    (1) In typical machine learning scenarios, the dataset is divided into three parts: 
    a training set used to develop the model, a testing set used to test during development
    of the model and another separate and unseen testing set used to evaluate the model's 
    performance on new data. As we are not splitting the dataset into one distinct training 
    set and two distinct training sets, we are essentially evaluating the model's ability 
    to memorize the dataset rather than generalize from it. 

    This approach can lead to overfitting, where the model performs well on the training 
    data but fails to generalize to new, unseen data. In other words, the model may 
    memorize the specific examples in the dataset without learning the underlying patterns 
    that would allow it to make accurate predictions on new, similar data. The risk of 
    overfitting is higher when using the same data for training and testing because the 
    model may simply memorize the training examples rather than learning the underlying 
    relationships.

    (4.2) See file "Plots.ipynb" for discussion.

    (4.4) See file "Figure_batch_train().png". As we can notice, there is decrease over 
    time of the cost function, indicating that the model is learning and minimizing the 
    difference between predicted and actual values. However, by the epoch 1000 we appear 
    to be on the way of stabilizing and reaching a plateau, which may suggest that the 
    model has converged, and further training may not lead to significant improvements
    in the accuracy. This convergence will be more accentuated in our minibatch() training.

    As expected, we got significant improvement in the accuracy of the classifier after 
    training. Our accuracy before training was ~0.176 and after training it was ~0.708.
    However we haven't tested on unseen data, so we cannot make any claims on overffiting.

    Following steps would include testing the model on unseen data, changing the values
    of learning rate and/or epochs, and even trying different optimization algorithms. 
    Also, we haven't calculated a F1-score. In order to improve the model, we should
    implement these aspects.

    (4.5) See file "Figure_minibatch().png". In mini-batch gradient descent, the dataset 
    is divided into smaller batches (in this case, a batch size of 64). The model 
    parameters are updated based on the average gradient computed from each batch. In SGD 
    (batch size of 1), the model parameters are updated after processing each individual 
    training example.

    In the SGD we get a smoother convergence. The averaging effect of the gradients from 
    multiple examples in each batch helps reduce the noise inherent in stochastic updates. 
    We can also notice a faster convergence. Its cost function plot is also less noisier 
    compared to the mini-batch gradient descent. However, in the mini-batch gradient descent
    we get a better accuracy (~0.715) than the SGD (~0.713).

    We should also consider how much computational time is needed in each case, in order to
    determine the more ecologically and economically efficient model. For example, both
    SGD and mini-batch models are more efficient than the regular training model (see point 
    4.2) as they converge faster. This implies that less resources are needed to train the
    model and get to the desired plateau. As SGD converges faster than the mini-batch example,
    it should be the most economically and ecogolically effective model.
    '''

    ##################################################################

    if not args.minibatch:
        print("Training FFNN using batch gradient descent...")
        batch_train(X, Y, model, train_flag=args.train)
    else:
        print("Training FFNN using mini-batch gradient descent...")
        minibatch_train(X, Y, model, train_flag=args.train)


if __name__ == "__main__":
    main()
