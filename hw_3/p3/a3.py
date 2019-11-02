import time
import math
import numpy as np

from collections import defaultdict
from scipy.io import loadmat

def load_news_data(filepath):
    news = loadmat(filepath)

    # From scipy csc matrix to 2D array
    train_data = news['data'].toarray()
    # From 2D array to 1D array
    train_labels = news['labels'].flatten()

    test_data = news['testdata'].toarray()
    test_labels = news['testlabels'].flatten()

    return train_data, train_labels, test_data, test_labels

def create_dictionary(filepath='news.vocab'):
    with open(filepath, 'r') as f:
        raw_list = f.readlines()

    list_of_words = []
    for elem in raw_list:
        list_of_words.append(elem.strip('\n'))

    return list_of_words

# Numerically stable sigmoid
def sigmoid(x):
    # Use 1 / (1 + np.exp(-x)) when x >= 0
    # Use np.exp(x) / (1 + np.exp(x) when x < 0 -> to avoid overflow
    # when x is a very small negative number
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

# This basically follows the equation provided in the problem
def calculate_negative_gradients(train_data, train_labels, weights):
    A = train_data
    b = train_labels
    w = weights

    # Multiply train_data matrix with weights vector
    # Use np.dot to produce a vector as the result
    Aw = np.dot(A, w)

    # Calculate the negation of b
    minus_b = np.multiply(b, -1)

    # IMPORTANT: Use multiply rather than dot product to obtain coordinate wise product
    minus_b_Aw = np.multiply(minus_b, Aw)

    # Take the sigmoid
    sigmoid_minus_b_Aw = sigmoid(minus_b_Aw)

    # Again, use multiply rather than dot product
    b_sigmoid_minus_b_Aw = np.multiply(b, sigmoid_minus_b_Aw)

    # Get the total gradient
    At_b_sigmoid_minus_b_Aw = np.dot(A.transpose(), b_sigmoid_minus_b_Aw)

    # Divide by n (length of train_data)
    return np.divide(At_b_sigmoid_minus_b_Aw, len(A))

# Default learning_rate is 1.0 as stated in the problem
def update_weights(weights, negative_gradients, learning_rate=1.0):
    # Simply add the weights with scaled negative_gradients
    return np.add(weights, np.multiply(negative_gradients, learning_rate))

# For binary experiment only
def predict(feature_vectors, weights):
    # Get the output of the linear classifier, i.e wx
    wx = np.dot(feature_vectors, weights)

    # Use sign operator to determine the label, i.e.
    # return 1 if wx is positive, return -1 otherwise
    return [1 if pred > 0 else -1 for pred in wx]

def compute_false_prediction(pred_result, labels):
    # We compute the number of false prediction here, so wrong prediction will yield 1
    # and correct prediction will yield 0
    pred_verdict = [1 if pred_result[i] != labels[i] else 0 for i in range(0, len(labels))]

    # Sum the wrong predictions and divide it by total test data
    return np.sum(pred_verdict)

def compute_error_rate(pred_result, labels):
    # We compute the error rate here, so wrong prediction will yield 1
    # and correct prediction will yield 0
    pred_verdict = [1 if pred_result[i] != labels[i] else 0 for i in range(0, len(labels))]

    # Sum the wrong predictions and divide it by total test data
    return np.sum(pred_verdict) / len(pred_verdict)

# For binary experiment only
def get_top_and_bottom_words(weights, word_list, k=20):
    minus_k = (-1) * k

    k_biggest_indexes = np.argpartition(weights, minus_k)[minus_k:]
    # k_biggest_indexes_slower = (-weights).argsort()[:k]

    # print(np.sort(k_biggest_indexes))
    # print(np.sort(k_biggest_indexes_slower))

    k_smallest_indexes = np.argpartition(weights, k)[:k]

    top_weights = np.take(weights, k_biggest_indexes)
    bottom_weights = np.take(weights, k_smallest_indexes)

    word_list = np.array(word_list)

    top_words = np.take(word_list, k_biggest_indexes)
    bottom_words = np.take(word_list, k_smallest_indexes)

    # Zip vocab_index, word and their weight
    zipped_top = [(k_biggest_indexes[i], top_words[i], top_weights[i]) for i in range(0, len(top_weights))]
    zipped_bottom = [(k_smallest_indexes[i], bottom_words[i], bottom_weights[i]) for i in range(0, len(bottom_weights))]

    # Return them in descending order of their absolute weight value
    zipped_top = sorted(zipped_top, key=lambda x: x[2], reverse=True)
    zipped_bottom = sorted(zipped_bottom, key=lambda x: x[2])

    return zipped_top, zipped_bottom

def run_experiment(verbose=False):
    start = time.time()

    # Get the data
    train_data, train_labels, test_data, test_labels = load_news_data('news_binary.mat')

    # Set w_j(t=0) = 0 for all j [0, 61188)
    weights = np.zeros(len(train_data[0]))

    t = 0 # Iteration/timestep
    while (True):
        # Calculate error_rate for the current timestep
        pred_result = predict(train_data, weights)
        false_preds = compute_false_prediction(pred_result, train_labels)
        error_rate = compute_error_rate(pred_result, train_labels)

        if verbose and t % 100 == 0:
            print('Iteration: ' + str(t))
            print('False preds: ' + str(false_preds) + ' out of ' + str(len(train_data)) + ' data')
            print('Error rate: ' + str(error_rate))
            print('Elapsed time: ' + str(time.time() - start) + ' seconds')
            print()

        # Terminate when there is no false prediction
        # I don't use error_rate == 0.0 as the condition in this case
        # to avoid floating value precision problem (if any)
        if false_preds == 0:
            print('Error rate reaches zero, terminate training')
            print('Total iterations: ' + str(t))
            print()
            break

        # Calculate the gradients for the current weights (w_t)
        negative_gradients = calculate_negative_gradients(train_data, train_labels, weights)

        # Calculate weights for the next timestep (w_{t+1})
        weights = update_weights(weights, negative_gradients)

        t = t + 1

    print('Done training')
    print('Elapsed time: ' + str(time.time() - start) + ' seconds')
    print()

    # Calculate test error rate
    test_pred_result = predict(test_data, weights)
    test_error_rate = compute_error_rate(test_pred_result, test_labels)

    print('Test error rate: ' + str(test_error_rate))
    print('Done calculating test error rate: ')
    print('Elapsed time: ' + str(time.time() - start) + ' seconds')
    print()

    # Create word vocab list
    word_list = create_dictionary()

    # Determine the most positive and negative words, 20 each
    zipped_top, zipped_bottom = get_top_and_bottom_words(weights, word_list)
    print('Most positive words, descendingly ordered based on the absolute weight value:')
    for tup in zipped_top:
        print(tup)
    print()

    print('Most negative words, descendingly ordered based on the absolute weight value:')
    for tup in zipped_bottom:
        print(tup)
    print()

    print('Done fetching the most positive and negative words')
    print('Elapsed time: ' + str(time.time() - start) + ' seconds')
    print()

if __name__ == '__main__':
    run_experiment(True)
