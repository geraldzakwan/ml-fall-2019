import time
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

# This is going to be the pi_y, i.e. label probability
def calculate_label_count_and_probability(labels):
    # Count label occurrence, store in 1D array
    # Set array size equals to the number of unique labels
    label_count = np.zeros(len(np.unique(labels)))
    for label in labels:
        # The important thing is to set index to (label - 1)
        # as index starts from zero but label starts from 1
        label_count[label - 1] = label_count[label - 1] + 1

    # Calculate label probability, store in 1D array
    label_prob = np.zeros(len(label_count))
    for label in labels:
        # Again, index is label - 1
        label_prob[label - 1] = label_count[label - 1] / len(labels)

    return label_count, label_prob

# This is going to be the miu_y_j
# The idea is to calculate separately for each value of y, i.e.
# getting train_data only for a particular value of y
# Then, sum over the word index and divide by the number of data
def calculate_word_given_label_prob(train_data, train_labels, label_count, word_list, is_sorted, laplace_smoothing=True):
    # The return array of shape (20, 61188)
    # i.e. (number of labels, word vocab size)
    word_prob = np.zeros((len(label_count), len(word_list)))

    # Iterate over label
    for i in range(0, len(label_count)):
        # For each label, find all indexes in train_labels that correspond
        indexes, = np.where(train_labels == i + 1) # Again, label is index plus one

        # Get the corresponding train_data for that label
        if is_sorted:
            # If the train_labels is sorted, we can just find two indexes
            # where a particular label starts and ends
            # Then, use that two indexes to slice the train data
            # This is the case for 'news.mat'
            first_idx = indexes[0]
            last_idx = indexes[-1]

            corr_train_data = train_data[first_idx:last_idx]
        else:
            # If the train_labels is not sorted, we need to use np.take
            # to take all train_data that correspond to the label
            # This is the case for 'news_binary.mat' and this is rather slow
            corr_train_data = np.take(train_data, indexes, axis=0)

        # Sum over axis=0, i.e. sum the word occurrence
        word_sum = np.sum(corr_train_data, axis=0)

        if laplace_smoothing:
            # Laplace smoothing, add each sum by 1
            word_sum = np.add(word_sum, 1)

        # Finally, calculate the word prob for this particular label
        if laplace_smoothing:
            # Divide by label_count + 2 (Laplace smoothing)
            word_prob_for_label = np.divide(word_sum, label_count[i] + 2)
        else:
            word_prob_for_label = np.divide(word_sum, label_count[i])

        # Assign result to the return array
        word_prob[i] = word_prob_for_label

    return word_prob

# Return True if array of probabilities sums up (closely) to 1
def check_sum_probability(array, epsilon=0.000001):
    return abs(np.sum(array) - 1.0) < epsilon

# Input: Feature vectors, 2D array of shape (n, d),
# where n is the number of data supplied and d is the dimension: 61188
# Return 1D array of predicted labels (size n)
def predict(feature_vectors, label_prob, word_prob):
    # Create the term 1 - x_j
    # Shape: (n, 61188)
    one_minus_feature_vectors = np.add(np.multiply(feature_vectors, -1), 1)

    # Create the term 1 - miu_y_j
    # Shape: (20, 61188)
    one_minus_word_prob = np.add(np.multiply(word_prob, -1), 1)

    # Log values of all the components
    pi_log_prob = np.log(label_prob) # Shape: (20,)
    log_word_prob = np.log(word_prob) # Shape: (20, 61188)
    log_one_minus_word_prob = np.log(one_minus_word_prob) # Shape: (20, 61188)

    # Dot product of x_j and ln(miu_y_j)
    # Shape: (n, 20)
    dot_product_one = np.dot(feature_vectors, log_word_prob.transpose())

    # Dot product of (1 - x_j) and ln(1 - miu_y_j)
    # Shape: (n, 20)
    dot_product_two = np.dot(one_minus_feature_vectors, log_one_minus_word_prob.transpose())

    # pi_log_prob will be broadcasted automatically,
    # i.e. from shape (20,) to (n, 20)
    # Shape: (n, 20)
    final_log_probs = dot_product_one + dot_product_two + pi_log_prob

    # Finally, return the label that has maximum logprob in each row
    # This is done by using argmax on axis=1
    # Add the argmax index result by 1 to obtain the correct label
    return np.add(np.argmax(final_log_probs, axis=1), 1) # Shape: (n,)

def compute_error_rate(pred_result, labels):
    # We compute the error rate here, so wrong prediction will yield 1
    # and correct prediction will yield 0
    pred_verdict = [1 if pred_result[i] != labels[i] else 0 for i in range(0, len(labels))]

    # Sum the wrong predictions and divide it by total test data
    return np.sum(pred_verdict) / len(pred_verdict)

# For binary experiment only
def calculate_bias(label_prob, word_prob):
    log_label_prob = np.log(label_prob)
    label_prob_diff = log_label_prob[1] - log_label_prob[0]

    one_minus_word_prob = np.add(np.multiply(word_prob, -1), 1)
    log_one_minus_word_prob = np.log(one_minus_word_prob)
    word_prob_diff = np.sum(log_one_minus_word_prob[1]) - np.sum(log_one_minus_word_prob[0])

    return label_prob_diff + word_prob_diff

# For binary experiment only
def calculate_weights(word_prob):
    one_minus_word_prob = np.add(np.multiply(word_prob, -1), 1)

    log_word_prob = np.log(word_prob)
    log_one_minus_word_prob = np.log(one_minus_word_prob)

    return (log_word_prob[1] - log_word_prob[0]) - (log_one_minus_word_prob[1] - log_one_minus_word_prob[0])

# For binary experiment only
def predict_affine(feature_vectors, bias, weights):
    wx = np.dot(feature_vectors, weights)
    b = bias
    y = wx + b

    return [2 if pred > 0 else 1 for pred in y]

# For binary experiment only
def get_top_and_bottom_words(weights, word_list, k=20):
    minus_k = (-1) * k

    k_biggest_indexes = np.argpartition(weights, minus_k)[:minus_k]
    k_smallest_indexes = np.argpartition(weights, k)[:k]

    top_weights = np.take(weights, k_biggest_indexes)
    bottom_weights = np.take(weights, k_smallest_indexes)

    word_list = np.array(word_list)

    top_words = np.take(word_list, k_biggest_indexes)
    bottom_words = np.take(word_list, k_smallest_indexes)

    return top_words, top_weights, bottom_words, bottom_weights

# Experiment for 20 labels
def experiment_3a():
    start = time.time()

    print('Experiment 3a: ')
    print()

    # Get the data
    train_data, train_labels, test_data, test_labels = load_news_data('news.mat')

    # Create word vocab list
    word_list = create_dictionary()

    # Calculate pi_y, i.e. the label probability
    label_count, label_prob = calculate_label_count_and_probability(train_labels)

    # Sanity check (probabilities sum up to 1)
    assert check_sum_probability(label_prob)

    # Calculate miu_y_j, i.e. the word probability
    word_prob = calculate_word_given_label_prob(train_data, train_labels, label_count, word_list, True)
    print('Done calculating word prob')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    # Prediction result on train data
    train_pred_result = predict(train_data, label_prob, word_prob)

    # Calculate train_error_rate
    train_error_rate = compute_error_rate(train_pred_result, train_labels)
    print('Train error rate: ' + str(train_error_rate))
    print('Done calculating train error rate')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    # Prediction result on test data
    test_pred_result = predict(test_data, label_prob, word_prob)

    # Calculate test_error_rate
    test_error_rate = compute_error_rate(test_pred_result, test_labels)
    print('Test error rate: ' + str(test_error_rate))
    print('Done calculating test error rate')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    print('----------------------------------------')
    print()

# Experiment for binary labels
def experiment_3b():
    start = time.time()

    print('Experiment 3b: ')
    print()

    # Get the data
    train_data, train_labels, test_data, test_labels = load_news_data('news_binary.mat')

    # Modify label from (-1, 1) to (1, 2) so it's easier to process
    # as it follows the previous convention that I use: label = index + 1
    train_labels = np.where(train_labels == -1, 1, 2)
    test_labels = np.where(test_labels == -1, 1, 2)

    # Create word vocab list
    word_list = create_dictionary()

    # Calculate pi_y, i.e. the label probability
    label_count, label_prob = calculate_label_count_and_probability(train_labels)

    # Sanity check (probabilities sum up to 1)
    assert check_sum_probability(label_prob)

    # Calculate miu_y_j, i.e. the word probability
    # The difference from experiment 3a is than the train_labels is not sorted
    word_prob = calculate_word_given_label_prob(train_data, train_labels, label_count, word_list, False)
    print('Done calculating word prob')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    # Prediction result on train data
    train_pred_result = predict(train_data, label_prob, word_prob)

    # Calculate train_error_rate
    train_error_rate = compute_error_rate(train_pred_result, train_labels)
    print('Train error rate: ' + str(train_error_rate))
    print('Done calculating train error rate')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    # Prediction result on test data
    test_pred_result = predict(test_data, label_prob, word_prob)

    # Calculate test_error_rate
    test_error_rate = compute_error_rate(test_pred_result, test_labels)
    print('Test error rate: ' + str(test_error_rate))
    print('Done calculating test error rate')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    print('----------------------------------------')
    print()

# Experiment for binary labels with affine function
def experiment_3c():
    start = time.time()

    print('Experiment 3c: ')
    print()

    # Get the data
    train_data, train_labels, test_data, test_labels = load_news_data('news_binary.mat')

    # Modify label from (-1, 1) to (1, 2) so it's easier to process
    # as it follows the previous convention that I use: label = index + 1
    train_labels = np.where(train_labels == -1, 1, 2)
    test_labels = np.where(test_labels == -1, 1, 2)

    # Create word vocab list
    word_list = create_dictionary()

    # Calculate pi_y, i.e. the label probability
    label_count, label_prob = calculate_label_count_and_probability(train_labels)

    # Sanity check (probabilities sum up to 1)
    assert check_sum_probability(label_prob)

    # Calculate miu_y_j, i.e. the word probability
    # The difference from experiment 3a is than the train_labels is not sorted
    word_prob = calculate_word_given_label_prob(train_data, train_labels, label_count, word_list, False)
    print('Done calculating word prob')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    bias = calculate_bias(label_prob, word_prob)
    weights = calculate_weights(word_prob)
    print('Done calculating bias and weights')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    # Prediction result on train data
    train_pred_result = predict_affine(train_data, bias, weights)

    # Calculate train_error_rate
    train_error_rate = compute_error_rate(train_pred_result, train_labels)
    print('Train error rate: ' + str(train_error_rate))
    print('Done calculating train error rate')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    # Prediction result on test data
    test_pred_result = predict_affine(test_data, bias, weights)

    # Calculate test_error_rate
    test_error_rate = compute_error_rate(test_pred_result, test_labels)
    print('Test error rate: ' + str(test_error_rate))
    print('Done calculating test error rate')
    print('Elapsed time: ' + str(time.time() - start))
    print()

    top_words, top_weights, bottom_words, bottom_weights = get_top_and_bottom_words(weights, word_list)
    print('Top words:')
    print(top_words)
    print()
    print('Top weights:')
    print(top_weights)
    print()
    print('Bottom words:')
    print(bottom_words)
    print()
    print('Bottom weights:')
    print(bottom_weights)
    print()

    print('----------------------------------------')
    print()

if __name__ == '__main__':
    # experiment_3a()
    #
    # experiment_3b()

    experiment_3c()
