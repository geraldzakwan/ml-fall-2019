import sys
import time
import numpy as np
import pickle
from collections import defaultdict
from scipy.io import loadmat

def load_news_data(filepath):
    news = loadmat(filepath)

    train_data = news['data']
    train_labels = news['labels']

    test_data = news['testdata']
    test_labels = news['testlabels']

    return train_data, train_labels, test_data, test_labels

def create_dictionary(filepath):
    with open(filepath, 'r') as f:
        list_of_words = f.readlines()

    return list_of_words

# This is going to be the pi_y
def calculate_label_count_and_probability(labels):
    label_count = defaultdict(int)

    for label in labels:
        label_count[label[0]] = label_count[label[0]] + 1

    label_probability = {}

    for label in labels:
        label_probability[label[0]] = label_count[label[0]] / len(labels)

    return label_count, label_probability

def check_sum_probability(collection):
    sum_of_probs = 0

    if isinstance(collection, dict):
        for label in label_probability:
            sum_of_probs = sum_of_probs + label_probability[label]
    else:
        sum_of_probs = np.sum(collection)

    print(sum_of_probs)
    return abs(sum_of_probs - 1.0) < 0.001

# This is going to be miu_y_j
# Return array of shape(20, 61188)
# TO DO: Optimize this, now takes around 30 - 35 seconds
def calculate_word_given_label_probability(train_data, train_labels, word_list):
    start = time.time()

    label_count, label_probability = calculate_label_count_and_probability(train_labels)

    # print(word_sum.shape)
    word_sum = np.zeros((len(label_count), len(word_list)))

    # Get word_sum
    # Iterate over 11269 data
    for i in range(0, len(train_data)):
        word_vector = train_data[i]
        # IMPORTANT, train_labels is a 2D ndarray, take index 0
        label = train_labels[i][0]

        if i % 100 == 0:
            print(i)

        # Iterate over word vector of 61188 elements
        for j in range(0, len(word_vector)):
            # IMPORTANT, reduce label by 1, e.g. label 1 for index 0
            word_sum[label-1][j] = word_sum[label-1][j] + word_vector[j]

    # Finally, divide the word_sum by the total occurence of
    # its corresponding label
    word_prob = np.zeros((len(label_count), len(word_list)))

    for i in range(0, len(word_prob)):
        if i % 100 == 0:
            print(i)

        for j in range(0, len(word_prob[i])):
            # IMPORTANT, ADD index by 1, e.g. index 0 for label 1
            word_prob[i][j] = word_sum[i][j] / label_count[i + 1]

    print('Time elapsed: ')
    print(time.time() - start)

    return word_prob

# Check if for all y, miu_y_j sums up to 1
def check_word_probability(word_prob):
    for prob_list in word_prob:
        if not check_sum_probability(prob_list):
            return False

    return True

    # This is going to be the P(X|Y)
    def calculate_feature_vector_probability(word_prob):
        likelihood_arr = np.zeros()


if __name__ == '__main__':
    # with open('word_prob.pickle', 'rb') as handle:
    #     word_prob = pickle.load(handle)
    #
    # print(word_prob[0])
    # print(word_prob[1])
    # print(word_prob[2])

    # print(check_word_probability(word_prob))

    train_data, train_labels, test_data, test_labels = load_news_data('news.mat')

    print(len(test_labels))

    sys.exit()

    # Convert train and test data to np array
    train_data = train_data.todense()
    # print(train_data.shape)
    # (11269, 61188)

    word_list = create_dictionary('news.vocab')
    # print(len(word_dict))
    # 61188

    word_prob = calculate_word_given_label_probability(train_data, train_labels, word_list)
    print(word_prob.shape)

    with open('word_prob.pickle', 'wb') as handle:
        pickle.dump(word_prob, handle)

    label_count, label_probability = calculate_label_and_count_probability(train_labels)
    print(check_sum_probability(label_probability))

    # So, basically 61188 is the vocab size
    # It is a one hot encoding
    # 11269 is the total sentence

    # Convert test data to np array
    test_data = test_data.todense()

    print(test_data.shape)
    # (7505, 61188)
