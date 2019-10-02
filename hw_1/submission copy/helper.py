from scipy.io import loadmat
from random import sample
import time
import numpy as np
import sys

def load_ocr():
    return loadmat('data/ocr.mat')

def calculate_distances_matrix(test_matrix, train_matrix):
    sum_squared_of_test_matrix = np.sum(np.square(test_matrix), axis=1)
    sum_squared_of_train_matrix = np.sum(np.square(train_matrix), axis=1)
    matrix_dot_product = np.dot(test_matrix, train_matrix.T)

    return np.sqrt(sum_squared_of_test_matrix[:, None] + sum_squared_of_train_matrix - 2 * matrix_dot_product)

# By default, Mac OS X can't directly render matplotlib
# To use matplotlib, please use Jupyter Notebook or set the backend properly (https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)
# import matplotlib.pyplot as plt

# def plot_learning_curve(title, train_sizes, mean_errors, std_errors):
#     plt.figure()
#     plt.title(title)
#     plt.xlabel("Training examples")
#     plt.ylabel("Average test error rates")
#     plt.grid()
#     plt.errorbar(train_sizes, mean_errors, yerr=std_errors, color="g", ecolor="r")
#
#     return plt
