#!/usr/bin/env python

# from __future__ import print_function
from scipy.io import loadmat
from random import sample
import numpy as np
import time
import sys

# def compute_distances_one_loop(self, X):
#     dists = np.zeros((num_test, num_train))
#     for i in xrange(num_test):
#         dists[i, :] = np.sum((self.X_train - X[i, :])**2, axis=1)
#     return dists

# Think of this as (x - y)^2  = x^2 + y^2 - 2xy.
def compute_distances_no_loops(x, y):
    # dist = np.sum(x) + np.sum(y) - 2 * np.dot(x, y.T)
    # dist = np.sum(np.square(x)) + np.sum(np.square(y)) - 2 * np.dot(x, y.T)
    # dist = np.linalg.norm(x, ord=1) + np.linalg.norm(y, ord=1) - 2 * np.dot(x, y.T)
    # dist = np.linalg.norm(x, ord=2) + np.linalg.norm(y, ord=2) - 2 * np.dot(x, y.T)
    dist = np.dot(x, x.T) + np.dot(y, y.T) - 2 * np.dot(x, y.T)
    return dist

def nn(X,Y,test):
    print(X.shape)
    print(Y.shape)
    print(test.shape)
    start = time.time()

    preds = []

    for vector in test[0:int(sys.argv[1])]:
        # Construct a matrix of dimension X in which each of its row contains test vector
        # print('0')
        test_matrix = np.tile(vector,(len(X),1))
        # print('1')

        # Substract matrix X with test_matrix, store it in dist matrix
        dist = (X - test_matrix)**2
        # print('2')
        dist = np.sum(dist, axis=1)
        # print('3')
        # dist = np.sqrt(dist)
        # print('4')
        min_index = np.argmin(dist)
        # print('5')
        preds.append(Y[min_index])

    print(time.time() - start)
    return preds

    # print(dist)
    # print(time.time()-start)

def nn_2(X,Y,test):
    print(X.shape)
    print(Y.shape)
    print(test.shape)
    start = time.time()

    preds = []

    for vector in test[0:int(sys.argv[1])]:
        # Construct a matrix of dimension X in which each of its row contains test vector
        # print('0')
        test_matrix = np.tile(vector,(len(X),1))
        # print('1')

        # Substract matrix X with test_matrix, store it in dist matrix
        dist = (X - test_matrix)**2
        # print('2')
        dist = np.sum(dist, axis=1)
        # print('3')
        dist = np.sqrt(dist)
        # print('4')
        min_index = np.argmin(dist)
        # print('5')
        preds.append(Y[min_index])

    print(time.time() - start)
    return preds

    # print(dist)
    # print(time.time()-start)

# Var definition
# ocr['data'] -> feature vectors, len 60k
# ocr['label'] -> label, len 60k

# ocr['testdata'] -> feature vectors, len 10k
# ocr['testlabel'] -> label, len 10k

# ocr = loadmat('ocr.mat')
# import matplotlib.pyplot as plt
# from matplotlib import cm
# plt.imshow(ocr['data'][0].reshape((28,28)), cmap=cm.gray_r)
# plt.show()

# nn([[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]], [0,0,0])
if __name__ == '__main__':
    print(compute_distances_no_loops(np.array([0, 0, 1]), np.array([1, 1, 0])))
    sys.exit()
    ocr = loadmat('ocr.mat')
    # num_trials = 10
    # for n in [ 1000, 2000, 4000, 8000 ]:
    num_trials = 1
    for n in [ 1000 ]:
        test_err = np.zeros(num_trials)
        for trial in range(num_trials):
            sel = sample(range(len(ocr['data'].astype('float'))),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            # test_err[trial] = np.mean(preds != ocr['testlabels'])
        # print("%d\t%g\t%g" % (n,np.mean(test_err),np.std(test_err)))

#plot -> x axis -> n
#y axis -> average test error out of 10 trials
#