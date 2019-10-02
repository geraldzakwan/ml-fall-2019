#!/usr/bin/env python

# from __future__ import print_function
from scipy.io import loadmat
from random import sample
import numpy as np
import time
import sys

def calculate_distances_matrix(test_vectors, train_vectors):
    sum_squared_of_test_vectors = np.sum(np.square(test_vectors), axis=1)
    sum_squared_of_train_vectors = np.sum(np.square(train_vectors), axis=1)
    matrix_multiplication = np.dot(test_vectors, train_vectors.T)

    return np.sqrt(sum_squared_of_test_vectors[:, np.newaxis] + sum_squared_of_train_vectors - 2 * matrix_multiplication)

def nn(X,Y,test):
    start = time.time()
    preds = []

    distances_matrix = calculate_distances_matrix(test, X)
    for row in distances_matrix:
        min_idx = np.argmin(row)
        preds.append(Y[min_idx])

    print(time.time()-start)
    return preds

if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    # num_trials = 10
    # for n in [ 1000, 2000, 4000, 8000 ]:
    num_trials = 1
    for n in [ 1000 ]:
        test_err = np.zeros(num_trials)
        for trial in range(num_trials):
            sel = sample(range(len(ocr['data'].astype('float'))),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            test_err[trial] = np.mean(preds != ocr['testlabels'])
        print("%d\t%g\t%g" % (n,np.mean(test_err),np.std(test_err)))