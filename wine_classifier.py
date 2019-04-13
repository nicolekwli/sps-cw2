#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission. 
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']    


def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of 
    # the function

    """
    ## Wait so this function should somehow come with two features as a return result?
    ## Or is this us displaying 13x13 then manually choosing, and then pass what we choose as an arg
    ## and then return the reduced data..?

    So from what I can tell we have to-
    * plot the pairwise combinations of features
    * manually choose two features
    * talk about why we chose them in the report + put pics of the plot
    * return the two features that are selected here (not sure if we return the string or the reduced matrix?)
    """

    ### attempt to display 13 x 13 things
    n_features = train_set.shape[1]
    #fig, ax = plt.subplots(n_features, n_features)
    fig, ax = plt.subplots(4, 5)

    colours = np.zeros_like(train_labels, dtype=np.object)
    colours[train_labels == 1] = CLASS_1_C
    colours[train_labels == 2] = CLASS_2_C
    colours[train_labels == 3] = CLASS_3_C
    
    # Gets features 1-4 vs 1-4 | A vs A
    """
    for row in range(0, n_features):
        for col in range(0, n_features):
            ax.scatter(train_set[:, row], train_set[:, col], s=10, c=colours)
            ax.set_title('Features {} vs {}'.format(row+1, col+1))
            print("here")
            # was trying to display each individual plot here
            plt.show()
    """

    # Gets features 1-4 vs 5-8 | A vs B
    for row in range(0, 4):
        for col in range(0, 5):
            ax[row][col].scatter(train_set[:, row], train_set[:, col+8], c=colours)
            ax[row][col].set_title('Features {} vs {}'.format(row+1, col+9))
    
    # Set A: 1-4
    #     B: 5-8
    #     C: 9-13
    # also A vs A, A vs B
    # Gets features A vs C
    # Gets features C vs C
    # Gets features C vs A
    # Gets features B vs A
    # Gets features B vs B
    # Gets features B vs C
    # Gets features C vs B 
    # so should be 9 diff plots
    # Unless we do a split like A 1-6 and B 7-13 ?

    plt.show()

    return []


def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path, 
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))