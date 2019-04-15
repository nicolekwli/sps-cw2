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

    """
    '''
    first 5 rows
    '''
    fig, ax = plt.subplots(5, 5)
    fig, ax1 = plt.subplots(5, 5)
    fig, ax2 = plt.subplots(5, 3)

    '''
    next 5 rows
    '''
    fig, ax3 = plt.subplots(5, 5)
    fig, ax4 = plt.subplots(5, 5)
    fig, ax5 = plt.subplots(5, 3)

    '''
    last 3 rows
    '''
    fig, ax6 = plt.subplots(3, 5)
    fig, ax7 = plt.subplots(3, 5)
    fig, ax8 = plt.subplots(3, 3)

    plt.subplots_adjust(left=0.1, right=0.99)

    colours = np.zeros_like(train_labels, dtype=np.object)
    colours[train_labels == 1] = CLASS_1_C
    colours[train_labels == 2] = CLASS_2_C
    colours[train_labels == 3] = CLASS_3_C
    
    '''
    This is 5 rows x 13 columns
    '''
    for row in range(0, 5):
        for col in range(0, 5):
            ax[row][col].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax[row][col].set_title('Features {} vs {}'.format(row+1, col+1))

    for row in range(0, 5):
        for col in range(5, 10):
            ax1[row][col-5].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax1[row][col-5].set_title('Features {} vs {}'.format(row+1, col+1))

    for row in range(0, 5):
        for col in range(10, 13):
            ax2[row][col-10].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax2[row][col-10].set_title('Features {} vs {}'.format(row+1, col+1))

    '''
    next 5 rows x 13 columns
    '''
    for row in range(5, 10):
        for col in range(0, 5):
            ax3[row-5][col].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax3[row-5][col].set_title('Features {} vs {}'.format(row+1, col+1))

    for row in range(5, 10):
        for col in range(5, 10):
            ax4[row-5][col-5].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax4[row-5][col-5].set_title('Features {} vs {}'.format(row+1, col+1))

    for row in range(5, 10):
        for col in range(10, 13):
            ax5[row-5][col-10].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax5[row-5][col-10].set_title('Features {} vs {}'.format(row+1, col+1))

    '''
    last 3 rows x 13 columns
    '''
    for row in range(10, 13):
        for col in range(0, 5):
            ax6[row-10][col].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax6[row-10][col].set_title('Features {} vs {}'.format(row+1, col+1))

    for row in range(10, 13):
        for col in range(5, 10):
            ax7[row-10][col-5].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax7[row-10][col-5].set_title('Features {} vs {}'.format(row+1, col+1))

    for row in range(10, 13):
        for col in range(10, 13):
            ax8[row-10][col-10].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax8[row-10][col-10].set_title('Features {} vs {}'.format(row+1, col+1))


    plt.show()
    """

    # The selected features are 7 and 13
    # The indices are 6 and 12
    return [6, 12]


'''
FUNCTIONS FOR K-NN ---------------------------------------------------------------------------
'''
def reduce_data(train_set, test_set, selected_features):
    # the following are (n, 2) arrays
    train_set_red = train_set[:, selected_features]
    test_set_red = test_set[:, selected_features]

    return train_set_red, test_set_red

def calculate_centroids(train_set, train_labels):
    classes = np.unique(train_labels)
    centroids = np.array([np.mean(train_set[train_labels == c, :], axis=0) for c in classes])
    
    return centroids, classes

# might not need this function in this form lol
def nearest_centroid(centroids, test_set):
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
    centroid_dist = lambda x : [dist(x, centroid) for centroid in centroids]
    predicted = np.argmin([centroid_dist(p) for p in test_set], axis=1).astype(np.int) + 1
    
    return predicted

'''
def distance(point, trainPoint):
    # so we already have this function but oh well we can make our own
    dist = 0
    dist = np.square(data1[x] - data2[x])
    
    return np.sqrt(dist)
'''

def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    print("hello")

    '''
    predict the class of a given unknown sample
    Use 'accuracy' to evaluate your classifier on the test set
    Use k∈{1,2,3,4,5,7} and discuss how results vary according to k on report
    Calculate the confusion matrix to discuss how the classifier behaves according to the three different classes
    You may either choose a single k for your discussion or show how the confusion matrix changes according to k
    both approaches are valid and we leave this decision to you
    '''
    # Step 0.1: I'm guessing we need a list of k values
    # -> dont think so because k is being passed as argument to this function 

    # Step 0.2: Reduce the data
    reduced_train, reduced_test = reduce_data(train_set, test_set, [6,12])

    # Step 1: Calculate each new data to all other training points (distance can by any type)
    # so instead of all other training points we use the centroids...? nahh
    # for i in range(len(train_set_red)):

    # Step 2: selects the K-nearest data points
    # check which class the k number of points are close to, selects majority class

    # Step 3: assigns the data point to the class to which the majority of the K data points belong

    #func to find the dist
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
     
    #iterate from 1 to total number of training data points
    for i in range(0, reduced_test.shape[0]):
        for j in range(0, reduced_test.shape[0]):
            # store the current test data point
            testPoint = reduced_test[i][j]

            #Calculate the distance between test data and each row of training data.
            #Should return the array with all the dist from testPoint to every train point
            #(lets ignore this chaos) dist_test_to_train = lambda x : [dist(x, train) for train in reduced_train]
            for k in range(0, reduced_train.shape[0]):
                dist_test_to_train = dist(testPoint, reduced_test[k]) # this should send the test point and a row?

            #Sort the calculated distances in ascending order based on distance values
            # This is to order the elements in nearest to furthest
            sortedDist = np.sort(dist_test_to_train)
            
            #Get top k rows from the sorted array
            for count in range(0, k):
                kNeighbours[count] = sortedDist[count]
        
            #Get the most frequent class of these rows
            #predicted[i] = most frequent class from the list kNeighbours

    # return predictions...?
    # this should also be a (1, n) or (n, 1) array
    return predicted

'''
FUNCTIONS FOR EVALUATING A CLASSIFIER --------------------------------------------------------
'''    

def calculate_accuracy(gt_labels, pred_labels):
    correct = 0
    wrong = 0
    for i in range (len(gt_labels)):
        if (gt_labels[i] == pred_labels[i]):
            correct+=1
        else:
            wrong+=1
    acc = correct / len(gt_labels)
    return acc

def percentage(gt_labels, pred_labels, classNum, isDiag):
    correct = 0
    wrong = 0
    total = 0
    #print(classNum)
    for i in range (len(gt_labels)):
        # go through this
        if (gt_labels[i] == classNum):
            total += 1
            if (pred_labels[i] == classNum):
                correct+=1
            else:
                wrong+=1

    if (isDiag):
        percentage = correct / total
    else:
        percentage = wrong / total

    return percentage

def calculate_confusion_matrix(gt_labels, pred_labels):

    gtClasses = np.unique(gt_labels)
    predClasses = np.unique(pred_labels)
    
    confuMatrix = np.zeros((len(gtClasses), len(gtClasses)))
    
    for row in gtClasses: 
        for col in gtClasses:
            if (row == col):
                confuMatrix[row-1][col-1]=percentage(gt_labels, pred_labels, row, True)
            else:
                confuMatrix[row-1][col-1]=percentage(gt_labels, pred_labels, row, False)
    return confuMatrix

'''
FUNCTIONS THAT THEY GAVE TO US -------------------------------------------------------------
'''





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
    
    knn(train_set, train_labels, test_set, 2)


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