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
import math
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']  

# because the repeated code everywhere is annoying me
def set_colours(train_labels):
    colours = np.zeros_like(train_labels, dtype=np.object)
    colours[train_labels == 1] = CLASS_1_C
    colours[train_labels == 2] = CLASS_2_C
    colours[train_labels == 3] = CLASS_3_C
    return colours

def feature_selection(train_set, train_labels, **kwargs):
    colours = set_colours(train_labels)

    ### attempt to display 13 x 13 things
    #n_features = train_set.shape[1]

    # this is code for outputting 3d stuff
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_set[:, 6], train_set[:, 10], train_set[:, 12], c=colours)
    plt.show()

    # The selected features are 7 and 10
    # The indices are 6 and 9
    return [6, 9]

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


def percentage(gt_labels, pred_labels, classNum, isDiag, otherClassNum=None):
    correct = 0
    wrong = 0
    total = 0
    percentage = 0
    for i in range (len(gt_labels)):
        # go through this
        if (gt_labels[i] == classNum):
            total += 1
            if (pred_labels[i] == classNum):
                correct+=1
            elif (pred_labels[i] == otherClassNum):
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
                confuMatrix[row-1][col-1] = percentage(gt_labels, pred_labels, row, True)
            else:
                confuMatrix[row-1][col-1] = percentage(gt_labels, pred_labels, row, False, col)
    return confuMatrix

'''
FUNCTIONS FOR K-NN ---------------------------------------------------------------------------
'''
def reduce_data(train_set, test_set, selected_features):
    # the following are (n, 2) arrays
    train_set_red = train_set[:, selected_features]
    test_set_red = test_set[:, selected_features]
    return train_set_red, test_set_red


def plot_matrix(matrix, ax=None):
    """
    Displays a given matrix as an image.
    
    Args:
        - matrix: the matrix to be displayed        
        - ax: the matplotlib axis where to overlay the plot. 
          If you create the figure with `fig, fig_ax = plt.subplots()` simply pass `ax=fig_ax`. 
          If you do not explicitily create a figure, then pass no extra argument.  
          In this case the  current axis (i.e. `plt.gca())` will be used        
    """    
    if ax is None:
        ax = plt.gca()
     
    #rip
    matrix = matrix.transpose()

    # Plot the colour image 
    cmap=plt.get_cmap('Wistia')    
    image = ax.imshow(matrix, cmap)
    
    # Create colorbar
    cbar = ax.figure.colorbar(image, ax=ax)
    
    # Displaying the text
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[0]):
            image.axes.text(i, j, round(matrix[i][j],2), horizontalalignment='center', verticalalignment='center', color='black')
    
    plt.show()


def knn(train_set, train_labels, test_set, k, **kwargs):
    selected_features = [6, 9]
    # Reduce the data
    reduced_train, reduced_test = reduce_data(train_set, test_set, selected_features)

    predicted = np.zeros((reduced_test.shape[0], 1))

    #func to find the dist
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
     
    # for loop to go through each test points
    for i in range(0, reduced_test.shape[0]):
        # store the current test data point
        testPoint = reduced_test[i]

        #Calculate the distance between test data and each row of training data.
        dist_test_to_train = lambda testPoint : [dist(testPoint, train) for train in reduced_train]
        
        results = dist_test_to_train(testPoint)
        
        # Selecting minimum k distances
        closestIndexs = np.argsort(results)[:k]
 
        #Get the most frequent class of these rows
        classes = []
        freqClass = []
        for index in closestIndexs:
            classes.append(train_labels[index])

        # PICK THE MOST FRQUEST CLASS 
        freqClass = np.argmax(np.bincount(classes))

        predicted[i] = freqClass
    
    
    # ----------- ACCURACY --------------------------------------------------
    accuracy = calculate_accuracy(kwargs["test_labels"], predicted)
    print("ACCURACY: " + str(accuracy))

    # ----------- CONFUSION MATRIX ------------------------------------------
    confuMat = calculate_confusion_matrix(kwargs["test_labels"], predicted)
    print("CONFUSION MATRIX: ")
    print(confuMat)
    
    # Plotting the confu mat
    fig, a = plt.subplots()
    plt.title("k = {}".format(k))
    plot_matrix(confuMat, ax = a)
    
    return predicted

'''
FUNCTIONS THAT THEY GAVE TO US -------------------------------------------------------------
'''
def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    """
    NAIVE BAYES CLASSIFIER
    posterior = (likelikood * prior) / evidence
    P(class|test_features) = ( P(train_features|class) * P(class) ) / P(train_features)
    """
    # Saving the selected features
    reduced_train, reduced_test = reduce_data(train_set, test_set, [6,9])
    feature1 = reduced_train[:, 0] 
    feature2 = reduced_train[:, 1] 

    predicted = np.zeros((reduced_test.shape[0], 1))

    # Calculating the priors ---------------------------------------------------------
    unique, counts = np.unique(train_labels, return_counts=True)
    total_no_classes = train_labels.shape[0]

    prior_class1 =  counts[0] / total_no_classes
    prior_class2 =  counts[1] / total_no_classes
    prior_class3 =  counts[2] / total_no_classes

    # Calculating the likelihood -----------------------------------------------------
    mean_pairs = np.zeros((2,3))
    var_pairs = np.zeros((2,3))

    for f in range(2):
        for c in range(3):
            if (f == 0): # feature 1
                mean_pairs[f][c] = np.mean(feature1[train_labels == c+1])
                var_pairs[f][c] = np.var(feature1[train_labels == c+1])

            elif (f == 1): # feature 2
                mean_pairs[f][c] = np.mean([feature2[train_labels == c+1]])
                var_pairs[f][c] = np.var([feature2[train_labels == c+1]])

    # Getting the posteriors for each test point------------------------------------
    for i in range(reduced_test.shape[0]):
        posterior = [0, 0, 0] # this is the array that will store the probabilities

        npdf = lambda x, mean, var: ( 1.0 / ( np.sqrt( 2.0*np.pi*var)) ) * np.exp( (-(x-mean)**2.0) / (2.0 * var) )
        likelihood = lambda f, c: npdf(reduced_test[i][f-1], mean_pairs[f-1][c-1], var_pairs[f-1][c-1] )

        posterior[0] = likelihood(1, 1) * likelihood(2, 1) * prior_class1
        posterior[1] = likelihood(1, 2) * likelihood(2, 2) * prior_class2
        posterior[2] = likelihood(1, 3) * likelihood(2, 3) * prior_class3

        predicted[i] = np.argmax(posterior) + 1

    # ----------- ACCURACY --------------------------------------------------
    accuracy = calculate_accuracy(kwargs["test_labels"], predicted)
    print("ACCURACY: " + str(accuracy))

    # ----------- CONFUSION MATRIX ------------------------------------------
    confuMat = calculate_confusion_matrix(kwargs["test_labels"], predicted)
    print("CONFUSION MATRIX: ")
    #print(confuMat)

    # Plotting the confu mat
    fig, a = plt.subplots()
    plt.title("Confusion Matrix for Naive Bayes Classifier")
    plot_matrix(confuMat, ax = a)

    return predicted

def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    ## STILL NEED TO DECIDE A THIRD FEATURE
    reduced_train, reduced_test = reduce_data(train_set, test_set, [6, 9, 12])
  
    predicted = np.zeros((reduced_test.shape[0], 1))

    #func to find the dist
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
     
    # for loop to go through each test points
    for i in range(0, reduced_test.shape[0]):
        testPoint = reduced_test[i]

        dist_test_to_train = lambda testPoint : [dist(testPoint, train) for train in reduced_train]
        results = dist_test_to_train(testPoint)

        closestIndexs = np.argsort(results)[:k]
 
        classes = []
        freqClass = []
        for index in closestIndexs:
            classes.append(train_labels[index])

        freqClass = np.argmax(np.bincount(classes))

        predicted[i] = freqClass


    # ----------- ACCURACY --------------------------------------------------
    accuracy = calculate_accuracy(kwargs["test_labels"], predicted)
    print("ACCURACY: " + str(accuracy))

    # ----------- CONFUSION MATRIX ------------------------------------------
    confuMat = calculate_confusion_matrix(kwargs["test_labels"], predicted)
    print("CONFUSION MATRIX: ")
    print(confuMat)
    
    # Plotting the confu mat
    fig, a = plt.subplots()
    plt.title("Confusion Matrix")
    plot_matrix(confuMat, ax = a)

    return predicted

'''
 PCA STUFF WOOT WOOT ------------------------------------------------------------
'''
def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    colours = set_colours()

    # creating PCA object
    pca = PCA(n_components=2)

    pca.fit(train_set)
    pca_red_train = pca.transform(train_set)
    pca_red_test = pca.transform(test_set)

    plt.title("PCA")
    plt.scatter(pca_red_train[:, 0], pca_red_train[:, 1] * (-1), c=colours, s=100)
    plt.show()

    # Running knn -----------------------------------------------------------------------
    predicted = np.zeros((pca_red_test.shape[0], 1))
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
     
    for i in range(0, pca_red_test.shape[0]):
        testPoint = pca_red_test[i]

        dist_test_to_train = lambda testPoint : [dist(testPoint, train) for train in pca_red_train]
        results = dist_test_to_train(testPoint)
        closestIndexs = np.argsort(results)[:k]
 
        classes = []
        freqClass = []
        for index in closestIndexs:
            classes.append(train_labels[index])

        freqClass = np.argmax(np.bincount(classes))
        predicted[i] = freqClass
    
    # end of knn ------------------------------------------------------------------------

    # ----------- ACCURACY --------------------------------------------------
    accuracy = calculate_accuracy(kwargs["test_labels"], predicted)
    print("ACCURACY: " + str(accuracy))

    # ----------- CONFUSION MATRIX ------------------------------------------
    confuMat = calculate_confusion_matrix(kwargs["test_labels"], predicted)
    print("CONFUSION MATRIX: ")
    print(confuMat)
    
    # Plotting the confu mat
    fig, a = plt.subplots()
    plt.title("Confusion Matrix")
    plot_matrix(confuMat, ax = a)

    return predicted


'''
MAIN FUNCTIONS --------------------------------------------------------------------------
'''
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
        predictions = knn(train_set, train_labels, test_set, args.k, test_labels=test_labels)
        print_predictions(predictions)

        
        # checking things
        neigh = KNeighborsClassifier(n_neighbors=7)
        neigh.fit(train_set[:,[6,9]], train_labels)
        print("comp predicted: ")
        comp = neigh.predict(test_set[:, [6,9]]) 
        print( comp )
        print("comp accuracy: ")
        print(neigh.score(test_set[:, [6,9]], test_labels, sample_weight=None))
        
        for i in range(0, predictions.shape[0]):
            if (predictions[i] != comp[i]):
                print("uh oh", i)
        

    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set, test_labels=test_labels)
        print_predictions(predictions)

        # some more checkings
        
        gnb = GaussianNB()
        gnb.fit(train_set[:,[6,9]], train_labels)
        print("comp predicted: ")
        comp = gnb.predict(test_set[:, [6,9]]) 
        print( comp )
        print("comp accuracy: ")
        print(gnb.score(test_set[:, [6, 9]], test_labels, sample_weight=None))

        for i in range(0, predictions.shape[0]):
            if (predictions[i] != comp[i]):
                print("uh oh", i)
        
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k, test_labels=test_labels)
        print_predictions(predictions)

        """
        # checking things
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(train_set[:,[6,9,12]], train_labels)
        print("comp predicted: ")
        comp = neigh.predict(test_set[:, [6,9,12]]) 
        print( comp )
        print("comp accuracy: ")
        print(neigh.score(test_set[:, [6,9,12]], test_labels, sample_weight=None))
        
        for i in range(0, predictions.shape[0]):
            if (predictions[i] != comp[i]):
                print("uh oh", i)
        """

    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k, test_labels=test_labels)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))