#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sys
import time


def euc(x, y):
    # calculate squared Euclidean distance

    # check dimension
    assert x.shape == y.shape

    diff = x - y

    return np.dot(diff, diff)


def mah(x, y, Sigma):
    # calculate squared Mahalanobis distance

    # check dimension
    assert x.shape == y.shape and max(x.shape) == max(Sigma.shape)
    
    diff = x - y
    
    return np.dot(np.dot(diff, np.linalg.inv(Sigma)), diff)


def gaussian(x, y, Sigma):
    # multivariate Gaussian

    assert x.shape == y.shape and max(x.shape) == max(Sigma.shape)

    d = max(x.shape)          # dimension
    dmah2 = mah2(x, y, Sigma)
    gx = 1.0 / ((2*np.pi)**(d/2) * np.linalg.det(Sigma)**0.5) * np.exp(-0.5 * dmah2)
    return gx


def accuracy_score(y, y_model):
    # calculate classification overall accuracy and classwise accuracy
    
    assert len(y) == len(y_model)
    classn = len(np.unique(y))       # number of different classes
    correct_all = y == y_model       # all correct classifications
    acc_overall = np.sum(correct_all) / len(y)
    acc_i = np.zeros(classn)
    for i in range(classn):   
        GT_i = y == i                # samples actually belong to class i
        acc_i[i] = (np.sum(GT_i & correct_all) / np.sum(GT_i))
        
    return acc_i, acc_overall


def load_data(f):
    """ Assume data format:
    feature1 feature 2 ... label 
    """

    # process training data
    data = np.genfromtxt(f)
    
    # return all feature columns except last
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    return X, y


def normalize(Tr, Te = None):
    # normalize the dataset such that different dimensions would have the same scale
    # use statistics of the training data to normalize both the training and testing sets
    # if only one argument, then just normalize that one set
    
    ntr, _ = Tr.shape
    stds = np.std(Tr, axis = 0)

    normTr = (Tr - np.tile(np.mean(Tr, axis = 0), (ntr, 1))) / stds
    if Te is not None:
        nte, _ = Te.shape
        normTe = (Te - np.tile(np.mean(Tr, axis = 0), (nte, 1))) / stds
    
    if Te is not None:
        return normTr, normTe
    else:
        return normTr
    
def mpp(Tr, yTr, Te, cases, P):
    # training process - derive the model
    covs, means = {}, {}     # dictionaries
    covsum = None

    classes = np.unique(yTr)   # get unique labels as dictionary items
    classn = len(classes)    # number of classes
    
    for c in classes:
        # filter out samples for the c^th class
        arr = Tr[yTr == c]  
        # calculate statistics
        covs[c] = np.cov(np.transpose(arr))
        means[c] = np.mean(arr, axis=0)  # mean along the columns
        # accumulate the covariance matrices for Case 1 and Case 2
        if covsum is None:
            covsum = covs[c]
        else:
            covsum += covs[c]
    
    # used by case 2
    covavg = covsum / classn
    # used by case 1
    varavg = np.sum(np.diagonal(covavg)) / classn
            
    # testing process - apply the learned model on test set 
    disc = np.zeros(classn)
    nr, _ = Te.shape
    y = np.zeros(nr)            # to hold labels assigned from the learned model

    for i in range(nr):
        for c in classes:
            if cases == 1:
                edist2 = euc(means[c], Te[i])
                disc[c] = -edist2 / (2 * varavg) + np.log(P[c] + 0.000001)
            elif cases == 2: 
                mdist2 = mah(means[c], Te[i], covavg)
                disc[c] = -mdist2 / 2 + np.log(P[c] + 0.000001)
            elif cases == 3:
                mdist2 = mah(means[c], Te[i], covs[c])
                disc[c] = -mdist2 / 2 - np.log(np.linalg.det(covs[c])) / 2 + np.log(P[c] + 0.000001)
            else:
                print("Can only handle case numbers 1, 2, 3.")
                sys.exit(1)
        y[i] = disc.argmax()
            
    return y    


def knn(Tr, yTr, Te, k):
    # training process - derive the model

    classes = np.unique(yTr)   # get unique labels as dictionary items
    classn = len(classes)      # number of classes
    ntr, _ = Tr.shape
    nte, _ = Te.shape
    
    y = np.zeros(nte)
    knn_count = np.zeros(classn)
    for i in range(nte):
        test = np.tile(Te[i,:], (ntr, 1))       # resembles MATLAB's repmat function
        dist = np.sum((test - Tr) ** 2, axis = 1) # calculate distance
        idist = np.argsort(dist)    # sort the array in the ascending order and return the index
        knn_label = yTr[idist[0:k]]
        for c in range(classn):
            knn_count[c] = np.sum(knn_label == c)
        y[i] = np.argmax(knn_count)
        
    return y    


def perceptron(Tr, yTr, Te = None):
    nr, nc = Tr.shape          # dimension
    w = np.random.rand(nc + 1) # initial weight
    y = np.zeros(nr)           # output from perceptron

    # training process
    finish = 0
    maxiter = 40
    n = 0        # number of iterations
    while not finish and n < maxiter:
        n += 1
        for i in range(nr):            
            y[i] = np.dot(w[:-1], Tr[i,:]) > w[-1]        # obtain the actual output
            w[:-1] = w[:-1] + (yTr[i] - y[i]) * Tr[i,:]  # online update weight
            w[-1] = w[-1] - (yTr[i] - y[i])       # update bias
        if np.dot(y - yTr, y - yTr) == 0:
            finish = 1
        print(f"Iteration {n}: Actual output from perceptron is: {y}, weights are {w}.")
        
    if Te is None:
        return w
    else:                   # the testing process
        ytest = np.matmul(w[:-1], np.transpose(Te)) > w[-1]
        return ytest.astype(int)


def main(): 
    # read in the datasets
    Xtrain, ytrain = load_data("python/datasets/synth.tr")
    Xtest, ytest = load_data("python/datasets/synth.te")
    Xtrain, Xtest = normalize(Xtrain, Xtest)
    # the training and testing datasets should have the same dimension
    _, nftrain = Xtrain.shape
    _, nftest = Xtest.shape
    assert nftrain == nftest   
    
    # normalize the datasets
    
    # ask the user to input which k to use
    str = input('Please input the number of nearest neighbors: ')
    k = int(str)
        
    # derive the decision rule from the training set and apply on the test set
    t0 = time.time()           # start time
    y_model = knn(Xtrain, ytrain, Xtest, k)
    t1 = time.time()           # ending time
    
    # calculate accuracy
    acc_classwise, acc_overall = util.accuracy_score(ytest, y_model)
    print(f'Overall accuracy = {acc_overall};')
    print(f'Classwise accuracy = {acc_classwise};')
    print(f'The learning process takes {t1 - t0} seconds.')
    
    
    
    # ask the user to input which discriminant function to use
    prompt = '''
    Type of discriminant functions supported assuming Gaussian pdf:
    1 - minimum Euclidean distance classifier
    2 - minimum Mahalanobis distance classifier
    3 - quadratic classifier
    '''
    print(prompt)
    str = input('Please input 1, 2, or 3: ')
    cases = int(str)
    
    # ask the user to input prior probability that needs to sum to 1
    prop_str = input("Please input prior probabilities in float numbers, separated by space, and they must add to 1: \n")
    numbers = prop_str.split()
    P = np.zeros(len(numbers))
    Psum = 0
    for i in range(len(numbers)):
        P[i] = float(numbers[i])
        Psum += P[i]
    if Psum != 1:
        print("Prior probabilities do not add up to 1. Please check!")
        sys.exit(1)
    
    # derive the decision rule from the training set and apply on the test set
    t0 = time.time()           # start time
    y_model = mpp(Xtrain, ytrain, Xtest, cases, P)
    t1 = time.time()           # ending time
    
    # calculate accuracy
    acc_classwise, acc_overall = util.accuracy_score(ytest, y_model)
    print(f'Overall accuracy = {acc_overall};')
    print(f'Classwise accuracy = {acc_classwise};')
    print(f'The learning process takes {t1 - t0} seconds.')
    
    
if __name__ == "__main__":
    main()
    

