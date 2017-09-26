# -*- coding: utf-8 -*-

import numpy as np
from helpers import *

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold cross-validation.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def get_classification_results(y, y_test):
    """
    Get the ratio of correct answers.
    """
    y = y.reshape(-1) # Linearize
    y_test = y_test.reshape(-1) # Linearize
    diff = y - y_test
    correct = np.sum(diff == 0)
    return correct / y_test.size

def cross_validation_iteration(model, Y, X, k_indices, k):
    """
    Execute a single run of cross-validation.
    Returns the ratio of correct answers on the validation set.
    """
    non_k_indices = k_indices[np.arange(k_indices.shape[0]) != k].ravel()
    tx_tr = X[non_k_indices]
    y_tr = Y[non_k_indices]
    tx_te = X[k_indices[k]]
    y_te = Y[k_indices[k]]
    
    model.initialize() # Reset model
    model.train(y_tr, tx_tr)
    #model.save('cv_' + str(k)) # For debug purposes
    
    # Run classification
    Z = model.classify(tx_te)
    
    # Calculate ground-truth labels
    img_patches_gt = create_patches_gt(y_te, 16, 16)
    y_real = np.mean(img_patches_gt, axis=(1, 2)) > 0.25
    
    return get_classification_results(y_real, Z)    
    
def k_fold_cross_validation(model, Y, X, k_fold, seed):
    """
    Run a full k-fold cross-validation and print mean accuracy and standard deviation.
    """
    np.random.seed(seed)
    k_indices = build_k_indices(Y, k_fold, seed)
    results = np.zeros(k_fold)
    for k in range(k_fold):
        results[k] = cross_validation_iteration(model, Y, X, k_indices, k)
        print('Accuracy: ' + str(results[k]))
    print(results)
    print('Cross validation accuracy: ' + str(np.mean(results)) + ', std=' + str(np.std(results)))
    
def fast_cross_validation(model, Y, X, k_fold, seed):
    """
    Execute cross-validation with a static validation set,
    i.e. a single run of k-fold cross-validation.
    """
    np.random.seed(seed)
    k_indices = build_k_indices(Y, k_fold, seed)
    result = cross_validation_iteration(model, Y, X, k_indices, 0)
    print('Cross validation accuracy: ' + str(result))
   