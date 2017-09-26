# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from helpers import *

class LogisticModel:
    
    def __init__(self):
        """
        Construct a logistic regression classifier.
        """
        self.patch_size = 16

    def initialize(self):
        """ Initialize or reset this model. """
        self.logreg = linear_model.LogisticRegression(C=1e5)
    
    def extract_features(self, img):
        """
        Extract features from a patch:
        - Average color for each RGB channel
        - Standard deviation for each RGB channel
        """
        feat_m = np.mean(img, axis=(0,1))
        feat_v = np.std(img, axis=(0,1))
        feat = np.append(feat_m, feat_v)
        return feat
    
    def poly_fit(self, X):
        """
        Fit the dataset using a polynomial basis.
        """
        poly = PolynomialFeatures(4, interaction_only=False)
        return poly.fit_transform(X)
    
    # Extract features for a given image
    def extract_img_features(self, filename):
        img = load_image(filename)
        img_patches = img_crop(img, patch_size, patch_size)
        X = np.asarray([self.extract_features(img_patches[i]) for i in range(len(img_patches))])
        return X
    
    def train(self, Y, X):
        """
        Train this model.
        """
        
        foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

        def value_to_class(v):
            df = np.sum(v)
            if df > foreground_threshold:
                return 1
            else:
                return 0
        
        # Extract patches from input images
        patch_size = self.patch_size
        img_patches = [img_crop(X[i], patch_size, patch_size, patch_size, 0) for i in range(X.shape[0])]
        gt_patches = [img_crop_gt(Y[i], patch_size, patch_size, patch_size) for i in range(X.shape[0])]

        # Linearize list of patches
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
        
        X = np.asarray([self.extract_features(img_patches[i]) for i in range(len(img_patches))])
        Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
           
        X = self.poly_fit(X)
        self.logreg.fit(X, Y)
        
        print('Training completed')
        
    def save(self, filename):
        # Nothing to do
        return
        
    def load(self, filename):
        # Nothing to do
        return
        
    def classify(self, X):
        """
        Classify an unseen set of samples.
        This method must be called after "train".
        Returns a list of predictions.
        """
        patch_size = self.patch_size
        img_patches = [img_crop(X[i], patch_size, patch_size, patch_size, 0) for i in range(X.shape[0])]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        X = np.asarray([self.extract_features(img_patches[i]) for i in range(len(img_patches))])
        X = self.poly_fit(X)
        Z = self.logreg.predict(X)
        # Regroup patches into images
        return group_patches(Z, X.shape[0])
        