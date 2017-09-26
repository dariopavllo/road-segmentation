# -*- coding: utf-8 -*-

import numpy as np

class NaiveModel:
    
    def __init__(self):
        """
        Construct a bogus classifier that classifies all pixels as background.
        """
        # Nothing to do
        return

    def initialize(self):
        """ Initialize or reset this model. """
        # Nothing to do
        return
    
    def train(self, Y, X):
        print('Training completed')
        # Nothing to do
        
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
        # Classify everything as background
        return np.zeros((X.shape[0], X.shape[1]*X.shape[2]//256))
        