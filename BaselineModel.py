#!/usr/bin/env python3

"""
We are defining a custom class BaselineModel to encapsulate any
baseline model like logistic regression, SVM, random forest...
and manipulate them easily.
"""

import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

from helpers_loading import *
from helpers_submission import *
from helpers_visualization import *

def extract_features_2d(img):
    """
    Extracts 2-dimensional features consisting of average gray color as well as variance.
    """
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features(img):
    """
    Extracts 6-dimensional features consisting of average RGB color as well as variance.
    """
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def enhance_features_poly(X, degree=4):
    """
    Basic function to enhance features
    """
    poly = PolynomialFeatures(degree, interaction_only=False)
    return poly.fit_transform(X)

class BaselineModel:
    
    def __init__(self, model, feature_extraction_function=extract_features, feature_enhancement_function=enhance_features_poly, extracted=False):
        """
        Creates a baseline model.
        """
        self.patch_size = 16
        
        self.model = model
        self.feature_extraction_function = feature_extraction_function
        self.feature_enhancement_function = feature_enhancement_function
        self.extracted = extracted
    
    def train(self, X, Y):
        """
        Train the model with the given inputs.
        """        
        if not self.extracted:
            X = extract_patches(X)
        
        X = patches_to_features(X, feature_extraction_function=self.feature_extraction_function)
        X = self.feature_enhancement_function(X)
    
        if not self.extracted:
            Y = extract_patches(Y)
        
        Y = patches_to_labels(Y)
        
        self.model.fit(X, Y)

    def predict(self, X):
        """
        Returns the predictions from the given images X.
        """
        if not self.extracted:
            X_f = extract_patches(X)
        
        X_f = patches_to_features(X_f, feature_extraction_function=self.feature_extraction_function)
        X_f = self.feature_enhancement_function(X_f)

        Y = self.model.predict(X_f)

        return labels_to_images(Y, n=X.shape[0], w=self.patch_size, h=self.patch_size, imgwidth=X.shape[1], imgheight=X.shape[2])
    
    def cross_validation(self, X, Y, cv=10, scoring='f1_micro'):
        """
        Used to guess the accuracy of the model.
        """
        if not self.extracted:
            X = extract_patches(X)
            Y = extract_patches(Y)
        
        X = patches_to_features(X, feature_extraction_function=self.feature_extraction_function)
        Y = patches_to_labels(Y)
        
        scores = cross_val_score(self.model, X, Y, cv=cv, scoring=scoring)
        
        return scores
    
    def cross_validation_print(self, X, Y, cv=10, scoring='f1_micro'):
        """
        Print the result of the cross-validation.
        """
        scores = self.cross_validation(X, Y, cv=cv, scoring=scoring)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))