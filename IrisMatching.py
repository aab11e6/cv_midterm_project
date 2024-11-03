# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:30:02 2024

@author: 11936
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid
    
def iris_matching(training_features, testing_features, metric):
    # Perform dimensionality reduction using Fisher Linear Discriminant (FLD)
    # Assuming training_features and testing_features are lists of tuples (label, feature_vector)
    train_labels = [label for label, _ in training_features]
    train_feature_vectors = np.array([vector for _, vector in training_features])

    lda = LDA(n_components=min(len(np.unique(train_labels)) - 1, 150))
    # lda = LDA(n_components=150)
    reduced_train_features = lda.fit_transform(train_feature_vectors, train_labels)
    
    # Classify each testing feature using Nearest Centroid Classifier
    clf = NearestCentroid(metric=metric)
    clf.fit(reduced_train_features, train_labels)
    
    match_results = []
    for _, test_feature_vector in testing_features:
        reduced_test_feature = lda.transform([test_feature_vector])
        match_result = clf.predict(reduced_test_feature)[0]
        match_results.append(match_result)
    
    return match_results
