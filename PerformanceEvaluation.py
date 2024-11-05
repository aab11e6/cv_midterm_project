from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid
from random import randint
from sklearn.metrics.pairwise import cosine_distances

def performance_evaluation(match_results, metric):

    # Create the ture classification of test image
    ground_truth = [i // 4 + 1 for i in range(432)]
    
    # Calculate the correct matches
    correct_matches = 0
    for i, result in enumerate(match_results):
        expected_label = ground_truth[i]
        if result == expected_label:
            correct_matches += 1
    
    # Calculate and print the accuracy
    accuracy = (correct_matches / len(match_results)) * 100
    # print(f"{metric} Accuracy: {accuracy:.2f}%")
    return accuracy

def iris_matching_table3(training_features, testing_features, metric):
    # Assuming training_features and testing_features are lists of tuples (label, feature_vector)
    train_labels = [label for label, _ in training_features]
    train_feature_vectors = np.array([vector for _, vector in training_features])
    
    # Classify each testing feature using Nearest Centroid Classifier
    clf = NearestCentroid(metric=metric)
    clf.fit(train_feature_vectors, train_labels)
    
    match_results = [] # to store the matching result
    for _, test_feature_vector in testing_features:
        # Predict the lable of test feature vector using classifier
        match_result = clf.predict([test_feature_vector])[0]
        match_results.append(match_result)
    
    return match_results

def iris_matching_figure10(training_features, testing_features, metric, n):
    # Perform dimensionality reduction using Fisher Linear Discriminant (FLD)
    # Assuming training_features and testing_features are lists of tuples (label, feature_vector)
    train_labels = [label for label, _ in training_features]
    train_feature_vectors = np.array([vector for _, vector in training_features])

    lda = LDA(n_components=min(len(np.unique(train_labels)) - 1, n))
    reduced_train_features = lda.fit_transform(train_feature_vectors, train_labels)
    
    # Classify each testing feature using Nearest Centroid Classifier
    clf = NearestCentroid(metric=metric)
    clf.fit(reduced_train_features, train_labels)
    
    match_results = [] # to store the matching result
    for _, test_feature_vector in testing_features:
        # Apply FLD feature dimensionality reduction
        reduced_test_feature = lda.transform([test_feature_vector])
        # Predict the lable of test feature vector using classifier
        match_result = clf.predict(reduced_test_feature)[0]
        match_results.append(match_result)
    
    return match_results
    
def iris_matching_table4(training_features, testing_features, metric):
    # Perform dimensionality reduction using Fisher Linear Discriminant (FLD)
    # Assuming training_features and testing_features are lists of tuples (label, feature_vector)
    train_labels = [label for label, _ in training_features]
    train_feature_vectors = np.array([vector for _, vector in training_features])

    lda = LDA(n_components=min(len(np.unique(train_labels)) - 1, 150))
    reduced_train_features = lda.fit_transform(train_feature_vectors, train_labels)
    
    # Classify each testing feature using Nearest Centroid Classifier
    clf = NearestCentroid(metric=metric)
    clf.fit(reduced_train_features, train_labels)
    
    match_results = []
    distances = []
    for _, test_feature_vector in testing_features:
        reduced_test_feature = lda.transform([test_feature_vector])
        match_result = clf.predict(reduced_test_feature)[0]
        centroid = clf.centroids_[list(clf.classes_).index(match_result)]
        if metric == 'cosine':
            dist = cosine_distances(reduced_test_feature, [centroid])[0][0]
        elif metric == 'manhattan':
            dist = np.sum(np.abs(reduced_test_feature - centroid))
        elif metric == 'euclidean':
            dist = np.linalg.norm(reduced_test_feature - centroid)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        match_results.append(match_result)
        distances.append(dist)
    
    return match_results, distances


def roc(dist, prediction, threshold):
    TP, FP, TN, FN = 0, 0, 0, 0

    # Calculate TP, FP, TN, FN values
    for i, dist in enumerate(dist):
        is_match = dist < threshold
        correct_prediction = prediction[i] == (i // 4) + 1

        if is_match and correct_prediction:
            TP += 1
        elif is_match and not correct_prediction:
            FP += 1
        elif not is_match and correct_prediction:
            FN += 1
        else:
            TN += 1

    # Calculate the false_match_rate, false_non_match_rate, true_positive_rate, false_positive_rate
    false_match_rate = FP / (TP + FP) if (TP + FP) > 0 else None
    false_non_match_rate = FN / (TN + FN) if (TN + FN) > 0 else None
    true_positive_rate = TP / (TP + FN) if (TP + FN) > 0 else None
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else None

    return false_match_rate, false_non_match_rate, true_positive_rate, false_positive_rate