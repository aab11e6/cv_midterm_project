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

# Define the function for iris matching table 4
# This function takes in training features, testing features, and a metric type for distance calculation.
# The function uses Fisher Linear Discriminant (FLD) for dimensionality reduction and Nearest Centroid for classification.
def iris_matching_table4(training_features, testing_features, metric):
    # Extract labels and feature vectors from the training features
    train_labels = [label for label, _ in training_features]
    train_feature_vectors = np.array([vector for _, vector in training_features])

    # Perform dimensionality reduction using Fisher Linear Discriminant (FLD)
    lda = LDA(n_components=min(len(np.unique(train_labels)) - 1, 150)) # Set the number of components as the minimum of (number of unique classes - 1) or 150, whichever is smaller.
    reduced_train_features = lda.fit_transform(train_feature_vectors, train_labels)
    
    # Instantiate and fit the Nearest Centroid Classifier using the reduced training features
    clf = NearestCentroid(metric=metric)
    clf.fit(reduced_train_features, train_labels)
    
    match_results = [] # Store the classification results for each testing feature
    distances = [] # Store the distances from the testing feature to its assigned centroid

    # Loop through each testing feature vector
    for _, test_feature_vector in testing_features:
        # Reduce the dimensionality of the testing feature using the previously trained LDA
        reduced_test_feature = lda.transform([test_feature_vector])

        # Predict the label for the testing feature using the Nearest Centroid classifier
        match_result = clf.predict(reduced_test_feature)[0]

        # Retrieve the centroid of the predicted class label
        centroid = clf.centroids_[list(clf.classes_).index(match_result)]

        # Calculate the distance between the reduced testing feature and the predicted class centroid
        if metric == 'cosine':
            # Use cosine distance to calculate the distance
            dist = cosine_distances(reduced_test_feature, [centroid])[0][0]
        elif metric == 'manhattan':
            # Use Manhattan distance to calculate the distance
            dist = np.sum(np.abs(reduced_test_feature - centroid))
        elif metric == 'euclidean':
            # Use Euclidean distance to calculate the distance
            dist = np.linalg.norm(reduced_test_feature - centroid)
        else:
            # Raise an error if an unsupported metric is provided
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Append the match result and the distance to their respective lists
        match_results.append(match_result)
        distances.append(dist)
    
    # Return the match results and distances
    return match_results, distances

# Define the function to compute ROC metrics
# This function takes in distances, predicted results, and a distance threshold to calculate performance metrics.
def roc(dist, prediction, threshold):
    # Initialize True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) counts to zero
    TP, FP, TN, FN = 0, 0, 0, 0

    # Calculate TP, FP, TN, FN values
    for i, dist in enumerate(dist):
        # Determine if the test case is a match based on the threshold value
        is_match = dist < threshold

        # Check if the prediction is correct by comparing the predicted label with the expected label
        # Assuming that each group of 4 consecutive testing features belong to the same label (i.e., (i // 4) + 1)
        correct_prediction = prediction[i] == (i // 4) + 1

        # Update the TP, FP, TN, FN counters based on the match status and prediction correctness
        if is_match and correct_prediction:
            TP += 1  # True Positive: correctly classified as a match
        elif is_match and not correct_prediction:
            FP += 1  # False Positive: incorrectly classified as a match
        elif not is_match and correct_prediction:
            FN += 1  # False Negative: incorrectly classified as not a match
        else:
            TN += 1  # True Negative: correctly classified as not a match

    # Calculate the false_match_rate, false_non_match_rate, true_positive_rate, false_positive_rate
    false_match_rate = FP / (TP + FP) if (TP + FP) > 0 else None
    false_non_match_rate = FN / (TN + FN) if (TN + FN) > 0 else None
    true_positive_rate = TP / (TP + FN) if (TP + FN) > 0 else None
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else None

    # Return the calculated ROC metrics
    return false_match_rate, false_non_match_rate, true_positive_rate, false_positive_rate