import os
import cv2
import numpy as np

from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid
# Import other modules
from IrisLocalization import iris_localization
from IrisNormalization import iris_normalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction
from IrisMatching import iris_matching
from PerformanceEvaluation import performance_evaluation
from sklearn.metrics.pairwise import euclidean_distances

import pickle

def iris_recognition():
    # Define paths for training and testing datasets
    training_base_path = './Data'
    testing_base_path = './Data'

    # Initialize lists to store training and testing data
    training_images = []
    testing_images = []

    # Load training images (first session)
    for i in range(1, 109):  # Loop through folders 001 to 108
        folder_path = os.path.join(training_base_path, f'{i:03d}', '1')
        for j in range(1, 4):  # Load three images per person
            image_name = f'{i:03d}_1_{j}.bmp'  # Image name format: 097_1_1
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                training_images.append((i, image))  # Store tuple of (person_id, image)
            else:
                print(f"Warning: Could not read image {image_path}")

    # Load testing images (second session)
    for i in range(1, 109):  # Loop through folders 001 to 108
        folder_path = os.path.join(testing_base_path, f'{i:03d}', '2')
        for j in range(1, 5):  # Load four images per person
            image_name = f'{i:03d}_2_{j}.bmp'  # Image name format: 097_2_1
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                testing_images.append((i, image))  # Store tuple of (person_id, image)
            else:
                print(f"Warning: Could not read image {image_path}")

    # # Process training images
    # training_features = []
    # for person_id, image in training_images:
    #     try:
    
    #         # Step 1: Iris Localization
    #         inner_boundary, outer_boundary = iris_localization(image)
            
    #         # Step 2: Iris Normalization
    #         normalized_iris = iris_normalization(image, inner_boundary, outer_boundary)
            
    #         # Step 3: Image Enhancement
    #         enhanced_iris = image_enhancement(normalized_iris)
            
    #         # Step 4: Feature Extraction
    #         features = feature_extraction(enhanced_iris)
    #         training_features.append((person_id, features))
    #     except ValueError as e:
    #         print(f"Error processing training image for person {person_id}: {e}")

    # # Process testing images
    # testing_features = []
    # for person_id, image in testing_images:
    #     try:
    #         # Step 1: Iris Localization
    #         inner_boundary, outer_boundary = iris_localization(image)
            
    #         # Step 2: Iris Normalization
    #         normalized_iris = iris_normalization(image, inner_boundary, outer_boundary)
            
    #         # Step 3: Image Enhancement
    #         enhanced_iris = image_enhancement(normalized_iris)
            
    #         # Step 4: Feature Extraction
    #         features = feature_extraction(enhanced_iris)
    #         testing_features.append((person_id, features))
            
    #     except ValueError as e:
    #         print(f"Error processing testing image for person {person_id}: {e}")
            
            
    # with open('training_features.pkl', 'wb') as file:
    #     pickle.dump(training_features, file)
    
    # with open('testing_features.pkl', 'wb') as file:
    #     pickle.dump(testing_features, file)
        
    with open('training_features.pkl', 'rb') as file:
        training_features = pickle.load(file)
        
    with open('testing_features.pkl', 'rb') as file:
        testing_features = pickle.load(file)
    
    for metric in ['euclidean', 'manhattan', 'cosine']:
    
        # Step 5: Iris Matching
        match_result = iris_matching(training_features, testing_features, metric)
        
        # Step 6: Performance Evaluation (optional, depending on mode)
        performance_evaluation(match_result, metric)

        f1, f2 = roc(training_features, testing_features, metric)
        print(f1)
        print(f2)


def roc(training_features, testing_features, metric):
    # Get matching results and classifier from iris_matching
    train_labels = [label for label, _ in training_features]
    train_feature_vectors = np.array([vector for _, vector in training_features])
    lda = LDA(n_components=min(len(np.unique(train_labels)) - 1, 150))
    reduced_train_features = lda.fit_transform(train_feature_vectors, train_labels)
    clf = NearestCentroid(metric=metric)
    clf.fit(reduced_train_features, train_labels)
    
    # Define the ground truth
    # ground_truth = [i // 4 + 1 for i in range(len(testing_features))]
    
    # Calculate FMR and FNMR at different thresholds for ROC calculation
    thresholds = np.linspace(-20, 20, 30)  # Example threshold values for similarity score
    fmr_list = []
    fnmr_list = []
    
    for threshold in thresholds:
        false_match = 0
        false_non_match = 0
        total_imposters = 0
        total_genuine = 0
        
        for label, test_feature_vector in testing_features:
            reduced_test_feature = lda.transform([test_feature_vector])
            distance = euclidean_distances(reduced_test_feature, clf.centroids_).min()
            similarity_score = -distance  # Negative distance for similarity
            
            if label == clf.predict(reduced_test_feature)[0]:  # Genuine match
                total_genuine += 1
                if similarity_score < threshold:
                    false_non_match += 1
            else:  # Imposter
                total_imposters += 1
                if similarity_score >= threshold:
                    false_match += 1
        
        fmr = false_match / total_imposters if total_imposters > 0 else 0
        fnmr = false_non_match / total_genuine if total_genuine > 0 else 0
        fmr_list.append(fmr)
        fnmr_list.append(fnmr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fmr_list, fnmr_list, color='darkorange', lw=2, label='ROC Curve')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title(f'Receiver Operating Characteristic ({metric})')
    plt.legend(loc='lower right')
    plt.show()
    
    return fmr_list, fnmr_list



if __name__ == "__main__":
    iris_recognition()
