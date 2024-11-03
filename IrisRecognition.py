import os
import cv2
import numpy as np
import pandas as pd

# Import other modules
from IrisLocalization import iris_localization
from IrisNormalization import iris_normalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction
from IrisMatching import iris_matching
from PerformanceEvaluation import *

import pickle
from sklearn.model_selection import ParameterGrid
from prettytable import PrettyTable
from sklearn import metrics

import warnings

# Ignure FutureWarning and UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
       
    # Process training images
    training_features = []
    for person_id, image in training_images:
        try:
    
            # Step 1: Iris Localization
            inner_boundary, outer_boundary = iris_localization(person_id, image)
            
            # Step 2: Iris Normalization
            normalized_iris = iris_normalization(image, inner_boundary, outer_boundary)
            
            # Step 3: Image Enhancement
            enhanced_iris = image_enhancement(normalized_iris)
            
            # Step 4: Feature Extraction
            features = feature_extraction(enhanced_iris)
            training_features.append((person_id, features))
        except ValueError as e:
            print(f"Error processing training image for person {person_id}: {e}")

    # Process testing images
    testing_features = []
    for person_id, image in testing_images:
        try:
            # Step 1: Iris Localization
            inner_boundary, outer_boundary = iris_localization(person_id, image)
            
            # Step 2: Iris Normalization
            normalized_iris = iris_normalization(image, inner_boundary, outer_boundary)
            
            # Step 3: Image Enhancement
            enhanced_iris = image_enhancement(normalized_iris)
            
            # Step 4: Feature Extraction
            features = feature_extraction(enhanced_iris)
            testing_features.append((person_id, features))
            
        except ValueError as e:
            print(f"Error processing testing image for person {person_id}: {e}")
            
            
    # with open('training_features.pkl', 'wb') as file:
    #     pickle.dump(training_features, file)
    
    # with open('testing_features.pkl', 'wb') as file:
    #     pickle.dump(testing_features, file)
        
    # with open('training_features.pkl', 'rb') as file:
    #     training_features = pickle.load(file)
        
    # with open('testing_features.pkl', 'rb') as file:
    #     testing_features = pickle.load(file)
    
    
    ## Ellie -------------------------------------------------------------------------
    for metric in ['euclidean', 'manhattan', 'cosine']:
    
        # Step 5: Iris Matching
        match_result = iris_matching(training_features, testing_features, metric)
        
        # Step 6: Performance Evaluation (optional, depending on mode)
        performance_evaluation(match_result, metric)
    
    
    ## Huishan --------------------------------------------------------------------------
    ## Get the cosine similarity results
    cos_prediction, cos_dist = iris_matching_classifier(training_features, testing_features, metric='cosine')

    ## Generate FMR-FNMR table
    thresholds = [0.526, 0.601, 0.761]
    roc_results = [roc(cos_dist, cos_prediction, threshold) for threshold in thresholds]

    ## Create table to display results
    table = PrettyTable(['Threshold', 'False Match Rate (%)', 'False Non-Match Rate (%)'])
    for threshold, (fm, fnm, _, _) in zip(thresholds, roc_results):
        table.add_row([threshold, fm, fnm])

    print("Table 4. False Match and False Non-Match Rates with Different Threshold Values")
    print(table)
    print()

    ## Curve evaluation (cosine distance)
    thresh_range = np.arange(0.1, 0.7, 0.01)
    metrics = [roc(cos_dist, cos_prediction, t) for t in thresh_range]

    fm, fnm, tpr, fpr = zip(*metrics)

    ## Prepare and plot ROC curve
    print("Preparing the ROC curve...\n")
    plt.figure()
    plt.plot(fm, fnm, linewidth=2, color='blue', label='ROC Curve')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('Receiver Operating Characteristic (Cosine Similarity)')
    plt.legend(loc='lower right')
    plt.show()



if __name__ == "__main__":
    iris_recognition()
