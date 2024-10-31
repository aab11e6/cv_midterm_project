# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:30:11 2024

@author: 11936
"""

def performance_evaluation(match_results, metric):
    # Define the ground truth
    ground_truth = [i // 4 + 1 for i in range(432)]
    
    # Calculate the correct matches
    correct_matches = 0
    for i, result in enumerate(match_results):
        expected_label = ground_truth[i]
        if result == expected_label:
            correct_matches += 1
    
    # Calculate and print the accuracy
    accuracy = (correct_matches / len(match_results)) * 100
    print(f"{metric} Accuracy: {accuracy:.2f}%")