# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:29:52 2024

@author: 11936
"""

import numpy as np
import cv2

def feature_extraction(enhanced_image):
    # Extract features using a bank of Gabor filters in multiple directions and scales
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Different orientations (0, 45, 90, 135 degrees)
    scales = [4.0, 8.0, 16.0]  # Different scales (frequencies)
    filtered_images = []

    for theta in orientations:
        for sigma in scales:
            gabor_kernel = cv2.getGaborKernel((19, 19), sigma, theta, 10.0, 0.4, 0, ktype=cv2.CV_32F)
            filtered_image = cv2.filter2D(enhanced_image, cv2.CV_8UC3, gabor_kernel)
            filtered_images.append(filtered_image)

    # Divide the filtered images into small blocks and calculate the mean and standard deviation for each block
    block_size = 8
    features = []
    for filtered_image in filtered_images:
        height, width = filtered_image.shape
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = filtered_image[y:y + block_size, x:x + block_size]
                if block.size == 0:
                    continue
                mean = np.mean(block)
                std_dev = np.std(block)
                features.extend([mean, std_dev])

    features = np.array(features)
    return features