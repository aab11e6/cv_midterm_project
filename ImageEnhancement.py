# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:29:36 2024

@author: 11936
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np

def image_enhancement(normalized_image):
    
    # Remove eyelashes by applying a threshold
    thresholded_image = np.where(normalized_image < 75, 255, normalized_image).astype(np.uint8)
    
    # Enhance the normalized iris image using histogram equalization
    enhanced_image = cv2.equalizeHist(thresholded_image)

    return enhanced_image