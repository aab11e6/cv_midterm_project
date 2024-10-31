# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:08:35 2024

@author: 11936
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def iris_localization(image):
    # Convert to grayscale image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # 调试用
    
    # Use Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 2)
    
    # Step 1: Detect the inner boundary (pupil region)
    
    
    circles = None
    param1 = 120
    param2 = 60
    while param2 > 0 and circles is None:
        param1 = 120
        while param1 > 20 and circles is None:
            circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                                       param1=param1, param2=param2, minRadius=20, maxRadius=95)
            param1 -= 5
        param2 -= 5
    
    # circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
    #                            param1=100, param2=40, minRadius=20, maxRadius=100)
    
    if circles is not None:
        circles = np.around(circles)[0].astype(int)
        inner_circle = min(circles, key=lambda c: c[2])  # Select the first detected circle as the inner boundary
        inner_boundary = (inner_circle[0], inner_circle[1], inner_circle[2])  # (x, y, r)
        
        # Draw the inner boundary circle on the image
        # cv2.circle(image, (inner_boundary[0], inner_boundary[1]), inner_boundary[2], (255, 0, 0), 2) # 调试用
    else:
        raise ValueError("Failed to detect the inner boundary of the iris, please check the image quality")
    
    # Step 2: Detect the outer boundary (iris region)
    min_radius_for_outer = int(inner_boundary[2] * 1.5)  # Set minimum radius based on inner circle's radius
    
    circles = None
    param1 = 120
    filtered_circles = []
    while param1 > 0 and (circles is None or not filtered_circles):
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=3,
                                   param1=param1, param2=20, minRadius=min_radius_for_outer, maxRadius=150)
        param1 -= 5
    
    
    # circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=3,
    #                            param1=30, param2=20, minRadius=min_radius_for_outer, maxRadius=200)
    
        if circles is not None:
            circles = np.around(circles)[0].astype(int)
            
            # Filter circles based on distance to inner circle and select the one with the largest radius
            center_diff_range = 10
            filtered_circles = [c for c in circles if np.sqrt((c[0] - inner_boundary[0])**2 + (c[1] - inner_boundary[1])**2) < center_diff_range]
            if filtered_circles:
                outer_circle = max(filtered_circles, key=lambda c: c[2])
                outer_boundary = (outer_circle[0], outer_circle[1], outer_circle[2])  # (x, y, r)
                
                # Draw the outer boundary circle on the image
                # cv2.circle(image, (outer_boundary[0], outer_boundary[1]), outer_boundary[2], (0, 255, 0), 2) # 调试用
            # else:
            #     raise ValueError("Failed to find a suitable outer boundary of the iris, please check the image quality")
    # if not filtered_circles:
    #     raise ValueError("Failed to detect the outer boundary of the iris, please check the image quality")
    
    # # Display the image using matplotlib to show in Spyder's plot pane # 调试用
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title("Inner and Outer Boundaries")
    # plt.axis('off')
    # plt.show()
    
    return inner_boundary, outer_boundary