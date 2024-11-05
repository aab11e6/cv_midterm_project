# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:08:35 2024

@author: 11936
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def iris_localization(person_id, image):
    # Convert to grayscale image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # 调试用
    
    
    # Use Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 2)
    
    # Apply binary thresholding to create a binary image
    _, binary_image = cv2.threshold(blurred_image, 65, 255, cv2.THRESH_BINARY)
    
    ## Step 1: Estimate pupil center using vertical and horizontal projection
    vertical_projection = np.sum(binary_image, axis=1)
    horizontal_projection = np.sum(binary_image, axis=0)
    center_y = np.argmin(vertical_projection) + 10 # add 10 to adjust the y-coordinate to eliminate the influence of eyelash
    center_x = np.argmin(horizontal_projection)
    # cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)  # Draw red dot at estimated center # 调试用
    
    ## Step 2: Detect the inner boundary (pupil region)
    
    # Initialize circles as None
    circles = None

    # Initial value for param1 for Hough Circle detection
    param1 = 200 
    param2 = 80

    # Initialize a list to store filtered circles
    filtered_circles = []

    # Try to detect circles with decreasing thresholds until an inner boundary is found
    while param2 > 0 and (circles is None or not filtered_circles):
        param1 = 120
        while param1 > 100 and (circles is None or not filtered_circles):
            # Detect circles using Hough Circle Transform
            circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                                        param1=param1, param2=param2, minRadius=20, maxRadius=95)
            
            # If circles are detected, round the values and convert to integers
            if circles is not None:
                circles = np.around(circles)[0].astype(int)
                
                # Filter circles based on distance to inner circle and select the one with the largest radius
                center_diff_range = 14 # Define the acceptable distance range
                filtered_circles = [c for c in circles if np.sqrt((c[0] - center_x)**2 + (c[1] - center_y)**2) < center_diff_range]
                
                # If any circles pass the filter, select the one with the smallest radius as the inner boundary
                if filtered_circles:
                    inner_circle = min(filtered_circles, key=lambda c: c[2])  # Select the first detected circle as the inner boundary
                    inner_boundary = (inner_circle[0], inner_circle[1], inner_circle[2])  # (x, y, r)
                    
                    # cv2.circle(image, (inner_boundary[0], inner_boundary[1]), inner_boundary[2], (255, 0, 0), 2) # 调试用
            
            param1 -= 5 # If no circle detected, decrease params
        param2 -= 5
    
    # Step 3: Detect the outer boundary (iris region)

    # Set minimum radius of outer boundary based on inner circle's radius
    min_radius_for_outer = int(inner_boundary[2] * 1.8)  
    
    circles = None # Re-initialize circles as None
    param1 = 120 # Reset param1 for Hough Circle detection
    filtered_circles = [] # Re-initialize filtered circles list

    # Try to detect circles for the outer boundary with decreasing param1 until a circle is found
    while param1 > 0 and (circles is None or not filtered_circles):
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=3,
                                   param1=param1, param2=20, minRadius=min_radius_for_outer, maxRadius=150)
        param1 -= 5
    
        # If circles are detected, round the values and convert to integers
        if circles is not None:
            circles = np.around(circles)[0].astype(int)
            
            # Filter circles based on distance to inner circle and select the one with the largest radius
            center_diff_range = 5
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
    # plt.title(f"{person_id} Inner and Outer Boundaries")
    # plt.axis('off')
    # plt.show()
    
    return inner_boundary, outer_boundary