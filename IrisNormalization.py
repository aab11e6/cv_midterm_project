# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:28:52 2024

@author: 11936
"""
import matplotlib.pyplot as plt
import numpy as np

def iris_normalization(image, inner_boundary, outer_boundary):
    # Convert the iris region from Cartesian coordinates to polar coordinates
    # Calculate the normalized iris image by unwrapping the circular iris into a rectangular strip
    center_x, center_y = inner_boundary[0], inner_boundary[1]
    radius_inner = inner_boundary[2]
    radius_outer = outer_boundary[2]

    # Define the output size of the normalized image
    output_height = 64  # Height of the normalized image (number of radial samples)
    output_width = 360  # Width of the normalized image (number of angular samples)

    # Create a linear space for the radius and angle
    r = np.linspace(radius_inner, radius_outer, output_height)
    theta = np.linspace(0, 2 * np.pi, output_width)
    
    # Create a meshgrid for polar coordinates
    r, theta = np.meshgrid(r, theta)
    
    # Convert polar coordinates to Cartesian coordinates
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    
    # Map the Cartesian coordinates to the original image
    x = np.clip(x, 0, image.shape[1] - 1).astype(int)
    y = np.clip(y, 0, image.shape[0] - 1).astype(int)

    # Generate the normalized image by sampling the original image at the calculated coordinates
    normalized_image = image[y, x]
    
    normalized_image = normalized_image[:240,:32]
    
    # plt.imshow(normalized_image, cmap='gray')
    # plt.title("Normalized Iris")
    # plt.axis('off')
    # plt.show()
    
    return normalized_image