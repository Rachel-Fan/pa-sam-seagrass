import cv2
import numpy as np
import os

# Define the input and output directories, change your path HERE
input_folder = '/home/geofly/pa-sam-Hao/data/Alaska/Alaska/valid/image'
output_folder = '/home/geofly/pa-sam-Hao/groundtruth/AKgt'
os.makedirs(output_folder, exist_ok=True)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    
    if filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        
        # Construct the output path for the ground truth image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary_image)
        
        print(f'Ground truth image saved to {output_path}')
