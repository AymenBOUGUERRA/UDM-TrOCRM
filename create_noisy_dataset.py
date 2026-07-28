import random
import os

import cv2

from udm_common import resize_and_pad

# Assign input directory
input_directory = './images_dataset_creation/input_images'
output_directory_noisy = './images_dataset_creation/noisy'
output_directory_clean = './images_dataset_creation/clean'

def add_noise(img):
    """
    Adds modified salt and pepper noise to the image.
    
    Args:
        img (numpy.ndarray): Input image.
    
    Returns:
        numpy.ndarray: Image with noise added.
    """
    row, col = img.shape
    number_of_pixels = random.randint(300, 10000)
    for _ in range(number_of_pixels):
        y_coord = random.randint(0, row - 2)
        x_coord = random.randint(0, col - 2)
        img[y_coord][x_coord] = 0
        img[y_coord+1][x_coord+1] = 0
        img[y_coord+1][x_coord] = 0
        img[y_coord][x_coord+1] = 0
    return img

def process_image(img, grid, filename, index):
    """
    Applies a random grid to the image, adds noise, and saves both the noisy and clean versions.
    
    Args:
        img (numpy.ndarray): Input image.
        grid (numpy.ndarray): Grid to apply.
        filename (str): Original filename of the image.
        index (int): Index for naming the output files.
    """
    img = resize_and_pad(img, (540, 540), 255)
    grid = resize_and_pad(grid, (540, 540), 255)
    
    blend = cv2.addWeighted(img, 0.5, grid, 0.5, 0.0)
    shadow = random.randint(120, 255)
    _, black_and_white_image = cv2.threshold(blend, 240, 255, cv2.THRESH_BINARY)
    black_and_white_image[black_and_white_image == 255] = shadow
    black_and_white_image = add_noise(black_and_white_image)
    
    _, black_and_white_image_original = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(f'{output_directory_noisy}/{index}.png', black_and_white_image)
    cv2.imwrite(f'{output_directory_clean}/{index}.png', black_and_white_image_original)
    print(f"Processed {filename} and saved as {index}.png")

os.makedirs(output_directory_noisy, exist_ok=True)
os.makedirs(output_directory_clean, exist_ok=True)

# Iterate over files in the input directory
index = 0
for filename in os.listdir(input_directory):
    rand = random.randint(2, 5)
    filepath = os.path.join(input_directory, filename)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    if rand == 2:
        grid = cv2.imread('images_dataset_creation/French_grid.png', cv2.IMREAD_GRAYSCALE)
    elif rand == 3:
        grid = cv2.imread('images_dataset_creation/lines_grid.png', cv2.IMREAD_GRAYSCALE)
    elif rand == 4:
        grid = cv2.imread('images_dataset_creation/standard_grid.png', cv2.IMREAD_GRAYSCALE)
    else:
        grid = cv2.imread('images_dataset_creation/no_grid.png', cv2.IMREAD_GRAYSCALE)
    
    process_image(img, grid, filename, index)
    index += 1
