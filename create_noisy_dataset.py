import random
import cv2
import os
import numpy as np

# Assign input directory
input_directory = './images_dataset_creation/input_images'
output_directory_noisy = './images_dataset_creation/noisy'
output_directory_clean = './images_dataset_creation/clean'

def resize_and_pad(img, size, pad_color):
    """
    Resizes the image to the desired size and pads the missing pixels to respect the initial aspect ratio.
    
    Args:
        img (numpy.ndarray): Input image.
        size (tuple): Desired size (width, height).
        pad_color (int or tuple): Padding color, 0 to 255 in grayscale or a tuple for color images.
    
    Returns:
        numpy.ndarray: Resized and padded image.
    """
    h, w = img.shape[:2]
    sh, sw = size

    # Choose interpolation method
    interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC

    aspect = float(w) / h
    saspect = float(sw) / sh

    if saspect > aspect or (saspect == 1 and aspect <= 1):
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color] * 3

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img

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
