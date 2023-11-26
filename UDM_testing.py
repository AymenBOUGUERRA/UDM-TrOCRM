import os
import sys
import patoolib
import random



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import tensorflow as tf


rar_file_path = 'data_unet_grey_interline_noise.rar'
output_folder = 'U-net_data/'

# Check if the output folder or any specific file exists
if not os.path.exists(output_folder) or not os.path.exists(os.path.join(output_folder, 'train/images_train/0.png')):
    # Extraction is not done, so perform it
    patoolib.extract_archive(rar_file_path, outdir=output_folder)
else:
    # Extraction is already done
    print("Extraction has already been performed.")



# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

TEST_PATH = 'U-net_data/test/images_test/'
TEST_PATH_masks = 'U-net_data/test/masks_test/'
# Directory path
dir_path = ''

# Function to load and preprocess images

def load_and_preprocess_images(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    return img

# Function to load and preprocess masks using OpenCV
def load_and_preprocess_masks(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    return mask

# Example usage
image_example = load_and_preprocess_images(os.path.join(TEST_PATH, '0.png'))
mask_example = load_and_preprocess_masks(os.path.join(TEST_PATH_masks, '0.png'))

# Example to show the loaded images
plt.subplot(1, 2, 1)
plt.imshow(image_example, cmap='gray')
plt.title('Example Image')

plt.subplot(1, 2, 2)
plt.imshow(mask_example, cmap='gray')
plt.title('Example Mask')

plt.show()


# Get train and test IDs

test_ids = next(os.walk(TEST_PATH))[2]
print("Number of test images: ", len(test_ids))


# Get and resize train images and masks
# Function to load and preprocess training data
# Function to read and preprocess images using OpenCV
def read_and_preprocess_image(image_path):
    img = cv2.imread(os.path.join(dir_path, image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    return img

# Function to read and preprocess masks using OpenCV
def read_and_preprocess_mask(mask_path):
    mask = cv2.imread(os.path.join(dir_path, mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    _, mask = cv2.threshold(mask, 227, 255, cv2.THRESH_BINARY)
    return mask
def load_and_preprocess_test_data(test_ids):
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    print('Getting and resizing test images ... ')
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = os.path.join(TEST_PATH, id_)

        # Read and preprocess test images
        img = read_and_preprocess_image(os.path.join(dir_path, path))
        X_test[n] = img

        # Read and preprocess masks
        mask_path = os.path.join(TEST_PATH_masks, id_)
        mask = read_and_preprocess_mask(mask_path)
        Y_test[n] = mask

    print('Done!')
    return X_test, Y_test

# Example usage

X_test, Y_test = load_and_preprocess_test_data(test_ids)


model = tf.keras.models.load_model('my_models_savedmodel')

# Predict on train, val, and test

preds_test = model.predict(X_test, verbose=1)
# Threshold predictions
preds_test_t = (preds_test > 0.5).astype(np.uint8)




# Choose 7 random indices
random_indices = random.sample(range(len(preds_test_t)), 7)

# Display input, ground truth, and predicted masks for 10 random samples
for ix in random_indices:
    plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    # Display input image
    plt.subplot(3, 3, 1)
    plt.imshow(X_test[ix], cmap='gray')
    plt.title('Input Image')

    # Display ground truth mask
    plt.subplot(3, 3, 2)
    plt.imshow(np.squeeze(Y_test[ix]), cmap='gray')
    plt.title('Ground Truth Mask')

    # Display predicted mask
    plt.subplot(3, 3, 3)
    plt.imshow(np.squeeze(preds_test_t[ix]), cmap='gray')
    plt.title('Predicted Mask')


    plt.subplot(3, 3, 4)
    plt.imshow(X_test[ix+1], cmap='gray')
    plt.title('Input Image')

    # Display ground truth mask
    plt.subplot(3, 3, 5)
    plt.imshow(np.squeeze(Y_test[ix+1]), cmap='gray')
    plt.title('Ground Truth Mask')

    # Display predicted mask
    plt.subplot(3, 3, 6)
    plt.imshow(np.squeeze(preds_test_t[ix+1]), cmap='gray')
    plt.title('Predicted Mask')



    plt.subplot(3, 3, 7)
    plt.imshow(X_test[ix+2], cmap='gray')
    plt.title('Input Image')

    # Display ground truth mask
    plt.subplot(3, 3, 8)
    plt.imshow(np.squeeze(Y_test[ix+2]), cmap='gray')
    plt.title('Ground Truth Mask')

    # Display predicted mask
    plt.subplot(3, 3, 9)
    plt.imshow(np.squeeze(preds_test_t[ix+2]), cmap='gray')
    plt.title('Model output')
    plt.show()

