import os
import sys
import random
import patoolib




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import tensorflow as tf




Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Dropout = tf.keras.layers.Dropout
Model = tf.keras.models.Model
Conv2D = tf.keras.layers.Conv2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
MaxPooling2D = tf.keras.layers.MaxPooling2D
concatenate = tf.keras.layers.concatenate
Lambda = tf.keras.layers.Lambda
BatchNormalization = tf.keras.layers.BatchNormalization
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint







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
IMG_WIDTH = 512
IMG_HEIGHT = 512
TRAIN_PATH = 'U-net_data/train/images_train/'
TRAIN_PATH_masks = 'U-net_data/train/masks_train/'

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
image_example = load_and_preprocess_images(os.path.join(TRAIN_PATH, '0.png'))
mask_example = load_and_preprocess_masks(os.path.join(TRAIN_PATH_masks, '0.png'))

# Example to show the loaded images
plt.subplot(1, 2, 1)
plt.imshow(image_example, cmap='gray')
plt.title('Example Image')

plt.subplot(1, 2, 2)
plt.imshow(mask_example, cmap='gray')
plt.title('Example Mask')

plt.show()


# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[2]
print("Number of train images: ", len(train_ids))



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
    _, mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
    return mask

# Function to load and preprocess training data
def load_and_preprocess_train_data(train_ids):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        # Read and preprocess images
        img_path = os.path.join(TRAIN_PATH, id_)
        img = read_and_preprocess_image(img_path)
        X_train[n] = img

        # Read and preprocess masks
        mask_path = os.path.join(TRAIN_PATH_masks, id_)
        mask = read_and_preprocess_mask(mask_path)
        Y_train[n] = mask

    return X_train, Y_train


X_train, Y_train = load_and_preprocess_train_data(train_ids)




ix = random.randint(0, len(train_ids))
plt.subplot(1, 2, 1)
plt.imshow(X_train[ix], cmap='gray')
print(X_train[ix].shape)
plt.subplot(1, 2, 2)
plt.imshow(Y_train[ix], cmap='gray')
plt.show()

# Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, 1))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Contraction path
c1 = Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_normal', padding='same') (s)
c1 = BatchNormalization()(c1)
c1 = Dropout(0.07) (c1)
c1 = Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
c1 = BatchNormalization()(c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(64, (6, 6), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
c2 = BatchNormalization()(c2)
c2 = Dropout(0.07) (c2)
c2 = Conv2D(64, (6, 6), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
c3 = BatchNormalization()(c3)
c3 = Dropout(0.07) (c3)
c3 = Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(256, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
c4 = BatchNormalization()(c4)
c4 = Dropout(0.07) (c4)
c4 = Conv2D(256, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(512, (4, 4), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
c5 = BatchNormalization()(c5)
c5 = Dropout(0.07) (c5)
c5 = Conv2D(512, (4, 4), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
c5 = BatchNormalization()(c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
c6 = BatchNormalization()(c6)
c6 = Dropout(0.07) (c6)
c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
c6 = BatchNormalization()(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
c7 = BatchNormalization()(c7)
c7 = Dropout(0.07) (c7)
c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
c7 = BatchNormalization()(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
c8 = BatchNormalization()(c8)
c8 = Dropout(0.07) (c8)
c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
c8 = BatchNormalization()(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
c9 = BatchNormalization()(c9)
c9 = Dropout(0.07) (c9)
c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
c9 = BatchNormalization()(c9)
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)



model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'model_unet_checkpoint_savedmodel',  # Save the best model with this name
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_split=0.07,
    batch_size=4,
    epochs=27,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the entire model in the SavedModel format
model.save("my_models_savedmodel")