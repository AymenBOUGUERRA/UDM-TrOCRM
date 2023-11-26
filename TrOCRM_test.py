import os
import sys
import patoolib
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from transformers import TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel, default_data_collator
import tensorflow as tf
import numpy as np
import re
from itertools import product
# Extract the data if not already done
rar_file_path = 'TrOCRM_clear_data.rar'
output_folder = 'TrOCRM_clear_data/'
rar_file_path_2 = 'TrOCRM_noise_data.rar'
output_folder_2 = 'TrOCRM_noise_data/'

if not os.path.exists(output_folder) or not os.path.exists(os.path.join(output_folder, 'train/caption.txt')) \
        or not os.path.exists(os.path.join(output_folder_2, 'images_test/caption.txt')):
    patoolib.extract_archive(rar_file_path, outdir=output_folder)
    patoolib.extract_archive(rar_file_path_2, outdir=output_folder_2)
else:
    print("Extraction has already been performed.")

# Read and preprocess the data
train_df = pd.read_table(output_folder + '/train/caption.txt', header=None)
train_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
train_df['file_name'] = train_df['file_name'].apply(lambda x: x + '.jpg')
train_df = train_df.dropna()

test_df = pd.read_table(output_folder + '/2014/caption.txt', header=None)
test_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
test_df['file_name'] = test_df['file_name'].apply(lambda x: x + '.jpg')
test_df = test_df.dropna()

test_df_noise = pd.read_table(output_folder_2 + '/images_test/caption.txt', header=None)
test_df_noise.rename(columns={0: "file_name", 1: "text"}, inplace=True)
test_df_noise['file_name'] = test_df_noise['file_name'].apply(lambda x: x + '.jpg')
test_df_noise = test_df_noise.dropna()

# Shuffle and reset indices
train_df = shuffle(train_df)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
test_df_noise.reset_index(drop=True, inplace=True)

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=490):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # Open image using OpenCV
        image = cv2.imread(os.path.join(self.root_dir, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize and normalize image
        image = cv2.resize(image, (224, 224))  # Adjust size as needed
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# Create datasets
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir=output_folder + 'train/', df=train_df, processor=processor)
eval_dataset = IAMDataset(root_dir=output_folder + '2014/', df=test_df, processor=processor)
eval_dataset_noise = IAMDataset(root_dir=output_folder_2 + 'images_test/', df=test_df_noise, processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

# Print information about encoding
encoding = train_dataset[0]
for k, v in encoding.items():
    print(k, v.shape)

# Print decoded label
labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)

# Load and configure TrOCR model
model_TrOCR = VisionEncoderDecoderModel.from_pretrained("TrOCRM_models/checkpoint-6500")
model_TrOCR.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model_TrOCR.config.pad_token_id = processor.tokenizer.pad_token_id
model_TrOCR.config.vocab_size = model_TrOCR.config.decoder.vocab_size

UDM = tf.keras.models.load_model('my_models_savedmodel')
# Function to perform OCR on an image
def ocr_image(src_img):
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model_TrOCR.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def resizeAndPad(img, size, padColor):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    return scaled_img






random_indices = random.sample(range(len(eval_dataset)), 5)

# Create a figure with subplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(17, 15))

for i, ix in enumerate(random_indices):
    # Display input image
    image_path = os.path.join(eval_dataset.root_dir, test_df['file_name'][ix])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[i, 0].imshow(image)
    axes[i, 0].set_title('Input Image')

    # Display model output
    model_output = ocr_image(image)
    axes[i, 1].text(0.5, 0.5, model_output, ha='center', va='center', fontsize=17)
    axes[i, 1].axis('off')
    axes[i, 1].set_title('Model Output')
fig.suptitle('Model prediction for random inputs from the no-noise test set', fontsize=27)
plt.tight_layout()
plt.show()

# Create a figure with subplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(17, 15))

for i, ix in enumerate(random_indices):
    # Display input image
    image_path = os.path.join(eval_dataset_noise.root_dir, test_df_noise['file_name'][ix])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[i, 0].imshow(image)
    axes[i, 0].set_title('Input Image')

    # Display model output
    model_output = ocr_image(image)
    axes[i, 1].text(0.5, 0.5, model_output, ha='center', va='center', fontsize=17)
    axes[i, 1].axis('off')
    axes[i, 1].set_title('Model Output')
fig.suptitle('Model prediction for the same inputs from the with-noise test set', fontsize=27)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(17, 25))

for i, ix in enumerate(random_indices):
    # Display input image
    image_path = os.path.join(eval_dataset_noise.root_dir, test_df_noise['file_name'][ix])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[i, 0].imshow(image)
    axes[i, 0].set_title('Input Image')

    resized_image = resizeAndPad(image, (512, 512), 255)
    image_gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    _, image_black = cv2.threshold(image_gray, 70, 255, cv2.THRESH_BINARY)
    image_array = np.zeros((32, 512, 512), dtype=np.uint8)
    image_array[0] = image_black
    image_array = UDM.predict(image_array, verbose=1)
    image_array_th = (image_array > 0.5).astype(np.uint8)
    axes[i, 1].imshow(image_array_th[0], cmap='gray')
    axes[i, 1].set_title('Input Image cleaned with UDM')

    # Display model output
    image = cv2.cvtColor(image_array_th[0]*255, cv2.COLOR_GRAY2RGB)
    model_output = ocr_image(image)
    axes[i, 2].text(0.5, 0.5, model_output, ha='center', va='center', fontsize=17)
    axes[i, 2].axis('off')
    axes[i, 2].set_title('Model Output')
fig.suptitle('Model prediction for the same inputs from the with-noise test set', fontsize=27)
plt.tight_layout()
plt.show()

