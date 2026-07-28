"""Helpers shared by the UDM / TrOCRM scripts."""

import os

import cv2
import numpy as np

# Keras 3 (TensorFlow >= 2.16) only writes/reads models in the `.keras` format.
UDM_MODEL_PATH = "udm_model.keras"
UDM_CHECKPOINT_PATH = "udm_checkpoint.keras"

# Directory produced by the pre-Keras-3 `model.save("my_models_savedmodel")`,
# which is also what the downloadable pre-trained model contains.
UDM_LEGACY_SAVEDMODEL_PATH = "my_models_savedmodel"

IMG_HEIGHT = 256
IMG_WIDTH = 256


def load_udm_model(path=UDM_MODEL_PATH, legacy_path=UDM_LEGACY_SAVEDMODEL_PATH):
    """Load the U-Net de-noising model.

    Prefers a `.keras` file. Falls back to the legacy TensorFlow SavedModel
    directory, which Keras 3 can no longer open with `load_model` but can still
    run through `TFSMLayer`.
    """
    import tensorflow as tf

    if os.path.exists(path):
        return tf.keras.models.load_model(path)

    if os.path.isdir(legacy_path):
        print(f"'{path}' not found, loading the legacy SavedModel '{legacy_path}'.")
        inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
        outputs = tf.keras.layers.TFSMLayer(
            legacy_path, call_endpoint="serving_default"
        )(inputs)
        if isinstance(outputs, dict):
            outputs = next(iter(outputs.values()))
        return tf.keras.Model(inputs, outputs)

    raise FileNotFoundError(
        f"No UDM model found: expected '{path}' or the legacy SavedModel "
        f"directory '{legacy_path}'. Train one with "
        "Data_preparation_and_UDM_model_training.py, or download the "
        "pre-trained model linked in the README."
    )


def resize_and_pad(img, size, pad_color):
    """Resize an image to `size` and pad it so the original aspect ratio is kept.

    Args:
        img (numpy.ndarray): Input image.
        size (tuple): Desired size (height, width).
        pad_color (int or tuple): Padding color, 0 to 255 in grayscale or a
            tuple for color images.

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
    return cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )
