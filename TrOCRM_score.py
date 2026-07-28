import os

import patoolib
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
import jiwer

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
model_TrOCR.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
model_TrOCR.generation_config.pad_token_id = processor.tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_TrOCR.to(device)


def evaluate(dataset, name):
    """Compute the Exact Match rate and the Character Error Rate on a dataset.

    The CER comes from jiwer, called directly: `datasets.load_metric` was
    removed from the `datasets` library.
    """
    dataloader = DataLoader(dataset, batch_size=1)

    predictions = []
    references = []

    print(f"Running evaluation on the {name} test set...")
    for batch in tqdm(dataloader):
        # predict using generate
        pixel_values = batch["pixel_values"].to(device)
        outputs = model_TrOCR.generate(pixel_values)
        # decode
        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(pred_str)
        references.extend(label_str)

    exact_match = sum(p == r for p, r in zip(predictions, references)) / len(references)

    # Samples with an empty ground truth carry no character to score, and would
    # otherwise make the CER blow up, so they are left out of it.
    scored = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    skipped = len(references) - len(scored)
    if skipped:
        print(f"{skipped} sample(s) with an empty ground truth excluded from the CER.")
    cer = jiwer.cer([r for _, r in scored], [p for p, _ in scored])

    print(f"Character error rate on {name} test set:", cer)
    print(f"Exact match rate (Exp Rate) on {name} test set:", exact_match)
    return cer, exact_match


evaluate(
    IAMDataset(root_dir=output_folder + '2014/', df=test_df, processor=processor),
    "clear",
)

evaluate(
    IAMDataset(root_dir=output_folder_2 + 'images_test/', df=test_df_noise, processor=processor),
    "noised",
)
