import os
import sys
import patoolib
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel, default_data_collator




rar_file_path = 'TrOCRM_clear_data.rar'
output_folder = 'TrOCRM_clear_data/'
rar_file_path_2 = 'TrOCRM_noise_data.rar'
output_folder_2 = 'TrOCRM_noise_data/'

# Check if the output folder or any specific file exists
if not os.path.exists(output_folder) or not os.path.exists(os.path.join(output_folder, 'train/caption.txt')) \
        or not os.path.exists(os.path.join(output_folder_2, 'images_test/caption.txt')):
    # Extraction is not done, so perform it
    patoolib.extract_archive(rar_file_path, outdir=output_folder)
    patoolib.extract_archive(rar_file_path_2, outdir=output_folder_2)
else:
    # Extraction is already done
    print("Extraction has already been performed.")

train_df = pd.read_table(output_folder + '/train/caption.txt', header=None) #fwf
train_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
train_df['file_name']= train_df['file_name'].apply(lambda x: x+'.jpg')
train_df = train_df.dropna()
print(train_df)

test_df = pd.read_table(output_folder + '/2014/caption.txt', header=None) #fwf
test_df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
test_df['file_name']= test_df['file_name'].apply(lambda x: x+'.jpg')
test_df = test_df.dropna()
print(test_df)

test_df_noise = pd.read_table(output_folder_2 + '/images_test/caption.txt', header=None) #fwf
test_df_noise.rename(columns={0: "file_name", 1: "text"}, inplace=True)
test_df_noise['file_name']= test_df_noise['file_name'].apply(lambda x: x+'.jpg')
test_df_noise = test_df_noise.dropna()
print(test_df_noise)


train_df = shuffle(train_df)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)



class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=490):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding





processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir= output_folder + 'train/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir= output_folder + '2014/',
                           df=test_df,
                           processor=processor)

eval_dataset_noise = IAMDataset(root_dir= output_folder_2 + 'images_test/',
                           df=test_df,
                           processor=processor)
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))
encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)

labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 24 # origin 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=3, #origin 8
    per_device_eval_batch_size=16, #origin 8
    fp16=False,
    output_dir="TrOCRM_models/",
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    num_train_epochs = 2,
)


torch.cuda.empty_cache()
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    #compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
#trainer.push_to_hub("checkpoint_eval_2014_small_stage1_num_beams=10/checkpoint-15000")

trainer.train()


