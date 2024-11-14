# -*- coding: utf-8 -*-
"""wav2vec2.ipynb

## Data Preprocessing
"""

import pandas as pd
import librosa
import torch
import torchaudio
import IPython.display as ipd
import numpy as np
import os
import re
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, Wav2Vec2Processor

# Initialize processor and model
model_name_or_path = "facebook/wav2vec2-large-960h"
pooling_mode = "mean"

label_list = ["spoof", "bona-fide'"]
num_labels = 2

# Config setup for Wav2Vec2
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

# Processor for feature extraction and normalization
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

# Load metadata
metadata = pd.read_csv('./ProjectStorage/meta.csv')

max_duration = 5  # Maximum audio duration in seconds

# Function to load and process audio files
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()

    # Determine max length based on target sampling rate
    max_length = int(target_sampling_rate * max_duration)

    # List to hold segmented audio
    segments = []

    # Segment audio if longer than max_length
    if len(speech) > max_length:
        num_segments = len(speech) // max_length
        for i in range(num_segments + 1):
            start = i * max_length
            end = start + max_length
            if end <= len(speech):
                segment = speech[start:end]
            else:
                segment = speech[start:]  # Last segment may be shorter

            # Pad or truncate to max_length
            if len(segment) < max_length:
                segment = np.pad(segment, (0, max_length - len(segment)), 'constant')

            segments.append(segment)
    else:
        # Pad or truncate if it's shorter than max_length
        speech = np.pad(speech, (0, max_length - len(speech)), 'constant')
        segments.append(speech)

    # Create attention masks for each segment
    attention_masks = []
    for segment in segments:
        attention_mask = np.zeros(max_length, dtype=np.float32)
        attention_mask[:len(segment)] = 1
        attention_masks.append(attention_mask)

    return segments, attention_masks

def label_to_id(label, label_list):
    return label_list.index(label) if label in label_list else -1

def load_audio_files_from_wav(metadata, wav_folder):
    audio_data = []
    label_dict = metadata.set_index('file')['label'].to_dict()  # Create a dictionary to map file names to labels

    print("Loading audio files and matching with labels from metadata...")
    # Loop through all .wav files in the directory
    for wav_file in tqdm(os.listdir(wav_folder), desc="Processing .wav files"):
        if wav_file.endswith('.wav'):
            audio_path = os.path.join(wav_folder, wav_file)
            audio_segments, attention_masks = speech_file_to_array_fn(audio_path)

            # Extract numerical part from the file name for label lookup
            numerical_part = re.search(r'\d+', wav_file).group()
            target_file = f"{numerical_part}.wav"

            # Get label from the metadata dictionary
            label = label_dict.get(target_file)
            if label is not None:
                for segment, attention_mask in zip(audio_segments, attention_masks):
                    audio_data.append((segment, attention_mask, label))
            else:
                print(f"Warning: Label not found for {wav_file}")

    print(f"Loaded {len(audio_data)} audio segments with labels successfully.")
    return audio_data

print("Train Data:")
audio_rw_train = load_audio_files_from_wav(metadata, './ProjectStorage/wav_train_set')
print("")

print("Validation Data:")
audio_rw_val = load_audio_files_from_wav(metadata, './ProjectStorage/wav_val_set')
print("")

print("Test Data:")
audio_rw_test = load_audio_files_from_wav(metadata, './ProjectStorage/wav_test_set')
print("")


def create_audio_dataframe(audio_data):
    # Convert audio data to a DataFrame
    audio_df = pd.DataFrame(audio_data, columns=['audio', 'attention_mask', 'label'])

    # Map labels to numerical values
    label_mapping = {'spoof': 0, 'bona-fide': 1}
    audio_df['label'] = audio_df['label'].map(label_mapping)

    return audio_df

audio_df_rw_train = create_audio_dataframe(audio_rw_train)
audio_df_rw_val = create_audio_dataframe(audio_rw_val)
audio_df_rw_test = create_audio_dataframe(audio_rw_test)

train_df = audio_df_rw_train
val_df = audio_df_rw_val
test_df = audio_df_rw_test

# Reset index
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# Assuming train_df and test_df are your DataFrames with columns: 'input_values', 'labels', 'attention_mask'

# Define a custom Dataset class
class AudioDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Return a dictionary that includes all necessary fields for the data collator
        return {
            'input_values': row['audio'],
            'attention_mask': row['attention_mask'],
            'labels': row['label']
        }

# Create datasets
train_dataset = AudioDataset(train_df, processor)
eval_dataset = AudioDataset(val_df, processor)
test_dataset = AudioDataset(test_df, processor)

idx = 0
print(f"Training input_values: {train_dataset[idx]['input_values']}")
print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
print(f"Training labels: {train_dataset[idx]['labels']}")

