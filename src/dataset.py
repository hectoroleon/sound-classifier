import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

# Loads samples from selected folds (we’ll use this for train/test splits).
# Pads or truncates audio to a fixed 5 seconds (to keep consistent input size).
# Converts each file to a normalized mel-spectrogram tensor.
# Returns (mel_tensor, label_tensor) ready for training.

class ESC50Dataset(Dataset):
    def __init__(self, csv_path, audio_dir, folds=[1], sr=22050, n_mels=128, duration=5.0):
        self.meta = pd.read_csv(csv_path)
        self.meta = self.meta[self.meta['fold'].isin(folds)].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sample_rate = sr
        self.n_mels = n_mels
        self.duration = duration
        self.num_samples = int(sr * duration)

        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.meta['category'].unique()))}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        label = self.label_to_idx[row['category']]

        # Load audio (ensuring fixed length)
        y, _ = librosa.load(file_path, sr=self.sample_rate)
        if len(y) < self.num_samples:
            padding = self.num_samples - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        else:
            y = y[:self.num_samples]

        # Convert to mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        # Add channel dimension and convert to tensor
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()  # shape: [1, n_mels, time]
        label_tensor = torch.tensor(label).long()

        return mel_tensor, label_tensor


class ESC50ResNetDataset(Dataset):
    def __init__(self, csv_path, audio_dir, folds=[1], sr=22050, n_mels=128, duration=5.0):
        self.meta = pd.read_csv(csv_path)
        self.meta = self.meta[self.meta['fold'].isin(folds)].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sample_rate = sr
        self.n_mels = n_mels
        self.duration = duration
        self.num_samples = int(sr * duration)

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.meta['category'].unique()))}

        # Transforms to convert to 3-channel image and resize for ResNet
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x.expand(3, -1, -1)),  # duplicate to 3 channels
            T.Normalize(mean=[0.5], std=[0.5])  # basic normalization
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        label = self.label_to_idx[row['category']]

        y, _ = librosa.load(file_path, sr=self.sample_rate)
        if len(y) < self.num_samples:
            padding = self.num_samples - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        else:
            y = y[:self.num_samples]

        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        mel_image = Image.fromarray(np.uint8(mel_norm * 255))

        image_tensor = self.transform(mel_image)
        label_tensor = torch.tensor(label).long()

        return image_tensor, label_tensor
