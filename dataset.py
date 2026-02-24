import os
import pandas as np
import torch
from torch.utils.data import Dataset
import librosa
import librosa.display
import numpy as np 
import pandas as pd

class ESC50Dataset(Dataset):
     def __init__(self, csv_path,audio_dir, folds):
          self.data = pd.read_csv(csv_path)
          self.data = self.data[self.data["fold"].isin(folds)]
          self.audi_dir = audio_dir

     def __len__(self):
          return len(self.data)

     def __getitem__(self, idx):
          row = self.data.iloc[idx]

          file_path = os.path.join(self.audio_dir, row["filename"])
          label = row["target"]

          y, sr = librosa.load(file_path, sr=22050)

          y, sr = librosa.load(file_path, sr=22050)

          mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
          mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

          mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0).float()

          return mel_spec_db, label


def audio_to_melspectrogram(file_path):
     #y = waveform, sr = samplerate
    y, sr =librosa.load(file_path, sr=22050)

   
    #Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels =128)

    #Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db
