import librosa
import librosa.display
import numpy as np 
import matplotlib.pyplot as pl

def audio_to_melspectrogram(file_path):
     #y = waveform, sr = samplerate
    y, sr =librosa.load(file_path, sr=22050)

   
    #Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels =128)

    #Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db
