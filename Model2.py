import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset2 import ESC50Dataset2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# def audio_to_melspectrogram(file_path, sr=22050, n_mels=128, hop_length=512, n_fft=2048):
#     audio, _ = librosa.load(file_path, sr=sr)
#     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     return mel_spec_db

# def visualize_spectogram():
#     spec = audio_to_melspectrogram("data/1-137-A-32.wav")
#     librosa.display.specshow(spec, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title("Mel Spectrogram")
#     plt.show()

# if __name__ == "__main__":
#     visualize_spectogram()

train_dataset = ESC50Dataset(csv_path="meta/esc50.csv", audio_dir="data", folds=[1, 2, 3, 4])
test_dataset = ESC50Dataset(csv_path="meta/esc50.csv", audio_dir="data", folds=[5])

class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d (32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = AudioCNN(num_classes=50)
criterion = nn.CrossEntropyLoss()         
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for spectrograms, labels in train_loader:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")


