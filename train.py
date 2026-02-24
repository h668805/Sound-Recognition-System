import torch
from torch.utils.data import DataLoader
from dataset import ESC50Dataset
from model import SoundCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ESC50Dataset(
    csv_path="data/ESC-50/meta/esc50.csv",
    audio_dir="data/ESC-50/audio",
    folds=[1,2,3,4]
)

test_dataset = ESC50Dataset(
    csv_path="data/ESC-50/meta/esc50.csv",
    audio_dir="data/ESC-50/audio",
    folds=[5]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model = SoundCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(15):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")