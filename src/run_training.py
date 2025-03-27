import sys
import os

# Add the project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import torch
from torch.utils.data import DataLoader
from src.dataset import ESC50Dataset
from src.model import AudioCNN
from src.train import train_one_epoch, validate


def main():
    # ✅ Configuration
    batch_size = 16
    epochs = 20
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Paths
    csv_path = 'data/ESC-50-master/meta/esc50.csv'
    audio_dir = 'data/ESC-50-master/audio'

    # ✅ Datasets
    train_dataset = ESC50Dataset(csv_path, audio_dir, folds=[2, 3, 4, 5])
    val_dataset = ESC50Dataset(csv_path, audio_dir, folds=[1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ✅ Model, Loss, Optimizer
    model = AudioCNN(num_classes=50).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ✅ Training loop
    for epoch in range(1, epochs + 1):
        print(f"\n🔁 Epoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"🔍 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ✅ Save model
    torch.save(model.state_dict(), 'C:/Documents/sound-classifier/models/audio_cnn.pth')
    print("🎉 Model saved to models/audio_cnn.pth")

if __name__ == '__main__':
    main()
