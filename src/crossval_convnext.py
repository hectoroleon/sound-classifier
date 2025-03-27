import sys
import os

# Add the project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import torch
from torch.utils.data import DataLoader
from src.dataset import ESC50ResNetDataset
from src.model_transfer_convnext import ConvNeXtAudioClassifier
from src.train import train_one_epoch, validate

import numpy as np

def main():
    # ✅ Config
    batch_size = 16
    epochs = 5
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = 'data/ESC-50-master/meta/esc50.csv'
    audio_dir = 'data/ESC-50-master/audio'

    fold_accuracies = []

    for val_fold in range(1, 6):
        print(f"\n==============================")
        print(f"📂 Fold {val_fold} — Training on folds ≠ {val_fold}")
        print(f"==============================")

        train_folds = [f for f in range(1, 6) if f != val_fold]

        train_dataset = ESC50ResNetDataset(csv_path, audio_dir, folds=train_folds)
        val_dataset = ESC50ResNetDataset(csv_path, audio_dir, folds=[val_fold])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = ConvNeXtAudioClassifier(num_classes=50).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            print(f"\n🔁 Epoch {epoch}/{epochs}")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"🔍 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        fold_accuracies.append(val_acc)

    # ✅ Final results
    print("\n📊 Cross-Validation Results:")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"Fold {i}: {acc:.4f}")

    print(f"\n🧠 Average Val Accuracy: {np.mean(fold_accuracies):.4f}")

if __name__ == '__main__':
    main()
