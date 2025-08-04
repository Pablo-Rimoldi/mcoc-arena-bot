from config.settings import config as Config
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import re
from typing import List, Tuple, Dict
import json
import sys


class MCOCDataset(Dataset):
    def __init__(self, file_paths: List[str], sequence_length: int = 10, transform=None):
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = self._create_sequences()

    def _create_sequences(self) -> List[Tuple[List[str], int]]:
        """Crea sequenze di frame con le relative label"""
        sequences = []

        # Raggruppa i file per sequenze consecutive
        for i in range(len(self.file_paths) - self.sequence_length + 1):
            sequence_files = self.file_paths[i:i + self.sequence_length]

            # Estrai la label dal nome del file (ultimo frame della sequenza)
            last_file = sequence_files[-1]
            label = self._extract_label(last_file)

            if label is not None:
                sequences.append((sequence_files, label))

        return sequences

    def _extract_label(self, filename: str) -> int:
        """Estrae la label dal nome del file"""
        # Estrai la parte dopo l'underscore
        match = re.search(r'_([^.]+)\.png$', filename)
        if match:
            label_str = match.group(1)
            return Config.LABEL_MAPPING.get(label_str)
        return None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_files, label = self.sequences[idx]

        # Carica le immagini della sequenza
        images = []
        for file_path in sequence_files:
            img = Image.open(file_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Stack delle immagini in un tensor (T, C, H, W)
        images_tensor = torch.stack(images)

        return {
            'images': images_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Model architecture


class MCOCActionPredictor(nn.Module):
    def __init__(self, feature_dim=256, hidden_size=128, num_classes=5, dropout_rate=0.3):
        super(MCOCActionPredictor, self).__init__()

        # CNN Backbone (MobileNetV2 preaddestrato)
        self.cnn = models.mobilenet_v2(pretrained=True)

        # Freeze i primi layer per risparmiare memoria
        for param in self.cnn.features[:10].parameters():
            param.requires_grad = False

        # Rimuovi il classificatore originale di MobileNetV2
        self.cnn.classifier = nn.Identity()

        # Feature extractor per ottenere feature_dim dimensioni
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, feature_dim),  # 1280 è l'output di MobileNetV2
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # RNN (GRU) per la sequenza temporale
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if 2 > 1 else 0
        )

        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape

        # Applica CNN a ogni frame della sequenza
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (batch_size, c, h, w)
            # Prima passa attraverso le features di MobileNetV2
            features = self.cnn.features(frame)  # (batch_size, 1280, H, W)
            # Poi applica il feature extractor
            features = self.feature_extractor(
                features)  # (batch_size, feature_dim)
            cnn_features.append(features)

        # Stack delle feature in sequenza temporale
        # (batch_size, seq_len, feature_dim)
        sequence_features = torch.stack(cnn_features, dim=1)

        # Passa attraverso GRU
        # (batch_size, seq_len, hidden_size)
        gru_output, _ = self.gru(sequence_features)

        # Usa solo l'output dell'ultimo timestep per la predizione
        final_output = gru_output[:, -1, :]  # (batch_size, hidden_size)

        # Classificazione finale
        logits = self.classifier(final_output)  # (batch_size, 5)

        return logits

# Training functions


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch['images'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), correct / total, all_predictions, all_labels


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Load and organize data
    print("Loading data...")
    all_files = []
    for filename in os.listdir(Config.DATA_DIR):
        if filename.endswith('.png'):
            file_path = os.path.join(Config.DATA_DIR, filename)
            all_files.append(file_path)

    # Sort files by number to maintain temporal order
    all_files = [f for f in all_files if re.search(
        r'\d+', os.path.basename(f))]

    # Sort files by the first number found
    all_files.sort(key=lambda x: int(
        re.search(r'\d+', os.path.basename(x)).group(0)))

    print(f"Total files found: {len(all_files)}")

    # Split data
    train_files, test_files = train_test_split(
        all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(
        train_files, test_size=0.2, random_state=42)

    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    # Create datasets
    train_dataset = MCOCDataset(train_files, Config.SEQUENCE_LENGTH, transform)
    val_dataset = MCOCDataset(val_files, Config.SEQUENCE_LENGTH, transform)
    test_dataset = MCOCDataset(test_files, Config.SEQUENCE_LENGTH, transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model
    model = MCOCActionPredictor(
        feature_dim=Config.FEATURE_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    print("Starting training...")
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(Config.NUM_EPOCHS):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': Config.__dict__
            }, os.path.join(Config.MODEL_SAVE_DIR, 'best_model.pth'))

        # Log progress
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'training_curves.png'))
    plt.close()

    # Test evaluation
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(
        Config.MODEL_SAVE_DIR, 'best_model.pth'))['model_state_dict'])
    test_loss, test_acc, test_predictions, test_labels = validate_epoch(
        model, test_loader, criterion, device)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions,
                                target_names=list(Config.LABEL_MAPPING.keys())))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(Config.LABEL_MAPPING.keys()),
                yticklabels=list(Config.LABEL_MAPPING.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()

    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,
        'config': Config.__dict__,
        'label_mapping': Config.LABEL_MAPPING
    }

    with open(os.path.join(Config.RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Training completed! Results saved in {Config.RESULTS_DIR}")
    print(f"Best model saved in {Config.MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
