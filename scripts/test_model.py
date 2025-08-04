from torchvision import transforms
from torch.utils.data import DataLoader
from train_model import MCOCActionPredictor, Config, MCOCDataset
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import re
from collections import defaultdict
import sys
sys.path.append('..')


def analyze_data_distribution():
    """Analizza la distribuzione delle classi nei dati"""
    print("Analizzando la distribuzione dei dati...")

    label_counts = defaultdict(int)
    total_files = 0

    for filename in os.listdir(Config.DATA_DIR):
        if filename.endswith('.png'):
            total_files += 1
            # Estrai label dal nome del file
            match = re.search(r'_([^.]+)\.png$', filename)
            if match:
                label = match.group(1)
                label_counts[label] += 1

    print(f"Totale file: {total_files}")
    print("\nDistribuzione delle classi:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_files) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")

    # Plot distribuzione
    plt.figure(figsize=(10, 6))
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.bar(labels, counts)
    plt.title('Distribuzione delle Classi nei Dati')
    plt.xlabel('Classe')
    plt.ylabel('Numero di Frame')
    plt.xticks(rotation=45)

    for i, count in enumerate(counts):
        plt.text(i, count + max(counts)*0.01, str(count), ha='center')

    plt.tight_layout()
    plt.savefig('results/data_distribution.png')
    plt.close()

    return label_counts


def test_model_performance(model_path="models/best_model.pth"):
    """Testa le performance del modello sui dati di test"""
    if not os.path.exists(model_path):
        print(f"Modello non trovato: {model_path}")
        print("Esegui prima il training con: python scripts/train_model.py")
        return

    print("Caricando il modello...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carica modello
    model = MCOCActionPredictor(
        feature_dim=Config.FEATURE_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepara dati di test
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Carica tutti i file
    all_files = []
    for filename in os.listdir(Config.DATA_DIR):
        if filename.endswith('.png'):
            file_path = os.path.join(Config.DATA_DIR, filename)
            all_files.append(file_path)

    all_files.sort(key=lambda x: int(
        re.search(r'(\d+)', os.path.basename(x)).group(1)))

    # Usa gli ultimi 20% per il test
    test_files = all_files[-len(all_files)//5:]

    test_dataset = MCOCDataset(test_files, Config.SEQUENCE_LENGTH, transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Testando su {len(test_dataset)} sequenze...")

    # Test
    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            confidence = torch.max(probabilities, 1)[0]

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

    # Calcola metriche
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    print(f"\nAccuracy complessiva: {accuracy:.4f}")
    print(f"Confidence media: {np.mean(all_confidences):.4f}")

    # Classification report
    print("\nClassification Report:")
    target_names = list(Config.LABEL_MAPPING.keys())
    print(classification_report(all_labels,
          all_predictions, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/test_confusion_matrix.png')
    plt.close()

    # Analisi per classe
    print("\nAnalisi per classe:")
    for i, class_name in enumerate(target_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_predictions)[class_mask] == i)
            class_conf = np.mean(np.array(all_confidences)[class_mask])
            print(
                f"{class_name}: Accuracy={class_acc:.3f}, Avg Confidence={class_conf:.3f}")

    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences
    }


def analyze_sequence_predictions(model_path="models/best_model.pth"):
    """Analizza le predizioni su sequenze temporali"""
    if not os.path.exists(model_path):
        print("Modello non trovato. Esegui prima il training.")
        return

    print("Analizzando predizioni su sequenze...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carica modello
    model = MCOCActionPredictor(
        feature_dim=Config.FEATURE_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepara dati
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Carica file
    all_files = []
    for filename in os.listdir(Config.DATA_DIR):
        if filename.endswith('.png'):
            file_path = os.path.join(Config.DATA_DIR, filename)
            all_files.append(file_path)

    all_files.sort(key=lambda x: int(
        re.search(r'(\d+)', os.path.basename(x)).group(1)))

    # Testa su una sequenza lunga
    test_files = all_files[-100:]  # Ultimi 100 frame
    test_dataset = MCOCDataset(test_files, Config.SEQUENCE_LENGTH, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    confidences = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            pred = torch.argmax(outputs, 1).item()
            conf = torch.max(probabilities, 1)[0].item()

            predictions.append(pred)
            confidences.append(conf)

    # Plot sequenza temporale
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(predictions, 'b-', label='Predicted Action')
    plt.title('Predizioni Temporali')
    plt.ylabel('Action Class')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(confidences, 'r-', label='Confidence')
    plt.title('Confidence nel Tempo')
    plt.xlabel('Frame Sequence')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/temporal_predictions.png')
    plt.close()

    print(f"Analisi temporale completata. Predizioni salvate in results/temporal_predictions.png")


def main():
    print("=== MCOC Model Testing ===\n")

    # 1. Analizza distribuzione dati
    print("1. Analisi distribuzione dati...")
    analyze_data_distribution()

    # 2. Test performance modello
    print("\n2. Test performance modello...")
    results = test_model_performance()

    # 3. Analisi sequenze temporali
    print("\n3. Analisi predizioni temporali...")
    analyze_sequence_predictions()

    print("\n=== Test completato! ===")
    print("Risultati salvati in:")
    print("- results/data_distribution.png")
    print("- results/test_confusion_matrix.png")
    print("- results/temporal_predictions.png")


if __name__ == "__main__":
    main()
