from train_model import MCOCActionPredictor, Config
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import time
import os
import sys
sys.path.append('..')


class MCOCPredictor:
    def __init__(self, model_path="models/best_model.pth"):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        self.model = MCOCActionPredictor(
            feature_dim=Config.FEATURE_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=Config.DROPOUT_RATE
        ).to(self.device)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Frame buffer
        self.frame_buffer = []
        self.sequence_length = Config.SEQUENCE_LENGTH

        # Action mapping
        self.action_names = {
            0: 'light_attack',
            1: 'special',
            2: 'block',
            3: 'evade',
            4: 'medium_attack'
        }

        print("Model loaded successfully!")

    def preprocess_frame(self, frame):
        """Preprocessa un frame per l'inferenza"""
        # Converti BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Converti a PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Applica trasformazioni
        tensor = self.transform(pil_image)

        return tensor

    def predict_action(self, frame):
        """Predice l'azione per un singolo frame"""
        # Preprocessa il frame
        processed_frame = self.preprocess_frame(frame)

        # Aggiungi al buffer
        self.frame_buffer.append(processed_frame)

        # Mantieni solo gli ultimi N frame
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)

        # Se non abbiamo abbastanza frame, ritorna light_attack come default
        if len(self.frame_buffer) < self.sequence_length:
            return 'block', 1.0

        # Prepara la sequenza per il modello
        sequence = torch.stack(self.frame_buffer).unsqueeze(
            0)  # (1, seq_len, C, H, W)
        sequence = sequence.to(self.device)

        # Inferenza
        with torch.no_grad():
            outputs = self.model(sequence)
            probabilities = torch.softmax(outputs, dim=1)

            # Trova la predizione più probabile
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

            action_name = self.action_names[predicted_class]

        return action_name, confidence

    def predict_from_video(self, video_path=None, camera_id=0):
        """Predice azioni da un video o dalla webcam"""
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        print("Press 'q' to quit, 's' to save frame")

        frame_count = 0
        fps_counter = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predici azione
            action, confidence = self.predict_action(frame)

            # Calcola FPS
            fps_counter += 1
            if time.time() - start_time >= 1.0:
                fps = fps_counter / (time.time() - start_time)
                fps_counter = 0
                start_time = time.time()

            # Disegna informazioni sul frame
            cv2.putText(frame, f"Action: {action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostra il frame
            cv2.imshow('MCOC Action Predictor', frame)

            # Gestisci input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salva frame
                cv2.imwrite(f"captured_frame_{frame_count}.png", frame)
                print(f"Frame saved as captured_frame_{frame_count}.png")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def predict_from_screenshots(self, screenshot_dir):
        """Predice azioni da una cartella di screenshot"""
        if not os.path.exists(screenshot_dir):
            print(f"Directory {screenshot_dir} not found")
            return

        # Lista tutti i file PNG nella directory
        files = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
        files.sort()  # Ordina per nome

        print(f"Found {len(files)} screenshots")

        for i, filename in enumerate(files):
            file_path = os.path.join(screenshot_dir, filename)

            # Carica immagine
            frame = cv2.imread(file_path)
            if frame is None:
                print(f"Could not load {filename}")
                continue

            # Predici azione
            action, confidence = self.predict_action(frame)

            print(f"{filename}: {action} (confidence: {confidence:.2f})")

            # Mostra immagine con predizione
            cv2.putText(frame, f"Action: {action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Prediction', frame)
            cv2.waitKey(100)  # Mostra per 100ms

        cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MCOC Action Predictor')
    parser.add_argument('--model', default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--mode', choices=['camera', 'video', 'screenshots'],
                        default='camera', help='Inference mode')
    parser.add_argument('--source', default='0',
                        help='Video file path or camera ID')
    parser.add_argument('--screenshot_dir', default='assets/data/raw',
                        help='Directory with screenshots')

    args = parser.parse_args()

    # Inizializza predictor
    predictor = MCOCPredictor(args.model)

    # Esegui inferenza
    if args.mode == 'camera':
        camera_id = int(args.source) if args.source.isdigit() else 0
        predictor.predict_from_video(camera_id=camera_id)
    elif args.mode == 'video':
        predictor.predict_from_video(video_path=args.source)
    elif args.mode == 'screenshots':
        predictor.predict_from_screenshots(args.screenshot_dir)


if __name__ == "__main__":
    main()
