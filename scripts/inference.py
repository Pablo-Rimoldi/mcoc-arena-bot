import os
import sys
import time
from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.settings import config as Config  # noqa: E402
from src.data_collection.screen_capture import ScreenCapture  # noqa: E402
from src.game_calibrator.game_area_detector import GameAreaDetector  # noqa: E402


class InputController:
    """Sends key inputs using the most reliable backend available for games.

    Priority: pydirectinput > pyautogui > keyboard.
    """

    def __init__(self):
        self.backend = None
        self._pdi = None
        self._pag = None
        self._kbd = None
        self._win = None

        # 1) Try Win32 SendInput scancodes (best for some games)
        try:
            if os.name == 'nt':
                self._win = _WindowsSendInput()
                self.backend = 'sendinput'
        except Exception:
            self._win = None

        # 2) Try pydirectinput
        if self.backend is None:
            try:
                import pydirectinput  # type: ignore
                self._pdi = pydirectinput
                self._pdi.PAUSE = 0
                self.backend = 'pydirectinput'
            except Exception:
                pass

        if self.backend is None:
            try:
                import pyautogui  # type: ignore
                self._pag = pyautogui
                self._pag.FAILSAFE = False
                self._pag.PAUSE = 0
                self.backend = 'pyautogui'
            except Exception:
                pass

        if self.backend is None:
            try:
                import keyboard  # type: ignore
                self._kbd = keyboard
                self.backend = 'keyboard'
            except Exception:
                pass

        print(f"Input backend: {self.backend if self.backend else 'none'}")

    def press(self, key: str, duration_ms: int):
        if not key or self.backend is None:
            return
        duration_s = max(0.0, duration_ms / 1000.0)
        if self.backend == 'sendinput':
            self._win.press(key, duration_s)
        elif self.backend == 'pydirectinput':
            self._pdi.keyDown(key)
            time.sleep(duration_s)
            self._pdi.keyUp(key)
        elif self.backend == 'pyautogui':
            self._pag.keyDown(key)
            time.sleep(duration_s)
            self._pag.keyUp(key)
        elif self.backend == 'keyboard':
            # keyboard.press handles press/release automatically if using send
            self._kbd.press(key)
            time.sleep(duration_s)
            self._kbd.release(key)


class _WindowsSendInput:
    """Minimal Windows SendInput scancode sender for WASD and Space."""

    # Map to scan codes (US layout): W=0x11, A=0x1E, S=0x1F, D=0x20, Space=0x39
    KEY_TO_SCANCODE = {
        'w': 0x11,
        'a': 0x1E,
        's': 0x1F,
        'd': 0x20,
        'space': 0x39,
    }

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes

        # Setup SendInput structures
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR),
            ]

        class INPUT(ctypes.Structure):
            class _I(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT)]
            _anonymous_ = ("i",)
            _fields_ = [("type", wintypes.DWORD), ("i", _I)]

        self.KEYBDINPUT = KEYBDINPUT
        self.INPUT = INPUT
        self.SendInput = ctypes.windll.user32.SendInput
        self.KEYEVENTF_SCANCODE = 0x0008
        self.KEYEVENTF_KEYUP = 0x0002
        self.INPUT_KEYBOARD = 1

    def _send_scan(self, scancode: int, keyup: bool = False):
        flags = self.KEYEVENTF_SCANCODE | (
            self.KEYEVENTF_KEYUP if keyup else 0)
        ki = self.KEYBDINPUT(0, scancode, flags, 0, None)
        inp = self.INPUT(self.INPUT_KEYBOARD, ki)
        n = self.SendInput(1, self.ctypes.byref(
            inp), self.ctypes.sizeof(self.INPUT))
        return n

    def press(self, key: str, duration_s: float):
        sc = self.KEY_TO_SCANCODE.get(key.lower())
        if sc is None:
            return
        self._send_scan(sc, keyup=False)
        time.sleep(max(0.0, duration_s))
        self._send_scan(sc, keyup=True)


class MCOCActionPredictorV2(nn.Module):
    """Model with action embedding, matching scripts/train.py."""

    def __init__(self, feature_dim=256, hidden_size=128, num_classes=5, dropout_rate=0.3, action_embedding_dim=32):
        super(MCOCActionPredictorV2, self).__init__()
        self.cnn = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in self.cnn.features[:10].parameters():
            param.requires_grad = False
        self.cnn.classifier = nn.Identity()
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.action_embedding = nn.Embedding(
            num_embeddings=num_classes + 1, embedding_dim=action_embedding_dim)
        gru_input_size = feature_dim + action_embedding_dim
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if 2 > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, prev_actions):
        batch_size, seq_len, c, h, w = x.shape
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn.features(x_reshaped)
        extracted_features = self.feature_extractor(features)
        sequence_features = extracted_features.view(batch_size, seq_len, -1)
        action_embeds = self.action_embedding(prev_actions)
        combined_features = torch.cat(
            [sequence_features, action_embeds], dim=-1)
        gru_output, _ = self.gru(combined_features)
        final_output = gru_output[:, -1, :]
        logits = self.classifier(final_output)
        return logits


class RealTimeInference:
    """Esegue inferenza in tempo reale catturando lo schermo alla stessa frequenza del training
    e premendo i tasti corrispondenti con pyautogui.
    """

    def __init__(self, model_path: str, fps: int = 4, press_ms: int = 80, min_confidence: float = 0.0):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load checkpoint and derive runtime config
        checkpoint = torch.load(model_path, map_location=self.device)
        ckpt_cfg = checkpoint.get('config', {}) or {}
        self.num_classes = int(ckpt_cfg.get('NUM_CLASSES', Config.NUM_CLASSES))
        self.feature_dim = int(ckpt_cfg.get('FEATURE_DIM', Config.FEATURE_DIM))
        self.hidden_size = int(ckpt_cfg.get('HIDDEN_SIZE', Config.HIDDEN_SIZE))
        self.dropout_rate = float(ckpt_cfg.get(
            'DROPOUT_RATE', Config.DROPOUT_RATE))
        self.sequence_length = int(ckpt_cfg.get(
            'SEQUENCE_LENGTH', Config.SEQUENCE_LENGTH))
        self.image_size = int(ckpt_cfg.get(
            'IMAGE_SIZE', getattr(Config, 'IMAGE_SIZE', 224)))
        self.label_mapping = ckpt_cfg.get(
            'LABEL_MAPPING', getattr(Config, 'LABEL_MAPPING', {}))
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items(
        )} if self.label_mapping else Config.REVERSE_LABEL_MAPPING
        self.action_embedding_dim = ckpt_cfg.get('ACTION_EMBEDDING_DIM', None)
        self.start_token_idx = int(ckpt_cfg.get(
            'START_TOKEN_IDX', self.num_classes))

        # Model setup depending on architecture
        if self.action_embedding_dim is not None:
            print("Model variant: with action embedding")
            self.model = MCOCActionPredictorV2(
                feature_dim=self.feature_dim,
                hidden_size=self.hidden_size,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
                action_embedding_dim=int(self.action_embedding_dim),
            ).to(self.device)
            self.uses_prev_actions = True
        else:
            print("error")
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded weights from: {model_path}")

        # Same preprocessing as training
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        # Frame and action buffers
        self.frame_buffer: List[torch.Tensor] = []
        # predicted action per frame in buffer
        self.action_buffer: List[Optional[int]] = []
        self.last_action_idx: int = self.start_token_idx

        # Inference timing
        self.fps = fps  # Match data collection (default 4 FPS)
        self.interval_s = 1.0 / max(1, self.fps)
        self.press_ms = max(1, press_ms)
        self.min_confidence = float(min_confidence)

        # Input mapping: class index -> key string (e.g., 0 -> 'w')
        self.index_to_key = self.reverse_label_mapping

        # Screen capture and calibration
        self.calibrator = GameAreaDetector()
        self.capture = ScreenCapture()
        self.region = self._calibrate_region()

        # Input controller (DirectInput preferred for games)
        self.input_controller = InputController()

        # Try to focus the game window by clicking the center of calibrated region
        self._focus_window()

    def _calibrate_region(self) -> Optional[dict]:
        """Calibra la regione del gioco usando un frame dello schermo."""
        full_frame = self.capture.capture_region(None)
        # Save calibration preview inside dataset directory
        self.calibrator.calibrate(full_frame, output_dir=Path(Config.DATA_DIR))
        region = self.calibrator.get_mss_region()
        if region is None:
            # Fallback to full screen region if needed
            h, w = full_frame.shape[:2]
            region = {"left": 0, "top": 0, "width": w, "height": h}
            print("Calibration failed, using full screen.")
        else:
            print(f"Calibration OK, region: {region}")
        return region

    def _preprocess_bgr_frame(self, bgr_frame: np.ndarray) -> torch.Tensor:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        tensor = self.transform(pil_image)
        return tensor

    @torch.no_grad()
    def _predict_key(self) -> Optional[str]:
        if len(self.frame_buffer) < self.sequence_length:
            return 'w'
        sequence = torch.stack(self.frame_buffer).unsqueeze(
            0).to(self.device)  # (1, T, C, H, W)

        if self.uses_prev_actions:
            # Build prev_actions indices aligned with frames in buffer
            prev_indices: List[int] = []
            for i in range(len(self.frame_buffer)):
                if i == 0:
                    prev_indices.append(self.last_action_idx)
                else:
                    prev_indices.append(
                        self.action_buffer[i-1] if self.action_buffer[i-1] is not None else self.start_token_idx)
            prev_actions_tensor = torch.tensor(
                prev_indices, dtype=torch.long).unsqueeze(0).to(self.device)
            logits = self.model(sequence, prev_actions_tensor)
        else:
            logits = self.model(sequence)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, predicted_index].item())
        if confidence < self.min_confidence:
            return None
        return self.index_to_key.get(predicted_index)

    def _press_key(self, key: str):
        if not key:
            return
        self.input_controller.press(key, self.press_ms)
        print(f"Pressed: {key}")

    def _focus_window(self):
        try:
            import pyautogui
            cx = int(self.region["left"] + self.region["width"] / 2)
            cy = int(self.region["top"] + self.region["height"] / 2)
            pyautogui.moveTo(cx, cy, duration=0)
            pyautogui.click(cx, cy)
            time.sleep(0.1)
            print(f"Focused game window at ({cx}, {cy})")
        except Exception as e:
            print(f"Focus click failed: {e}")

    def run(self):
        print(
            f"Starting real-time inference at ~{self.fps} FPS. Press Ctrl+C to stop.")
        try:
            while True:
                loop_start = time.time()

                frame = self.capture.capture_region(self.region)
                frame_tensor = self._preprocess_bgr_frame(frame)

                self.frame_buffer.append(frame_tensor)
                self.action_buffer.append(None)
                if len(self.frame_buffer) > self.sequence_length:
                    self.frame_buffer.pop(0)
                    self.action_buffer.pop(0)

                key_to_press = self._predict_key()
                if key_to_press is not None:
                    # Map key back to index
                    # If key not in mapping (shouldn't happen), keep last_action_idx unchanged
                    inv_map = {v: k for k, v in self.index_to_key.items()}
                    predicted_index = inv_map.get(
                        key_to_press, self.last_action_idx)
                    # Update buffers with the latest predicted class index
                    if len(self.action_buffer) > 0:
                        self.action_buffer[-1] = predicted_index
                    self.last_action_idx = predicted_index
                    self._press_key(key_to_press)

                elapsed = time.time() - loop_start
                time_to_sleep = max(0.0, self.interval_s - elapsed)
                time.sleep(time_to_sleep)
        except KeyboardInterrupt:
            print("\nStopping inference.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time MCOC inference and control")
    parser.add_argument("--model", default=os.path.join(Config.MODEL_SAVE_DIR,
                        "best_model.pth"), help="Path to trained model file")
    parser.add_argument("--fps", type=int, default=4,
                        help="Capture frequency (must match training data FPS)")
    parser.add_argument("--press_ms", type=int, default=80,
                        help="Key press duration in milliseconds")
    parser.add_argument("--min_conf", type=float, default=0.0,
                        help="Minimum confidence to trigger a key press")

    args = parser.parse_args()

    engine = RealTimeInference(
        model_path=args.model,
        fps=args.fps,
        press_ms=args.press_ms,
        min_confidence=args.min_conf,
    )
    engine.run()


if __name__ == "__main__":
    time.sleep(2)
    main()
