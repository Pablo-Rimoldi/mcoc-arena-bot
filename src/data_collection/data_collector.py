import os
import time
import uuid
from datetime import datetime
from threading import Thread, Event
from typing import Optional, Tuple, Dict
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from config.settings import config

try:
    import mss
except ImportError:
    mss = None

try:
    import keyboard
except ImportError:
    keyboard = None


class InputHandler:
    """Handles keyboard input detection"""

    KEY_MAPPINGS = {
        'w': 'w',
        'a': 'a',
        's': 's',
        'd': 'd',
        'space': 'space'
    }

    @staticmethod
    def get_current_action() -> str:
        """Returns the first key pressed or 'idle' if none"""
        if not keyboard:
            return 'idle'

        for key, action in InputHandler.KEY_MAPPINGS.items():
            if keyboard.is_pressed(key):
                return action
        return 'idle'


class ScreenCapture:
    """Handles screen capture operations"""

    def __init__(self):
        if not mss:
            raise ImportError("mss library is required for screen capture")
        self._sct = None

    def __enter__(self):
        self._sct = mss.mss()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._sct:
            self._sct.close()

    def capture_region(self, region: Optional[Dict] = None) -> np.ndarray:
        """Capture screen region and return as BGR numpy array"""
        if not self._sct:
            raise RuntimeError("ScreenCapture not properly initialized")
        monitor = region if region else self._sct.monitors[1]
        screenshot = np.array(self._sct.grab(monitor))
        if screenshot.shape[2] == 4:
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        return screenshot


class GameAreaDetector:
    """Detects MCOC game area using computer vision"""

    def __init__(self):
        self.game_region: Optional[Tuple[int, int, int, int]] = None
        self.is_calibrated = False

    def detect_by_border_color(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect game area by finding the specific red border color"""
        target_color = np.array(config.MCOC_BORDER_COLOR_BGR)
        tolerance = config.COLOR_TOLERANCE

        lower_bound = np.clip(target_color - tolerance, 0, 255)
        upper_bound = np.clip(target_color + tolerance, 0, 255)
        mask = cv2.inRange(screenshot, lower_bound, upper_bound)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        largest_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.MIN_GAME_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if (config.GAME_ASPECT_RATIO_MIN < aspect_ratio < config.GAME_ASPECT_RATIO_MAX and
                    w > config.MIN_GAME_WIDTH and h > config.MIN_GAME_HEIGHT):

                if area > largest_area:
                    largest_area = area
                    margin = 3
                    best_rect = (x + margin, y + margin,
                                 w - 2*margin, h - 2*margin)

        return best_rect

    def detect_by_top_bar(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect game area by finding the top red bar"""
        target_color = np.array(config.MCOC_BORDER_COLOR_BGR)
        tolerance = config.COLOR_TOLERANCE

        lower_bound = np.clip(target_color - tolerance, 0, 255)
        upper_bound = np.clip(target_color + tolerance, 0, 255)
        mask = cv2.inRange(screenshot, lower_bound, upper_bound)

        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
        mask_horizontal = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel_horizontal)

        contours, _ = cv2.findContours(
            mask_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if (w > 400 and 10 < h < 80 and y < screenshot.shape[0] // 2):
                game_height = int(w * 0.75)

                if y + game_height <= screenshot.shape[0] and w > config.MIN_GAME_WIDTH:
                    margin = 3
                    return (x + margin, y + margin, w - 2*margin, game_height - margin)

        return None

    def calibrate(self, screenshot: np.ndarray) -> bool:
        """Calibrate detector to find game area"""
        detection_methods = [
            self.detect_by_border_color,
            self.detect_by_top_bar
        ]

        for method in detection_methods:
            region = method(screenshot)
            if region:
                self.game_region = region
                self.is_calibrated = True
                return True

        # Fallback to full screen
        h, w = screenshot.shape[:2]
        self.game_region = (0, 0, w, h)
        self.is_calibrated = True
        return False

    def get_mss_region(self) -> Optional[Dict]:
        """Convert region tuple to mss format"""
        if not self.is_calibrated or not self.game_region:
            return None

        x, y, w, h = self.game_region
        return {"left": x, "top": y, "width": w, "height": h}


class FileManager:
    """Handles file operations and naming"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_filename(self, action: str) -> str:
        """Generate unique filename with timestamp and action"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        unique_id = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_id}_{action}.png"

    def save_image(self, image: np.ndarray, filename: str) -> bool:
        """Save image to disk"""
        filepath = self.output_dir / filename
        return cv2.imwrite(str(filepath), image)


class MCOCDataCollector:
    """Main class for collecting MCOC gameplay data"""

    def __init__(self, fps: int = 30, auto_detect: bool = True, output_dir: str = None):
        # Validate requirements
        if not mss:
            raise ImportError("mss library is required")
        if not keyboard:
            raise ImportError("keyboard library is required")

        self.fps = fps
        self.interval = 1.0 / self.fps
        self.auto_detect = auto_detect

        # Initialize components
        self.detector = GameAreaDetector() if auto_detect else None
        self.file_manager = FileManager(output_dir or config.ASSETS_DIR)
        self.input_handler = InputHandler()

        # Template matching
        self.template_path = config.TEMPLATE_IMAGE_PATH
        self.template_match_threshold = config.TEMPLATE_MATCH_THRESHOLD
        self.template = cv2.imread(self.template_path)
        if self.template is None:
            raise FileNotFoundError(
                f"Template image not found: {self.template_path}")

        # Control
        self.stop_event = Event()
        self.collection_thread = None
        self.is_running = False

        # Auto-calibrate if enabled
        if self.auto_detect:
            self._auto_calibrate()

    def _auto_calibrate(self):
        """Auto-calibrate the detector on initialization"""
        try:
            with ScreenCapture() as capture:
                screenshot = capture.capture_region()
            self.detector.calibrate(screenshot)
        except Exception:
            # Silent fallback - will use full screen
            pass

    def _template_present(self, screenshot: np.ndarray) -> bool:
        """Check if the template is present in the screenshot using template matching."""
        if screenshot.shape[0] < self.template.shape[0] or screenshot.shape[1] < self.template.shape[1]:
            return False
        res = cv2.matchTemplate(
            screenshot, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val >= self.template_match_threshold

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        region = self.detector.get_mss_region() if self.detector else None

        with ScreenCapture() as capture:
            while not self.stop_event.is_set():
                loop_start = time.time()

                try:
                    # Capture and save
                    screenshot = capture.capture_region(region)
                    if self._template_present(screenshot):
                        action = self.input_handler.get_current_action()
                        filename = self.file_manager.generate_filename(action)
                        self.file_manager.save_image(screenshot, filename)

                except Exception:
                    # Continue on errors
                    pass

                # Maintain FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def start(self):
        """Start data collection"""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()

        # Start collection thread
        self.collection_thread = Thread(target=self._capture_loop, daemon=True)
        self.collection_thread.start()

    def stop(self):
        """Stop data collection"""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        # Wait for thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)

        self.collection_thread = None

    def is_collecting(self) -> bool:
        """Check if currently collecting data"""
        return self.is_running

    def get_output_dir(self) -> str:
        """Get the output directory path"""
        return str(self.file_manager.output_dir)


if __name__ == "__main__":
    print("Starting MCOC Data Collector (10 second test)...")
    try:
        time.sleep(2)
        collector = MCOCDataCollector()
        print(f"Saving to: {collector.get_output_dir()}")
        print("Press W, A, S, D, SPACE while testing...")
        collector.start()
        time.sleep(10)  # Run for 10 seconds
        collector.stop()
        print("Test completed!")

    except KeyboardInterrupt:
        print("\nStopping...")
        collector.stop()
    except Exception as e:
        print(f"Error: {e}")
