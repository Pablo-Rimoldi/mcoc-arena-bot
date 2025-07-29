import os
import time
from threading import Thread, Event
from typing import Optional, Tuple, Dict
import cv2
import numpy as np
from pathlib import Path
import mss
import keyboard
from config.settings import config


class InputHandler:
    """Handles keyboard input detection."""

    KEY_MAPPINGS = {
        'w': 'w',
        'a': 'a',
        's': 's',
        'd': 'd',
        'space': 'space'
    }

    @staticmethod
    def get_current_action() -> str:
        """Returns the first key pressed or 'idle' if none."""
        try:
            for key, action in InputHandler.KEY_MAPPINGS.items():
                if keyboard.is_pressed(key):
                    return action
            return 'idle'
        except ImportError:
            # keyboard library might not work in some environments (e.g., remote shells)
            return 'idle'


class ScreenCapture:
    """Handles screen capture operations."""
    def __init__(self):
        self._sct = mss.mss()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._sct:
            self._sct.close()
    def capture_region(self, region: Optional[Dict] = None) -> np.ndarray:
        """Capture screen region and return as BGR numpy array."""
        # If no region is specified, capture the primary monitor
        monitor = region if region else self._sct.monitors[1]
        # Grab the data
        sct_img = self._sct.grab(monitor)
        # Convert to a NumPy array
        screenshot = np.array(sct_img)
        # Convert from BGRA to BGR
        if screenshot.shape[2] == 4:
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        return screenshot


class GameAreaDetector:
    def __init__(self):
        self.game_region: Optional[Tuple[int, int, int, int]] = None
        self.is_calibrated = False

    def find_main_red_rectangle(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Finds the largest contour corresponding to the red game window frame."""
        lower_bound = np.clip(config.TARGET_COLOR_BGR -
                              config.COLOR_TOLERANCE, 0, 255)
        upper_bound = np.clip(config.TARGET_COLOR_BGR +
                              config.COLOR_TOLERANCE, 0, 255)
        mask = cv2.inRange(screenshot, lower_bound, upper_bound)

        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("Debug: No red contours found on screen.")
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        if w < config.MIN_WIDTH or h < config.MIN_HEIGHT:
            print(
                f"Debug: Largest contour found at ({x},{y}) with size ({w},{h}) is too small. Skipping.")
            return None

        print(
            f"Debug: Found main game window at ({x},{y}) with size ({w},{h}).")

        top_crop_pixels = int(h * config.TOP_BAR_CROP_PERCENTAGE)
        final_x = x + config.SIDE_MARGIN
        final_y = y + top_crop_pixels
        final_w = w - (2 * config.SIDE_MARGIN)
        final_h = h - top_crop_pixels - config.BOTTOM_MARGIN
        return (final_x, final_y, final_w, final_h)

    def calibrate(self, screenshot: np.ndarray, output_dir: Path) -> bool:
        """Calibrate by finding the main red rectangle and save a preview."""
        print("Calibrating game area by finding the main red rectangle...")

        region = self.find_main_red_rectangle(screenshot)

        if region:
            self.game_region = region
            self.is_calibrated = True
            print(
                f"Calibration successful! Game area defined at: {self.game_region}")

            try:
                x, y, w, h = self.game_region
                preview_image = screenshot[y:y+h, x:x+w]

                if preview_image.size > 0:
                    preview_path = output_dir / "_calibration_preview.png"
                    cv2.imwrite(str(preview_path), preview_image)
                    print(f"Saved calibration preview to: {preview_path}")
                else:
                    print("Warning: Preview image is empty, could not save.")
            except Exception as e:
                print(f"Error saving calibration preview: {e}")

            return True

        h_scr, w_scr = screenshot.shape[:2]
        self.game_region = (0, 0, w_scr, h_scr)
        self.is_calibrated = True
        print("Warning: Game area auto-detection failed. Falling back to full screen.")
        return False

    def get_mss_region(self) -> Optional[Dict]:
        """Convert region tuple to mss format."""
        if not self.is_calibrated or not self.game_region:
            return None
        x, y, w, h = self.game_region
        return {"left": x, "top": y, "width": w, "height": h}


class FileManager:
    """Handles file operations and naming."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter = self._init_counter()
    def _init_counter(self) -> int:
        """Initialize the counter based on existing files in the directory."""
        existing_files = list(self.output_dir.glob("*.png"))
        existing_files = [
            f for f in existing_files if f.name != "_calibration_preview.png"]
        if not existing_files:
            return 0

        nums = []
        for f in existing_files:
            try:
                num = int(f.stem.split('_')[0])
                nums.append(num)
            except (ValueError, IndexError):
                continue

        return max(nums) + 1 if nums else 0

    def generate_filename(self, action: str = 'idle') -> str:
        """Generate filename as a zero-padded integer plus action, e.g., 000001_w.png."""
        filename = f"{self.counter:06d}_{action}.png"
        self.counter += 1
        return filename

    def save_image(self, image: np.ndarray, filename: str) -> bool:
        """Save image to disk."""
        filepath = self.output_dir / filename
        return cv2.imwrite(str(filepath), image)


class MCOCDataCollector:
    """Main class for collecting MCOC gameplay data."""
    def __init__(self, fps: int = 4, auto_detect: bool = True):
        self.fps = fps
        self.interval = 1.0 / self.fps
        self.auto_detect = auto_detect

        self.file_manager = FileManager(config.ASSETS_DIR)
        self.detector = GameAreaDetector() if auto_detect else None
        self.input_handler = InputHandler()
        self.is_in_fight = True

        self.stop_event = Event()
        self.collection_thread = None
        self.is_running = False

        if self.auto_detect:
            self._auto_calibrate()

    def _auto_calibrate(self):
        """Auto-calibrate the detector and save a preview image."""
        try:
            with ScreenCapture() as capture:
                full_screenshot = capture.capture_region()

            if self.detector:
                self.detector.calibrate(
                    full_screenshot, self.file_manager.output_dir)
        except Exception as e:
            print(f"An error occurred during calibration: {e}")

    def _is_in_fight_area(self, screenshot: np.ndarray) -> bool:
        """Determines if the screenshot contains the fight UI."""
        # Sostituisci questo con una logica di template matching reale se necessario
        return self.is_in_fight

    def _resize_to_fixed(self, image: np.ndarray, size: int = 224) -> np.ndarray:
        """Resize image to a fixed square size."""
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    def _capture_loop(self):
        """Main capture loop running in a separate thread."""
        region = self.detector.get_mss_region(
        ) if self.detector and self.detector.is_calibrated else None

        if region is None:
            print("Capture loop cannot start: game area not detected or not calibrated.")
            return

        with ScreenCapture() as capture:
            while not self.stop_event.is_set():
                loop_start_time = time.time()
                try:
                    screenshot = capture.capture_region(region)

                    if self._is_in_fight_area(screenshot):
                        action = self.input_handler.get_current_action()

                        # Salta il salvataggio se l'azione è 'idle'
                        if action == 'idle':
                            continue

                        screenshot_resized = self._resize_to_fixed(
                            screenshot, 224)
                        filename = self.file_manager.generate_filename(
                            action=action)
                        self.file_manager.save_image(
                            screenshot_resized, filename)

                except Exception as e:
                    print(f"Error in capture loop: {e}")
                    # Evita di bloccare il loop in caso di errore
                    pass

                elapsed = time.time() - loop_start_time
                sleep_time = max(0, self.interval - elapsed)
                time.sleep(sleep_time)

    def start(self):
        """Start data collection."""
        if self.is_running:
            print("Collector is already running.")
            return

        self.is_running = True
        self.stop_event.clear()

        self.collection_thread = Thread(target=self._capture_loop, daemon=True)
        self.collection_thread.start()
        print("Data collection started.")

    def stop(self):
        """Stop data collection."""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)

        self.collection_thread = None
        print("Data collection stopped.")

    def get_output_dir(self) -> str:
        """Get the output directory path."""
        return str(self.file_manager.output_dir)


if __name__ == "__main__":
    print("--- MCOC Data Collector ---")
    print("Starting in 3 seconds. Please switch to the game window.")
    time.sleep(3)

    collector = None
    try:
        collector = MCOCDataCollector(output_dir="mcoc_data")

        if not collector.detector or not collector.detector.is_calibrated:
            print("Could not initialize collector properly. Exiting.")
        else:
            print(f"Saving images to: {collector.get_output_dir()}")
            print(
                "Press W, A, S, D, or SPACE to capture data. Collection will run for 2 minutes.")
            print("Press Ctrl+C in this console to stop early.")
            collector.start()

            # Esegui per 120 secondi (2 minuti)
            time.sleep(120)

    except KeyboardInterrupt:
        print("\nInterruption detected. Stopping collector...")
    except Exception as e:
        print(f"A critical error occurred: {e}")
    finally:
        if collector and collector.is_running:
            collector.stop()
        print("Script finished.")
