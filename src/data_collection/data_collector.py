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
from .screen_capture import ScreenCapture
from .file_manager import FileManager
from ..game_calibrator import GameAreaDetector

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
