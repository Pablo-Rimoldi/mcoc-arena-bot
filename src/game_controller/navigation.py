#import pyautogui
import cv2
import time
from config.settings import config
from src.data_collection.screen_capture import ScreenCapture
from src.game_calibrator.game_area_detector import GameAreaDetector
from src.data_collection.file_manager import FileManager


class Navigator():
    def __init__(self ) -> None:
        self.file_manager = FileManager(config.ASSETS_DIR)
        self.detector = GameAreaDetector()
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
    
    def move_to_arena(self):
        print(self.detector.game_region)

    def relative_position_encoding(self, position:tuple):
        pass



time.sleep(5)
navigator = Navigator()
navigator.move_to_arena()
