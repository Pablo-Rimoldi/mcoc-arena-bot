#import pyautogui
import cv2
import time
from config.settings import config
from src.data_collection.screen_capture import ScreenCapture
from src.game_calibrator.game_area_detector import GameAreaDetector
from src.data_collection.file_manager import FileManager
import pyautogui
import random


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
        self._random_move(r"assets\config\fight_menu.png")


    def relative_position_encoding(self):
        pass

    def _random_move(self, filepathofimage, n=5):
        x, y = pyautogui.locateCenterOnScreen(filepathofimage, confidence=.8)
        dx = random.randint(-n, n)
        dy = random.randint(-n, n)
        perturbed_x = x + dx
        perturbed_y = y + dy
        pyautogui.moveTo(perturbed_x, perturbed_y)
        pyautogui.click()
        time.sleep(1)


time.sleep(5)
navigator = Navigator()
navigator.move_to_arena()
