import pyautogui as pt
import cv2
from src.data_collection.screen_capture import ScreenCapture
from src.game_calibrator.game_area_detector import GameAreaDetector


class Navigator():
    def __init__(self, detector: GameAreaDetector ) -> None:
        self.detector = detector
        self.region = self._auto_calibrate()

    def _auto_calibrate(self):
        """Auto-calibrate the detector and save a preview image."""
        try:
            with ScreenCapture() as capture:
                full_screenshot = capture.capture_region()

            if self.detector:
                self.detector.calibrate(
                    full_screenshot, self.file_manager.output_dir)
                region = self.detector.get_mss_region()
                return region
        except Exception as e:
            print(f"An error occurred during calibration: {e}")
    
    def move_to_arena(self):



