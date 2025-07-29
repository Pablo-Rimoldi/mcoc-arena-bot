from typing import Optional, Tuple, Dict
import cv2
import numpy as np
from pathlib import Path
from config.settings import config


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
