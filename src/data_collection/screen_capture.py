from typing import Optional, Dict
import cv2
import numpy as np
import mss


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
