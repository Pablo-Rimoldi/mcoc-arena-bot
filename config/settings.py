from typing import Tuple
import os
import numpy as np

class Config:
    ASSETS_DIR: str = os.path.join(
        os.path.dirname(__file__), '../assets/data/raw')
    FPS: int = 30
    TARGET_COLOR_BGR = np.array([12, 0, 208])    
    COLOR_TOLERANCE: int = 20
    MIN_WIDTH: int = 600
    MIN_HEIGHT: int = 400
    TOP_BAR_CROP_PERCENTAGE = 0.085
    SIDE_MARGIN = 3
    BOTTOM_MARGIN = 3

    # Template matching config
    TEMPLATE_IMAGE_PATH: str = r"assets\config\pause.png"
    TEMPLATE_MATCH_THRESHOLD: float = 0.85

config = Config()
