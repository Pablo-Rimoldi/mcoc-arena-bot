from typing import Tuple
import os


class Config:
    ASSETS_DIR: str = os.path.join(
        os.path.dirname(__file__), '../assets/data/raw')
    FPS: int = 30
    MCOC_BORDER_COLOR_BGR: Tuple[int, int, int] = (
        13, 1, 208)  # rgba(208,1,13,255)
    COLOR_TOLERANCE: int = 15
    MIN_GAME_AREA: int = 240000
    GAME_ASPECT_RATIO_MIN: float = 1.2
    GAME_ASPECT_RATIO_MAX: float = 2.5
    MIN_GAME_WIDTH: int = 600
    MIN_GAME_HEIGHT: int = 400

    # Template matching config
    TEMPLATE_IMAGE_PATH: str = r"assets\config\pause.png"
    TEMPLATE_MATCH_THRESHOLD: float = 0.85


config = Config()
