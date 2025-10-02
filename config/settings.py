from typing import Tuple
import os
import numpy as np


class Config:
    # General settings
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

    # Training model paths
    DATA_DIR = "assets/data/raw"
    MODEL_SAVE_DIR = "models"
    RESULTS_DIR = "results"

    # Model parameters
    SEQUENCE_LENGTH = 10  # 10 frame = 2.5 secondi a 4 FPS
    IMAGE_SIZE = 224
    FEATURE_DIM = 256
    HIDDEN_SIZE = 128
    NUM_CLASSES = 5

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30
    DROPOUT_RATE = 0.3

    # Labels mapping
    LABEL_MAPPING = {
        'w': 0,  # light_attack
        'd': 1,  # medium_attack  
        'a': 2,  # evade
        's': 3,  # parry
        'space': 4  # special_attack
    }

    REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


config = Config()
