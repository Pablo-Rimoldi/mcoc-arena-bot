import cv2
import numpy as np
from pathlib import Path

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

