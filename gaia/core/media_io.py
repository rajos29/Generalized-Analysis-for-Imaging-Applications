from __future__ import annotations

from pathlib import Path

import cv2
try:
    import imageio.v3 as iio
except ImportError:
    iio = None
import numpy as np

from gaia.core.image_ops import ensure_rgb

def load_image(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is not None:
        if frame.ndim == 3 and frame.shape[2] == 4:
            return ensure_rgb(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA), source_order="RGB")
        return ensure_rgb(frame, source_order="BGR")
    if iio is None:
        raise RuntimeError("Image loading requires OpenCV or imageio. Install dependencies with: python -m pip install -r requirements.txt")
    return ensure_rgb(iio.imread(path), source_order="RGB")


def load_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(ensure_rgb(frame, source_order="BGR"))
    capture.release()
    if not frames:
        raise ValueError(f"No frames were read from {path}.")
    return frames

