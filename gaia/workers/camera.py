from __future__ import annotations

import time

import cv2
import numpy as np
from qtpy.QtCore import QThread, Signal

from gaia.config import BACKENDS, FORMATS
from gaia.core.image_ops import ensure_rgb

class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)
    failed_frame = Signal()
    status = Signal(str)

    def __init__(
        self,
        camera_index: int,
        backend_name: str,
        pixel_format: str,
        width: int | None,
        height: int | None,
        capture_fps: int | None,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.backend_name = backend_name
        self.pixel_format = pixel_format
        self.width = width
        self.height = height
        self.capture_fps = capture_fps
        self.running = False

    def run(self) -> None:
        backend_names = [self.backend_name] + [name for name in BACKENDS if name != self.backend_name]
        capture = None
        actual_backend = self.backend_name
        for candidate in backend_names:
            backend = BACKENDS[candidate]
            capture = cv2.VideoCapture(self.camera_index, backend) if backend else cv2.VideoCapture(self.camera_index)
            if capture.isOpened():
                actual_backend = candidate
                break
            capture.release()
            capture = None
        if capture is None:
            self.status.emit(f"Could not open camera {self.camera_index}")
            return

        fourcc_code = FORMATS[self.pixel_format]
        if fourcc_code:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_code))
        if self.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.capture_fps:
            capture.set(cv2.CAP_PROP_FPS, self.capture_fps)

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status.emit(f"Camera {self.camera_index} {actual_backend} {self.pixel_format} {actual_width}x{actual_height}")

        for _ in range(4):
            capture.read()

        self.running = True
        while self.running:
            ok, frame = capture.read()
            if ok and frame is not None:
                self.frame_ready.emit(ensure_rgb(frame, source_order="BGR"))
            else:
                self.failed_frame.emit()
                self.msleep(2)
        capture.release()

    def stop(self) -> None:
        self.running = False
        self.wait(1500)


