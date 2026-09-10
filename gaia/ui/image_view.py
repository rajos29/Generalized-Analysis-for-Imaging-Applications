from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
from qtpy.QtCore import QPoint, QRect, Qt, Signal
from qtpy.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from qtpy.QtWidgets import QLabel

from gaia.config import ROLLING_FPS_FRAMES
from gaia.core.image_ops import ensure_rgb

def pixmap_from_rgb(frame_rgb: np.ndarray, width: int, height: int) -> QPixmap:
    frame = ensure_rgb(frame_rgb)
    qimage = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(width, height, Qt.KeepAspectRatio, Qt.FastTransformation)


def pixmap_from_gray(frame_gray: np.ndarray, width: int, height: int) -> QPixmap:
    frame = np.ascontiguousarray(frame_gray.astype(np.uint8, copy=False))
    qimage = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimage).scaled(width, height, Qt.KeepAspectRatio, Qt.FastTransformation)


class FpsTracker:
    def __init__(self) -> None:
        self.times: deque[float] = deque(maxlen=ROLLING_FPS_FRAMES)
        self.previous_time: float | None = None
        self.failed_frames = 0

    def reset(self) -> None:
        self.times.clear()
        self.previous_time = None
        self.failed_frames = 0

    def mark_frame(self) -> tuple[float, float]:
        now = time.monotonic()
        instant = 0.0 if self.previous_time is None else 1.0 / max(0.001, now - self.previous_time)
        self.previous_time = now
        self.times.append(now)
        if len(self.times) > 1:
            rolling = (len(self.times) - 1) / max(0.001, self.times[-1] - self.times[0])
        else:
            rolling = instant
        return instant, rolling


class ZoomableImageView(QLabel):
    roi_selected = Signal(object)
    polygon_selected = Signal(object)
    annotation_pin_selected = Signal(object)
    annotation_polygon_selected = Signal(object)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._source_pixmap: QPixmap | None = None
        self._message = text
        self.zoom = 1.0
        self.pan = QPoint(0, 0)
        self._drag_start: QPoint | None = None
        self._drag_pan_start = QPoint(0, 0)
        self.roi_mode = False
        self.polygon_mode = False
        self.annotation_pin_mode = False
        self.annotation_polygon_mode = False
        self._roi_start: QPoint | None = None
        self._roi_current: QPoint | None = None
        self._polygon_points: list[QPoint] = []
        self._closed_polygon_points: list[tuple[int, int]] = []
        self._annotation_polygon_points: list[QPoint] = []
        self._annotation_overlays: list[dict[str, object]] = []
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

    def setPixmap(self, pixmap: QPixmap) -> None:  # type: ignore[override]
        self._source_pixmap = pixmap
        self._message = ""
        self.update()

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._source_pixmap = None
        self._message = text
        self.update()

    def clear(self) -> None:  # type: ignore[override]
        self._source_pixmap = None
        self._message = ""
        self.update()

    def set_image_rgb(self, frame_rgb: np.ndarray) -> None:
        self.setPixmap(pixmap_from_rgb(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0]))

    def set_image_gray(self, frame_gray: np.ndarray) -> None:
        self.setPixmap(pixmap_from_gray(frame_gray, frame_gray.shape[1], frame_gray.shape[0]))

    def set_roi_mode(self, enabled: bool) -> None:
        self.roi_mode = bool(enabled)
        if self.roi_mode:
            self.polygon_mode = False
            self.annotation_pin_mode = False
            self.annotation_polygon_mode = False
            self._polygon_points = []
            self._closed_polygon_points = []
            self._annotation_polygon_points = []
        self._roi_start = None
        self._roi_current = None
        self.setCursor(Qt.CrossCursor if self.roi_mode else Qt.OpenHandCursor)
        self.update()

    def set_polygon_mode(self, enabled: bool) -> None:
        self.polygon_mode = bool(enabled)
        if self.polygon_mode:
            self.roi_mode = False
            self.annotation_pin_mode = False
            self.annotation_polygon_mode = False
            self._roi_start = None
            self._roi_current = None
            self._closed_polygon_points = []
            self._annotation_polygon_points = []
        self._polygon_points = []
        self.setCursor(Qt.CrossCursor if self.polygon_mode else Qt.OpenHandCursor)
        self.update()

    def set_annotation_pin_mode(self, enabled: bool) -> None:
        self.annotation_pin_mode = bool(enabled)
        if self.annotation_pin_mode:
            self.roi_mode = False
            self.polygon_mode = False
            self.annotation_polygon_mode = False
            self._roi_start = None
            self._roi_current = None
            self._polygon_points = []
            self._annotation_polygon_points = []
        self.setCursor(Qt.CrossCursor if self.annotation_pin_mode else Qt.OpenHandCursor)
        self.update()

    def set_annotation_polygon_mode(self, enabled: bool) -> None:
        self.annotation_polygon_mode = bool(enabled)
        if self.annotation_polygon_mode:
            self.roi_mode = False
            self.polygon_mode = False
            self.annotation_pin_mode = False
            self._roi_start = None
            self._roi_current = None
            self._polygon_points = []
        self._annotation_polygon_points = []
        self.setCursor(Qt.CrossCursor if self.annotation_polygon_mode else Qt.OpenHandCursor)
        self.update()

    def set_polygon_overlay(self, image_points: object | None) -> None:
        if image_points is None:
            self._closed_polygon_points = []
        else:
            cleaned: list[tuple[int, int]] = []
            for point in image_points:  # type: ignore[operator]
                try:
                    x, y = point
                except (TypeError, ValueError):
                    continue
                cleaned.append((int(x), int(y)))
            self._closed_polygon_points = cleaned if len(cleaned) >= 3 else []
        self.update()

    def set_annotation_overlays(self, overlays: list[dict[str, object]]) -> None:
        self._annotation_overlays = overlays
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom / 1.25)

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan = QPoint(0, 0)
        self.update()

    def set_zoom(self, value: float) -> None:
        self.zoom = max(1.0, min(16.0, float(value)))
        if self.zoom <= 1.0:
            self.pan = QPoint(0, 0)
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._source_pixmap is None:
            return
        if self.roi_mode:
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._source_pixmap is None:
            return
        if self.polygon_mode:
            if event.button() == Qt.LeftButton:
                self._polygon_points.append(event.pos())
                self.update()
                return
            if event.button() == Qt.RightButton:
                selection = self.polygon_to_image_points()
                self.set_polygon_mode(False)
                if selection is not None:
                    self.set_polygon_overlay(selection)
                    self.polygon_selected.emit(selection)
                return
            return
        if self.annotation_pin_mode:
            if event.button() == Qt.LeftButton:
                point = self.point_to_image_xy(event.pos())
                self.set_annotation_pin_mode(False)
                if point is not None:
                    self.annotation_pin_selected.emit(point)
                return
            return
        if self.annotation_polygon_mode:
            if event.button() == Qt.LeftButton:
                self._annotation_polygon_points.append(event.pos())
                self.update()
                return
            if event.button() == Qt.RightButton:
                selection = self.annotation_polygon_to_image_points()
                self.set_annotation_polygon_mode(False)
                if selection is not None:
                    self.annotation_polygon_selected.emit(selection)
                return
            return
        if event.button() != Qt.LeftButton:
            return
        if self.roi_mode:
            self._roi_start = event.pos()
            self._roi_current = event.pos()
            self.update()
            return
        if self.zoom > 1.0:
            self._drag_start = event.pos()
            self._drag_pan_start = QPoint(self.pan)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self.roi_mode and self._roi_start is not None:
            self._roi_current = event.pos()
            self.update()
            return
        if self._drag_start is None:
            return
        self.pan = self._drag_pan_start + (event.pos() - self._drag_start)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self.polygon_mode:
            return
        if event.button() == Qt.LeftButton:
            if self.roi_mode and self._roi_start is not None and self._source_pixmap is not None:
                self._roi_current = event.pos()
                selection = self.selection_to_image_rect(self._roi_start, self._roi_current)
                self.set_roi_mode(False)
                if selection is not None:
                    self.roi_selected.emit(selection)
                return
            self._drag_start = None
            self.setCursor(Qt.OpenHandCursor)

    def pixmap_draw_rect(self) -> QRect | None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return None
        source_width = self._source_pixmap.width()
        source_height = self._source_pixmap.height()
        if source_width <= 0 or source_height <= 0:
            return None
        fit_scale = min(self.width() / source_width, self.height() / source_height)
        scale = max(0.01, fit_scale * self.zoom)
        target_width = max(1, int(source_width * scale))
        target_height = max(1, int(source_height * scale))
        x = int((self.width() - target_width) / 2 + self.pan.x())
        y = int((self.height() - target_height) / 2 + self.pan.y())
        return QRect(x, y, target_width, target_height)

    def selection_to_image_rect(self, start: QPoint, end: QPoint) -> tuple[int, int, int, int] | None:
        draw_rect = self.pixmap_draw_rect()
        if draw_rect is None or self._source_pixmap is None:
            return None
        selected = QRect(start, end).normalized().intersected(draw_rect)
        if selected.width() < 3 or selected.height() < 3:
            return None
        scale_x = self._source_pixmap.width() / max(1, draw_rect.width())
        scale_y = self._source_pixmap.height() / max(1, draw_rect.height())
        x = int((selected.left() - draw_rect.left()) * scale_x)
        y = int((selected.top() - draw_rect.top()) * scale_y)
        width = int(selected.width() * scale_x)
        height = int(selected.height() * scale_y)
        x = max(0, min(x, self._source_pixmap.width() - 1))
        y = max(0, min(y, self._source_pixmap.height() - 1))
        width = max(1, min(width, self._source_pixmap.width() - x))
        height = max(1, min(height, self._source_pixmap.height() - y))
        return x, y, width, height

    def point_to_image_xy(self, point: QPoint) -> tuple[int, int] | None:
        draw_rect = self.pixmap_draw_rect()
        if draw_rect is None or self._source_pixmap is None or not draw_rect.contains(point):
            return None
        scale_x = self._source_pixmap.width() / max(1, draw_rect.width())
        scale_y = self._source_pixmap.height() / max(1, draw_rect.height())
        x = int((point.x() - draw_rect.left()) * scale_x)
        y = int((point.y() - draw_rect.top()) * scale_y)
        x = max(0, min(x, self._source_pixmap.width() - 1))
        y = max(0, min(y, self._source_pixmap.height() - 1))
        return x, y

    def polygon_to_image_points(self) -> list[tuple[int, int]] | None:
        points = [self.point_to_image_xy(point) for point in self._polygon_points]
        image_points = [point for point in points if point is not None]
        if len(image_points) < 3:
            return None
        return image_points

    def annotation_polygon_to_image_points(self) -> list[tuple[int, int]] | None:
        points = [self.point_to_image_xy(point) for point in self._annotation_polygon_points]
        image_points = [point for point in points if point is not None]
        if len(image_points) < 3:
            return None
        return image_points

    def image_to_widget_point(self, point: tuple[int, int]) -> QPoint | None:
        draw_rect = self.pixmap_draw_rect()
        if draw_rect is None or self._source_pixmap is None:
            return None
        scale_x = draw_rect.width() / max(1, self._source_pixmap.width())
        scale_y = draw_rect.height() / max(1, self._source_pixmap.height())
        x = draw_rect.left() + int(point[0] * scale_x)
        y = draw_rect.top() + int(point[1] * scale_y)
        return QPoint(x, y)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#05070a"))
        if self._source_pixmap is None or self._source_pixmap.isNull():
            painter.setPen(QColor("#38bdf8"))
            painter.drawText(self.rect(), Qt.AlignCenter, self._message)
            painter.setPen(QColor("#1f2937"))
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
            return

        source_width = self._source_pixmap.width()
        source_height = self._source_pixmap.height()
        if source_width <= 0 or source_height <= 0:
            return
        fit_scale = min(self.width() / source_width, self.height() / source_height)
        scale = max(0.01, fit_scale * self.zoom)
        target_width = max(1, int(source_width * scale))
        target_height = max(1, int(source_height * scale))
        scaled = self._source_pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.FastTransformation)
        x = int((self.width() - scaled.width()) / 2 + self.pan.x())
        y = int((self.height() - scaled.height()) / 2 + self.pan.y())
        painter.drawPixmap(x, y, scaled)
        if self.roi_mode and self._roi_start is not None and self._roi_current is not None:
            painter.setPen(QColor("#f59e0b"))
            painter.drawRect(QRect(self._roi_start, self._roi_current).normalized())
        if self.polygon_mode and self._polygon_points:
            painter.setPen(QColor("#f59e0b"))
            for point in self._polygon_points:
                painter.drawEllipse(point, 4, 4)
            for first, second in zip(self._polygon_points, self._polygon_points[1:]):
                painter.drawLine(first, second)
        if self.annotation_polygon_mode and self._annotation_polygon_points:
            painter.setPen(QColor("#22c55e"))
            for point in self._annotation_polygon_points:
                painter.drawEllipse(point, 4, 4)
            for first, second in zip(self._annotation_polygon_points, self._annotation_polygon_points[1:]):
                painter.drawLine(first, second)
        if self._closed_polygon_points:
            widget_points = [self.image_to_widget_point(point) for point in self._closed_polygon_points]
            closed_points = [point for point in widget_points if point is not None]
            if len(closed_points) >= 3:
                painter.setPen(QColor("#f59e0b"))
                for point in closed_points:
                    painter.drawEllipse(point, 4, 4)
                for first, second in zip(closed_points, closed_points[1:]):
                    painter.drawLine(first, second)
                painter.drawLine(closed_points[-1], closed_points[0])
        for overlay in self._annotation_overlays:
            points_value = overlay.get("points", [])
            points: list[QPoint] = []
            if isinstance(points_value, list):
                for raw_point in points_value:
                    try:
                        px, py = raw_point  # type: ignore[misc]
                    except (TypeError, ValueError):
                        continue
                    widget_point = self.image_to_widget_point((int(px), int(py)))
                    if widget_point is not None:
                        points.append(widget_point)
            if not points:
                continue
            painter.setPen(QColor("#22c55e"))
            annotation_type = str(overlay.get("annotation_type", "pin"))
            if annotation_type == "polygon" and len(points) >= 3:
                for first, second in zip(points, points[1:]):
                    painter.drawLine(first, second)
                painter.drawLine(points[-1], points[0])
                for point in points:
                    painter.drawEllipse(point, 3, 3)
            else:
                point = points[0]
                painter.drawLine(point.x() - 7, point.y(), point.x() + 7, point.y())
                painter.drawLine(point.x(), point.y() - 7, point.x(), point.y() + 7)
                painter.drawEllipse(point, 5, 5)
        painter.setPen(QColor("#1f2937"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class NeonTextLabel(QLabel):
    def __init__(
        self,
        text: str,
        text_color: str,
        glow_color: str,
        glow_alpha: int = 130,
        glow_offsets: tuple[int, ...] = (4, 2),
    ) -> None:
        super().__init__(text)
        self.text_color = QColor(text_color)
        self.glow_color = QColor(glow_color)
        self.glow_alpha = glow_alpha
        self.glow_offsets = glow_offsets
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background: transparent; border: none; }")

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font())
        rect = self.rect()
        glow = QColor(self.glow_color)
        for index, radius in enumerate(self.glow_offsets):
            alpha = min(255, self.glow_alpha + index * 35)
            glow.setAlpha(alpha)
            painter.setPen(glow)
            for dx, dy in (
                (-radius, 0),
                (radius, 0),
                (0, -radius),
                (0, radius),
                (-radius, -radius),
                (radius, radius),
            ):
                painter.drawText(rect.adjusted(dx, dy, dx, dy), self.alignment(), self.text())
        painter.setPen(self.text_color)
        painter.drawText(rect, self.alignment(), self.text())
