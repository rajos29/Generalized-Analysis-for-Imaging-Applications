from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget
try:
    from vispy import scene
    from vispy.color import get_colormap
except ImportError:
    scene = None
    get_colormap = None

from gaia.core.image_ops import ensure_rgb, rgb_to_luminance

class VispySurfaceWidget(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        if scene is None or get_colormap is None:
            raise RuntimeError("VisPy is not available.")
        self.title = title
        self.mesh = None
        self.default_elevation = 35
        self.default_azimuth = -55
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="#05070a", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=45,
            elevation=self.default_elevation,
            azimuth=self.default_azimuth,
            distance=850,
        )
        self.axis_lines = []
        self.axis_labels = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)

    def clear_axes(self) -> None:
        for visual in self.axis_lines + self.axis_labels:
            visual.parent = None
        self.axis_lines = []
        self.axis_labels = []

    def draw_axes(self, width: int, height: int) -> None:
        self.clear_axes()
        x0 = -width / 2
        x1 = width / 2
        y0 = -height / 2
        y1 = height / 2
        axes = (
            (np.array([[x0, y0, 0], [x1, y0, 0]], dtype=np.float32), (0.15, 0.39, 0.92, 1.0), "x px", (x1 + 36, y0, 0)),
            (np.array([[x0, y0, 0], [x0, y1, 0]], dtype=np.float32), (0.13, 0.77, 0.37, 1.0), "y px", (x0, y1 + 36, 0)),
            (np.array([[x0, y0, 0], [x0, y0, 255]], dtype=np.float32), (0.53, 0.94, 0.67, 1.0), "v 0-255", (x0, y0, 295)),
        )
        for points, color, label, position in axes:
            line = scene.visuals.Line(pos=points, color=color, width=3, parent=self.view.scene)
            text = scene.visuals.Text(
                label,
                pos=position,
                color=color,
                font_size=16,
                bold=True,
                parent=self.view.scene,
            )
            self.axis_lines.append(line)
            self.axis_labels.append(text)

    def set_surface(self, frame_rgb: np.ndarray | None, step: int, title: str) -> None:
        if frame_rgb is None:
            if self.mesh is not None:
                self.mesh.parent = None
                self.mesh = None
            self.clear_axes()
            self.canvas.update()
            return
        gray = rgb_to_luminance(ensure_rgb(frame_rgb)).astype(np.float32)
        step = max(1, int(step))
        z = gray[::step, ::step]
        y_values = np.arange(0, gray.shape[0], step, dtype=np.float32)
        x_values = np.arange(0, gray.shape[1], step, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        vertices = np.column_stack(
            (
                x_grid.ravel() - gray.shape[1] / 2,
                y_grid.ravel() - gray.shape[0] / 2,
                z.ravel(),
            )
        ).astype(np.float32)
        rows, cols = z.shape
        base = np.arange((rows - 1) * (cols - 1), dtype=np.uint32)
        row = base // (cols - 1)
        col = base % (cols - 1)
        v0 = row * cols + col
        v1 = v0 + 1
        v2 = v0 + cols
        v3 = v2 + 1
        faces = np.vstack(
            (
                np.column_stack((v0, v1, v2)),
                np.column_stack((v1, v3, v2)),
            )
        ).astype(np.uint32)
        normalized = z / 255.0
        colors = get_colormap("viridis").map(normalized.ravel()).astype(np.float32)
        if self.mesh is None:
            self.mesh = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=colors,
                parent=self.view.scene,
            )
        else:
            self.mesh.set_data(vertices=vertices, faces=faces, vertex_colors=colors)
        self.draw_axes(gray.shape[1], gray.shape[0])
        self.title = title
        self.canvas.update()

    def reset_view(self) -> None:
        self.view.camera.elevation = self.default_elevation
        self.view.camera.azimuth = self.default_azimuth
        self.view.camera.distance = 850
        self.canvas.update()


