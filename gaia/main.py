from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys

from qtpy.QtWidgets import QApplication, QMessageBox

from gaia.config import BACKENDS, FORMATS, HDF5_EXTENSIONS, VIDEO_EXTENSIONS
from gaia.core.hdf5_io import hdf5_dataset_catalog
from gaia.core.media_io import load_image, load_video_frames
from gaia.ui.main_window import FastQtPlatform
from gaia.ui.theme import GAIA_STYLE

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAIA multimodal sensing analysis platform.")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--capture-fps", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default="DirectShow" if platform.system() == "Windows" else "Default",
    )
    parser.add_argument("--format", choices=list(FORMATS), default="Auto")
    parser.add_argument("--open", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyleSheet(GAIA_STYLE)
    window = FastQtPlatform(
        camera_index=args.camera,
        backend_name=args.backend,
        pixel_format=args.format,
        width=args.width,
        height=args.height,
        capture_fps=args.capture_fps,
    )
    screen = app.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        window.resize(
            min(window.width(), max(900, available.width() - 80)),
            min(window.height(), max(650, available.height() - 80)),
        )
        frame = window.frameGeometry()
        frame.moveCenter(available.center())
        window.move(frame.topLeft())
    window.show()
    window.raise_()
    window.activateWindow()

    if args.open:
        window.show_image_video_workflow()
        try:
            if args.open.suffix.lower() in VIDEO_EXTENSIONS:
                window.video_frames = load_video_frames(args.open)
                window.current_source = f"video:{args.open.name}"
                window.frame_slider.setEnabled(True)
                window.frame_slider.setRange(0, len(window.video_frames) - 1)
                window.set_frame(window.video_frames[0])
                window.status_label.setText(f"Video: {args.open.name} ({len(window.video_frames)} frames)")
            elif args.open.suffix.lower() in HDF5_EXTENSIONS:
                catalog = hdf5_dataset_catalog(args.open)
                groups: dict[str, dict[str, dict[str, object]]] = catalog["groups"]  # type: ignore[assignment]
                samples: list[str] = catalog["common_samples"] or catalog["all_samples"]  # type: ignore[assignment]
                input_group = "noisy" if "noisy" in groups else "noisy_1" if "noisy_1" in groups else sorted(groups)[0]
                reference_group = "clean" if "clean" in groups else input_group
                sample = samples[0]
                shape = tuple(groups[input_group][sample]["shape"])  # type: ignore[index]
                z_index = int(shape[0]) // 2 if len(shape) == 3 and shape[-1] not in (3, 4) else 0
                selected = {
                    "sample": sample,
                    "input_group": input_group,
                    "reference_group": reference_group,
                    "z_index": z_index,
                }
                window.populate_hdf5_controls(args.open, catalog, selected)
                window.load_current_hdf5_selection()
            else:
                window.current_source = f"image:{args.open.name}"
                window.set_frame(load_image(args.open))
                window.status_label.setText(f"Image: {args.open.name}")
        except Exception as exc:
            QMessageBox.warning(window, "Open failed", str(exc))

    return app.exec_()
