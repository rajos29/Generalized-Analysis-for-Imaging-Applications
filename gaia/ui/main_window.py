from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvas = None
    Figure = None
try:
    from vispy import scene
    from vispy.color import get_colormap
except ImportError:
    scene = None
    get_colormap = None
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QColor, QFont, QIcon, QPixmap, QTextListFormat
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout,
    QHeaderView, QHBoxLayout, QGraphicsDropShadowEffect, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from gaia.annotations.store import append_jsonl, json_safe, read_json, read_jsonl, write_json
from gaia.config import (
    ALGORITHMS, ANNOTATIONS_PATH, APP_DIR, BACKENDS, DEFAULT_IDRT_DATASET_DIR, FORMATS,
    GENERIC_SOURCES, HDF5_EXTENSIONS, IDRT_PAIRS, METRIC_NAMES, PANE_SOURCES, SATELLITE_SOURCES,
    VIDEO_EXTENSIONS,
)
from gaia.core.hdf5_io import (
    frame_from_array, hdf5_dataset_catalog, load_hdf5_sample_frame, list_hdf5_image_datasets,
)
from gaia.core.image_ops import (
    biobridge_gradient_hessian_operator, ensure_rgb, process_frame, rgb_to_luminance,
)
from gaia.core.media_io import load_image, load_video_frames
from gaia.core.metrics import (
    compare_to_reference, compare_variance_maps, compute_processed_z_variance_map, compute_z_variance_map,
    compute_z_variance_metrics, empty_variance_qa_metrics, empty_z_variance_metrics, image_metrics,
    load_hdf5_stack, normalize_float_map, processed_metrics, residual_heatmap_rgb, sensor_domain_metrics,
    template_match_metrics, variance_difference_display_map,
    z_variance_metrics_from_map,
)
from gaia.core.satellite import (
    estimate_satellite_affine_alignment, estimate_satellite_projective_alignment, satellite_change_heatmap,
    satellite_change_mask, satellite_change_metrics, satellite_shared_gxh_reference, resize_like, transform_frame_affine,
    transform_frame_projective,
)
from gaia.core.template_matching import template_match_search
from gaia.experiments.exporter import experiment_id, list_experiments, summarize_metrics_csv
from gaia.ui.dialogs import Hdf5DatasetDialog
from gaia.ui.image_view import FpsTracker, NeonTextLabel, ZoomableImageView
from gaia.ui.surface import VispySurfaceWidget
from gaia.workers.camera import CameraWorker
from gaia.workers.ender3 import Ender3Worker

class FastQtPlatform(QMainWindow):
    def __init__(
        self,
        camera_index: int,
        backend_name: str,
        pixel_format: str,
        width: int,
        height: int,
        capture_fps: int | None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("GAIA - Generalized Analysis for Imaging Applications")
        self.resize(1500, 900)

        self.requested_width = width
        self.requested_height = height
        self.capture_fps = capture_fps
        self.camera_worker: CameraWorker | None = None
        self.ender_worker: Ender3Worker | None = None
        self.video_frames: list[np.ndarray] = []
        self.video_index = 0
        self.fps = FpsTracker()
        self.show_processed = True
        self.satellite_pre_frame: np.ndarray | None = None
        self.satellite_post_frame: np.ndarray | None = None
        self.satellite_shared_gxh_frame: np.ndarray | None = None
        self.satellite_shared_gxh_mask: np.ndarray | None = None
        self.satellite_homography: np.ndarray | None = None
        self.satellite_homography_summary = ""
        self.satellite_homography_score: float | str = ""
        self.satellite_homography_inliers: int | str = ""
        self.satellite_pair_name = ""
        self.current_frame: np.ndarray | None = None
        self.previous_frame: np.ndarray | None = None
        self.comparison_frame: np.ndarray | None = None
        self.current_source = "idle"
        self.frame_counter = 0
        self.last_processed: np.ndarray | None = None
        self.latest_metrics: dict[str, object] = {}
        self.adjacent_similarity_percent = 100.0
        self.template_match_template: np.ndarray | None = None
        self.template_match_mask: np.ndarray | None = None
        self.template_match_source = ""
        self.template_match_result: dict[str, object] = {}
        self.template_crop_base_label = "Raw/Input"
        self.annotations: list[dict[str, object]] = []
        self.visible_annotations: list[dict[str, object]] = []
        self.pending_annotation: dict[str, object] | None = None
        self.annotation_capture_source = ""
        self.annotation_capture_view: ZoomableImageView | None = None
        self._pane_refreshing = False
        self.log_file = None
        self.log_writer: csv.DictWriter | None = None
        self.active_experiment_dir: Path | None = None
        self.hdf5_path: Path | None = None
        self.hdf5_catalog: dict[str, object] | None = None
        self.hdf5_reference_frame: np.ndarray | None = None
        self.hdf5_loading_controls = False
        self.z_variance_metrics: dict[str, float | str] = empty_z_variance_metrics()
        self.variance_qa_metrics: dict[str, float | str] = empty_variance_qa_metrics()
        self.z_variance_map: np.ndarray | None = None
        self.reference_z_variance_map: np.ndarray | None = None
        self.processed_z_variance_map: np.ndarray | None = None
        self.processed_z_variance_cache_key: tuple[object, ...] | None = None
        self.show_advanced_view = False

        self.stack = QStackedWidget()
        self.home_screen = QWidget()
        self.image_video_screen = QWidget()
        self.camera_screen = QWidget()
        self.experiments_screen = QWidget()
        self.analysis_workspace = QWidget()
        self.start_image_video_button = QPushButton("Image / Video Analysis")
        self.start_camera_button = QPushButton("Camera Acquisition")
        self.open_experiments_button = QPushButton("Open Experiment")
        self.future_workflow_label = QLabel("Sensor domains: satellite imagery, radar-like spectral maps, LiDAR relief, ultrasonic envelopes, microscopy arrays")
        self.image_video_back_button = QPushButton("Back")
        self.camera_back_button = QPushButton("Back")
        self.go_camera_workflow_button = QPushButton("Camera Acquisition")
        self.go_image_video_workflow_button = QPushButton("Image / Video Analysis")
        self.experiments_back_button = QPushButton("Back")
        self.refresh_experiments_button = QPushButton("Refresh")
        self.experiments_list = QListWidget()
        self.experiment_details = QTableWidget(0, 2)
        self.experiment_details.setHorizontalHeaderLabels(["Field", "Value"])
        self.experiment_details.verticalHeader().setVisible(False)
        self.experiment_details.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.experiment_details.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.experiment_details.setAlternatingRowColors(True)
        self.experiment_summary_label = QLabel("Select an experiment to view metrics.")

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.setValue(camera_index)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(BACKENDS.keys())
        self.backend_combo.setCurrentText(backend_name)
        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMATS.keys())
        self.format_combo.setCurrentText(pixel_format)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(ALGORITHMS)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(50)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 255)
        self.threshold_slider.setValue(80)
        self.process_every_spin = QSpinBox()
        self.process_every_spin.setRange(1, 60)
        self.process_every_spin.setValue(1)
        self.operator_target_combo = QComboBox()
        self.operator_target_combo.addItems(("Post/current image", "Pre/reference image"))
        self.operator_target_combo.setToolTip("Choose which image single-image operators process in paired satellite mode.")
        self.template_base_combo = QComboBox()
        self.template_base_combo.addItems(("Raw/Input", "Processed Output", "Reference/Pre"))
        self.template_base_combo.setToolTip("Choose which visible image GAIA searches inside.")
        self.template_filter_source_combo = QComboBox()
        self.template_filter_source_combo.addItems(("Raw/Input", "Processed Output", "Reference/Pre", "Shared GxH", "Current GxH"))
        self.template_filter_source_combo.setToolTip("Choose which image GAIA crops the search filter from.")
        self.left_pane_combo = QComboBox()
        self.center_pane_combo = QComboBox()
        self.right_pane_combo = QComboBox()
        for combo in (self.left_pane_combo, self.center_pane_combo, self.right_pane_combo):
            combo.addItems(PANE_SOURCES)
            combo.setToolTip("Choose which data product this display pane shows.")
        self.left_pane_combo.setCurrentText("Raw/Input")
        self.center_pane_combo.setCurrentText("Processed Output")
        self.right_pane_combo.setCurrentText("Reference/Pre")
        self.template_load_button = QPushButton("Load Filter")
        self.template_crop_button = QPushButton("Crop Filter")
        self.template_polygon_button = QPushButton("Polygon Filter")
        self.template_run_button = QPushButton("Run Image Search")
        self.template_help_button = QPushButton("CIS Help")
        self.template_clear_button = QPushButton("Clear Search")
        self.template_stride_spin = QSpinBox()
        self.template_stride_spin.setRange(1, 64)
        self.template_stride_spin.setValue(4)
        self.template_stride_spin.setSuffix(" px")
        self.template_sweep_mode_combo = QComboBox()
        self.template_sweep_mode_combo.addItems(("Light 90 deg", "Deep 45 deg", "Custom fine"))
        self.template_sweep_mode_combo.setToolTip("Choose the CIS rotation sweep. Light checks 0/90/180/270. Deep checks the full circle every 45 degrees. Custom uses the min/max/step boxes.")
        self.template_angle_min_spin = QDoubleSpinBox()
        self.template_angle_max_spin = QDoubleSpinBox()
        self.template_angle_step_spin = QDoubleSpinBox()
        for spinbox in (self.template_angle_min_spin, self.template_angle_max_spin, self.template_angle_step_spin):
            spinbox.setRange(-180.0, 180.0)
            spinbox.setDecimals(1)
            spinbox.setSuffix(" deg")
        self.template_angle_min_spin.setValue(-15.0)
        self.template_angle_max_spin.setValue(15.0)
        self.template_angle_step_spin.setRange(0.5, 90.0)
        self.template_angle_step_spin.setValue(3.0)
        self.template_load_button.setToolTip("Load a desired object/filter image for Convolutional Image Search.")
        self.template_crop_button.setToolTip("Draw a crop box on the selected filter-source image to create the search filter.")
        self.template_polygon_button.setToolTip("Click polygon points around an object on the selected filter source; right-click to close and create a blank-background kernel.")
        self.template_run_button.setToolTip("Run grayscale template matching with rotation sweep and show the match heatmap.")
        self.template_help_button.setToolTip("Show the exact CIS method, current sweep angles, and metric meanings.")
        self.template_clear_button.setToolTip("Clear the current search filter and match result.")
        self.template_sweep_label = QLabel("")
        self.template_sweep_label.setStyleSheet("QLabel { color: #9ca3af; font-size: 9pt; }")
        self.align_x_spin = QSpinBox()
        self.align_y_spin = QSpinBox()
        self.align_rotation_spin = QDoubleSpinBox()
        self.align_scale_spin = QDoubleSpinBox()
        self.auto_alignment_button = QPushButton("Auto Align")
        self.auto_tilt_button = QPushButton("Auto Tilt")
        self.reset_alignment_button = QPushButton("Reset Align")
        for spinbox in (self.align_x_spin, self.align_y_spin):
            spinbox.setRange(-500, 500)
            spinbox.setSuffix(" px")
            spinbox.setToolTip("Translate the pre/reference image before change detection.")
        self.align_rotation_spin.setRange(-15.0, 15.0)
        self.align_rotation_spin.setDecimals(2)
        self.align_rotation_spin.setSingleStep(0.25)
        self.align_rotation_spin.setSuffix(" deg")
        self.align_rotation_spin.setToolTip("Rotate the pre/reference image before change detection.")
        self.align_scale_spin.setRange(80.0, 120.0)
        self.align_scale_spin.setDecimals(2)
        self.align_scale_spin.setSingleStep(0.25)
        self.align_scale_spin.setValue(100.0)
        self.align_scale_spin.setSuffix(" %")
        self.align_scale_spin.setToolTip("Scale the pre/reference image before change detection.")
        self.auto_alignment_button.setToolTip("Estimate pre/reference alignment with a downsampled affine search. Runs only when clicked.")
        self.auto_tilt_button.setEnabled(False)
        self.auto_tilt_button.setToolTip("Future work: projective tilt/homography alignment for view-angle and roof-parallax mismatch.")

        self.original_only_button = QPushButton("Original Only")
        self.analysis_button = QPushButton("Processed View")
        self.advanced_view_button = QPushButton("Image Space View")
        self.advanced_view_button.setEnabled(False)
        self.advanced_view_button.setToolTip("Show x/y/grayscale 3D surfaces for the processed image and available reference/current image.")
        self.open_camera_button = QPushButton("Open Camera")
        self.open_image_button = QPushButton("Open Image")
        self.open_idrt_button = QPushButton("Open IDRT Pair")
        self.open_video_button = QPushButton("Open Video")
        self.ender_port_combo = QComboBox()
        self.ender_port_combo.setEditable(True)
        self.ender_port_combo.setMinimumWidth(140)
        self.refresh_ender_ports_button = QPushButton("Refresh Ports")
        self.ender_baud_combo = QComboBox()
        self.ender_baud_combo.addItems(("115200", "250000", "57600"))
        self.ender_baud_combo.setCurrentText("115200")
        self.ender_connect_button = QPushButton("Connect Ender")
        self.ender_disconnect_button = QPushButton("Disconnect")
        self.ender_disconnect_button.setEnabled(False)
        self.ender_home_button = QPushButton("Home XYZ")
        self.ender_position_button = QPushButton("Get Position")
        self.ender_soft_stop_button = QPushButton("Soft Stop")
        self.ender_emergency_stop_button = QPushButton("EMERGENCY STOP")
        self.ender_emergency_stop_button.setToolTip("Sends M112. Firmware may require a reset after this.")
        self.ender_step_spin = QDoubleSpinBox()
        self.ender_step_spin.setRange(0.01, 50.0)
        self.ender_step_spin.setDecimals(2)
        self.ender_step_spin.setValue(1.0)
        self.ender_step_spin.setSuffix(" mm")
        self.ender_feed_spin = QSpinBox()
        self.ender_feed_spin.setRange(60, 6000)
        self.ender_feed_spin.setValue(600)
        self.ender_feed_spin.setSuffix(" mm/min")
        self.ender_max_x_spin = QDoubleSpinBox()
        self.ender_max_y_spin = QDoubleSpinBox()
        self.ender_max_z_spin = QDoubleSpinBox()
        for spinbox, value in (
            (self.ender_max_x_spin, 220.0),
            (self.ender_max_y_spin, 220.0),
            (self.ender_max_z_spin, 250.0),
        ):
            spinbox.setRange(1.0, 1000.0)
            spinbox.setDecimals(1)
            spinbox.setValue(value)
            spinbox.setSuffix(" mm")
        self.ender_x_minus_button = QPushButton("X-")
        self.ender_x_plus_button = QPushButton("X+")
        self.ender_y_minus_button = QPushButton("Y-")
        self.ender_y_plus_button = QPushButton("Y+")
        self.ender_z_minus_button = QPushButton("Z-")
        self.ender_z_plus_button = QPushButton("Z+")
        self.ender_position_label = QLabel("Position X -- | Y -- | Z --")
        self.ender_status_label = QLabel("Ender disconnected")
        self.ender_log_label = QLabel("Motion log idle")
        self.ender_motion_buttons = (
            self.ender_home_button,
            self.ender_position_button,
            self.ender_soft_stop_button,
            self.ender_emergency_stop_button,
            self.ender_x_minus_button,
            self.ender_x_plus_button,
            self.ender_y_minus_button,
            self.ender_y_plus_button,
            self.ender_z_minus_button,
            self.ender_z_plus_button,
        )
        for button in self.ender_motion_buttons:
            button.setEnabled(False)
        self.play_button = QPushButton("Play")
        self.save_experiment_button = QPushButton("Save Experiment")
        self.save_experiment_button.setToolTip("Save manifest, current metrics, and current frame snapshots into experiments/.")
        self.save_row_button = QPushButton("Save Row")
        self.save_row_button.setToolTip("Save the current measurement table row to a CSV file.")
        self.start_log_button = QPushButton("Start CSV Log")
        self.start_log_button.setToolTip("Continuously append the live measurement table values to a CSV file.")
        self.stop_log_button = QPushButton("Stop Log")
        self.stop_log_button.setToolTip("Stop the active CSV log.")
        self.stop_log_button.setEnabled(False)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.operation_progress = QProgressBar()
        self.operation_progress.setRange(0, 100)
        self.operation_progress.setValue(0)
        self.operation_progress.setFormat("Idle")
        self.operation_progress.setToolTip("Shows progress for prompted operations such as image search, auto alignment, and processing refreshes.")
        self.hdf5_controls = QWidget()
        self.hdf5_file_label = QLabel("No HDF5 dataset loaded")
        self.hdf5_sample_combo = QComboBox()
        self.hdf5_input_combo = QComboBox()
        self.hdf5_reference_combo = QComboBox()
        self.hdf5_z_spin = QSpinBox()
        self.hdf5_z_spin.setMinimumSize(QSize(140, 30))
        self.hdf5_z_down_button = QPushButton("-")
        self.hdf5_z_up_button = QPushButton("+")
        for button in (self.hdf5_z_down_button, self.hdf5_z_up_button):
            button.setFixedSize(QSize(34, 30))
        self.hdf5_prev_button = QPushButton("Previous Sample")
        self.hdf5_next_button = QPushButton("Next Sample")
        self.surface_downsample_spin = QSpinBox()
        self.surface_downsample_spin.setRange(1, 32)
        self.surface_downsample_spin.setValue(8)
        self.surface_downsample_spin.setMinimumSize(QSize(120, 30))
        self.surface_downsample_spin.setToolTip("Render every Nth pixel in image-space 3D surfaces.")
        self.surface_downsample_down_button = QPushButton("-")
        self.surface_downsample_up_button = QPushButton("+")
        for button in (self.surface_downsample_down_button, self.surface_downsample_up_button):
            button.setFixedSize(QSize(34, 30))
        self.surface_home_button = QPushButton("Reset View")
        self.surface_home_button.setToolTip("Reset the 3D surface camera angles.")
        self.image_zoom_in_button = QPushButton("Zoom In")
        self.image_zoom_out_button = QPushButton("Zoom Out")
        self.image_reset_view_button = QPushButton("Reset Image View")
        self.image_zoom_in_button.setToolTip("Zoom into all standard image panes. Mouse wheel works over each pane too.")
        self.image_zoom_out_button.setToolTip("Zoom out of all standard image panes.")
        self.image_reset_view_button.setToolTip("Reset standard image panes to fit the available space.")
        self.annotation_pin_button = QPushButton("Pin Note")
        self.annotation_polygon_button = QPushButton("Polygon Note")
        self.annotation_confirm_button = QPushButton("Confirm Note")
        self.annotation_cancel_button = QPushButton("Cancel Note")
        self.annotation_clear_draft_button = QPushButton("Clear Draft")
        self.annotation_pin_button.setToolTip("Click a visible image pane to drop an evidence pin and open the note editor.")
        self.annotation_polygon_button.setToolTip("Draw an evidence polygon on a visible image pane. Right-click closes it and opens the note editor.")
        self.annotation_confirm_button.setToolTip("Save the current annotation note to experiments/annotations.jsonl.")
        self.annotation_cancel_button.setToolTip("Cancel annotation capture mode without saving.")
        self.annotation_clear_draft_button.setToolTip("Clear the current draft note and pending annotation.")
        self.qa_mode_combo = QComboBox()
        self.qa_mode_combo.addItems(("Denoising QA", "Variance QA"))
        self.qa_mode_combo.setToolTip("Choose whether the fourth pane shows denoising residual error or z-variance analysis.")
        self.residual_map_combo = QComboBox()
        self.residual_map_combo.addItems(("Processed - Reference Residual", "Input - Reference Residual"))
        self.residual_map_combo.setToolTip("Choose whether residual error compares processed output or raw input against the clean reference.")
        self.variance_norm_combo = QComboBox()
        self.variance_norm_combo.addItems(("Full Range", "Percentile Clip"))
        self.variance_norm_combo.setToolTip("Choose how z-variance values are normalized into the heatmap.")
        self.variance_map_combo = QComboBox()
        self.variance_map_combo.addItems(
            (
                "Input Variance",
                "Reference Variance",
                "Input - Reference Difference",
                "Processed - Reference Difference",
            )
        )
        self.variance_map_combo.setToolTip("Choose which z-variance QA map to show.")
        self.variance_clip_spin = QSpinBox()
        self.variance_clip_spin.setRange(90, 100)
        self.variance_clip_spin.setValue(99)
        self.variance_clip_spin.setSuffix(" %")
        self.variance_clip_spin.setMinimumSize(QSize(96, 30))
        self.variance_clip_spin.setToolTip("Upper percentile used for clipped heatmap normalization and overlay threshold.")
        self.variance_threshold_check = QCheckBox("Threshold Overlay")
        self.variance_threshold_check.setToolTip("Highlight pixels above the selected z-variance percentile.")
        self.export_variance_button = QPushButton("Export Variance")
        self.export_variance_button.setToolTip("Save the current variance heatmap as a PNG.")
        self.last_variance_heatmap_rgb: np.ndarray | None = None
        self.last_residual_heatmap_rgb: np.ndarray | None = None

        self.status_label = QLabel("Idle")
        self.metrics_label = QLabel("FPS --")
        self.algorithm_label = QLabel("Algorithm: Gradient x Hessian | strength 50 | threshold 80")
        self.log_label = QLabel("CSV log idle")
        self.metrics_help_label = QLabel("Evidence table: current settings, alignment/search parameters, and exportable quantitative outputs.")
        self.metrics_help_label.setWordWrap(True)
        self.metrics_help_label.setStyleSheet("QLabel { color: #9ca3af; font-size: 9pt; }")
        self.evidence_tabs = QTabWidget()
        self.metrics_tab = QWidget()
        self.annotations_tab = QWidget()
        self.notes_tab = QWidget()
        self.annotations_table = QTableWidget(0, 6)
        self.annotations_table.setHorizontalHeaderLabels(["ID", "Type", "Source", "X", "Y", "Label"])
        self.annotations_table.verticalHeader().setVisible(False)
        self.annotations_table.setAlternatingRowColors(True)
        self.annotations_table.setWordWrap(False)
        self.annotations_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.annotations_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.annotations_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.annotations_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.annotations_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.annotations_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.annotations_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.annotations_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.annotation_label_edit = QLineEdit()
        self.annotation_label_edit.setPlaceholderText("Annotation label")
        self.annotation_context_label = QLabel("No active annotation.")
        self.annotation_context_label.setWordWrap(True)
        self.annotation_context_label.setStyleSheet("QLabel { color: #9ca3af; font-size: 9pt; }")
        self.annotation_notes_edit = QTextEdit()
        self.annotation_notes_edit.setPlaceholderText("Write analyst notes for this pin or polygon...")
        self.annotation_notes_edit.setAcceptRichText(True)
        self.annotation_notes_edit.setMinimumHeight(170)
        self.annotation_bold_button = QPushButton("B")
        self.annotation_italic_button = QPushButton("I")
        self.annotation_underline_button = QPushButton("U")
        self.annotation_bullet_button = QPushButton("Bullets")
        for button in (
            self.annotation_bold_button,
            self.annotation_italic_button,
            self.annotation_underline_button,
            self.annotation_bullet_button,
        ):
            button.setMinimumWidth(42)
        self.input_title = QLabel("Raw Input")
        self.processed_title = QLabel("Processed")
        self.original_title = QLabel("Original / Reference")
        self.input_label = ZoomableImageView("Input")
        self.processed_label = ZoomableImageView("Gradient x Hessian")
        self.original_label = ZoomableImageView("Original")
        self.advanced_input_label = QLabel("Noisy/Input")
        self.advanced_reference_label = QLabel("Clean/Reference")
        self.advanced_input_surface_label = QLabel("Input Surface")
        self.advanced_reference_surface_label = QLabel("Reference Surface")
        self.views_stack = QStackedWidget()
        self.standard_views_widget = QWidget()
        self.advanced_views_widget = QWidget()
        self.advanced_input_canvas = self.create_surface_canvas("Processed Surface")
        self.advanced_reference_canvas = self.create_surface_canvas("Reference Surface")
        self.variance_heatmap_title = QLabel("Variance Heatmap")
        self.variance_heatmap_label = ZoomableImageView("No z-stack variance map loaded")
        self.input_panel: QWidget | None = None
        self.variance_panel: QWidget | None = None
        self.metrics_table = QTableWidget(len(METRIC_NAMES), 2)
        self.metrics_table.setToolTip("Live measurements for the current frame and processed output.")
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setMinimumWidth(520)
        self.metrics_table.setMinimumHeight(320)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setWordWrap(False)
        self.metrics_table.setTextElideMode(Qt.ElideNone)
        for row, name in enumerate(METRIC_NAMES):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem("--"))
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setDefaultSectionSize(28)
        for label in (
            self.input_label,
            self.processed_label,
            self.original_label,
            self.advanced_input_label,
            self.advanced_reference_label,
            self.advanced_input_surface_label,
            self.advanced_reference_surface_label,
            self.variance_heatmap_label,
        ):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(320, 240)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setStyleSheet(
                "QLabel { background: #05070a; color: #38bdf8; border: 1px solid #1f2937; }"
            )

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.advance_video)
        self.video_timer.setInterval(1)
        self.processing_update_timer = QTimer(self)
        self.processing_update_timer.setSingleShot(True)
        self.processing_update_timer.setInterval(160)
        self.processing_update_timer.timeout.connect(self.update_algorithm_label)
        self.alignment_update_timer = QTimer(self)
        self.alignment_update_timer.setSingleShot(True)
        self.alignment_update_timer.setInterval(220)
        self.alignment_update_timer.timeout.connect(self.update_satellite_alignment)
        self.surface_render_timer = QTimer(self)
        self.surface_render_timer.setSingleShot(True)
        self.surface_render_timer.setInterval(180)
        self.surface_render_timer.timeout.connect(self.render_image_space_surfaces)

        self._build_ui()
        self.apply_instrument_glow()
        self._connect()
        self.update_template_sweep_controls()
        self.set_display_mode(True)
        self.load_annotations()
        self.show_home()

    def _build_ui(self) -> None:
        self._build_home_screen()
        self._build_image_video_screen()
        self._build_camera_screen()
        self._build_analysis_workspace()
        self._build_experiments_screen()
        self.stack.addWidget(self.home_screen)
        self.stack.addWidget(self.image_video_screen)
        self.stack.addWidget(self.camera_screen)
        self.stack.addWidget(self.experiments_screen)
        self.setCentralWidget(self.stack)

    def _build_home_screen(self) -> None:
        layout = QVBoxLayout(self.home_screen)
        layout.setContentsMargins(48, 44, 48, 44)
        app_icon = QLabel()
        app_icon.setAlignment(Qt.AlignCenter)
        app_icon.setFixedSize(QSize(1620, 293))
        splash_path = APP_DIR / "assets" / "gaia_ascii_splash.png"
        if splash_path.exists():
            app_icon.setPixmap(QPixmap(str(splash_path)).scaled(app_icon.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            app_icon.setText("## G.A.I.A. ##")
        app_icon.setStyleSheet(
            "QLabel { background: transparent; border: none; color: #38bdf8; "
            "font-family: Consolas, 'Courier New', monospace; font-size: 48pt; "
            "font-weight: 800; letter-spacing: 0px; }"
        )
        self.add_neon_glow(app_icon, "#2563eb", blur=18, alpha=80)
        subtitle = QLabel("Image analysis for microscopy, satellite, and sensor data.")
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setMinimumHeight(40)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("QLabel { background: transparent; border: none; font-size: 14pt; color: #d1d5db; font-weight: 500; }")
        for button in (
            self.start_image_video_button,
            self.start_camera_button,
            self.open_experiments_button,
        ):
            button.setMinimumHeight(96)
            button.setStyleSheet("QPushButton { font-size: 18pt; }")
        self.future_workflow_label.setAlignment(Qt.AlignCenter)
        self.future_workflow_label.setText("Coming soon: radar, LiDAR, ultrasonic, and more.")
        self.future_workflow_label.setStyleSheet("QLabel { color: #9ca3af; font-size: 10pt; }")

        cards = QHBoxLayout()
        cards.addStretch(1)
        cards.addWidget(self.start_image_video_button, 2)
        cards.addWidget(self.start_camera_button, 2)
        cards.addWidget(self.open_experiments_button, 2)
        cards.addStretch(1)

        layout.addStretch(1)
        icon_row = QHBoxLayout()
        icon_row.addStretch(1)
        icon_row.addWidget(app_icon)
        icon_row.addStretch(1)
        layout.addLayout(icon_row)
        layout.addSpacing(2)
        layout.addWidget(subtitle)
        layout.addSpacing(30)
        layout.addLayout(cards)
        layout.addSpacing(18)
        layout.addWidget(self.future_workflow_label)
        layout.addSpacing(22)
        layout.addStretch(2)

    def _build_image_video_screen(self) -> None:
        root_layout = QVBoxLayout(self.image_video_screen)

        controls = QGridLayout()
        controls.addWidget(self.image_video_back_button, 0, 0)
        controls.addWidget(self.open_image_button, 0, 1, 1, 2)
        controls.addWidget(self.open_idrt_button, 0, 3, 1, 2)
        controls.addWidget(self.open_video_button, 0, 5, 1, 2)
        controls.addWidget(self.play_button, 0, 7)
        controls.addWidget(self.go_camera_workflow_button, 0, 8)
        hdf5_layout = QGridLayout(self.hdf5_controls)
        hdf5_layout.setHorizontalSpacing(10)
        hdf5_layout.setVerticalSpacing(8)
        hdf5_layout.setColumnMinimumWidth(1, 140)
        hdf5_layout.setColumnMinimumWidth(2, 180)
        hdf5_layout.setColumnMinimumWidth(4, 210)
        hdf5_layout.setColumnMinimumWidth(6, 210)
        hdf5_layout.addWidget(self.hdf5_file_label, 0, 0, 1, 8)
        hdf5_layout.addWidget(self.hdf5_prev_button, 1, 0)
        hdf5_layout.addWidget(QLabel("Sample"), 1, 1)
        hdf5_layout.addWidget(self.hdf5_sample_combo, 1, 2)
        hdf5_layout.addWidget(QLabel("Input"), 1, 3)
        hdf5_layout.addWidget(self.hdf5_input_combo, 1, 4)
        hdf5_layout.addWidget(QLabel("Reference"), 1, 5)
        hdf5_layout.addWidget(self.hdf5_reference_combo, 1, 6)
        hdf5_layout.addWidget(self.hdf5_next_button, 1, 7)
        hdf5_layout.addWidget(QLabel("Z slice"), 2, 0)
        hdf5_layout.addWidget(self.make_stepper(self.hdf5_z_down_button, self.hdf5_z_spin, self.hdf5_z_up_button), 2, 1, 1, 2)
        hdf5_layout.addWidget(QLabel("Surface downsample"), 2, 3)
        hdf5_layout.addWidget(
            self.make_stepper(
                self.surface_downsample_down_button,
                self.surface_downsample_spin,
                self.surface_downsample_up_button,
            ),
            2,
            4,
            1,
            2,
        )
        hdf5_layout.addWidget(self.surface_home_button, 2, 6)
        hdf5_layout.addWidget(QLabel("QA mode"), 3, 0)
        hdf5_layout.addWidget(self.qa_mode_combo, 3, 1, 1, 2)
        hdf5_layout.addWidget(QLabel("Variance norm"), 3, 3)
        hdf5_layout.addWidget(self.variance_norm_combo, 3, 4, 1, 2)
        hdf5_layout.addWidget(QLabel("Map"), 3, 6)
        hdf5_layout.addWidget(self.variance_map_combo, 3, 7)
        hdf5_layout.addWidget(QLabel("Clip"), 4, 0)
        hdf5_layout.addWidget(self.variance_clip_spin, 4, 1)
        hdf5_layout.addWidget(self.variance_threshold_check, 4, 2, 1, 2)
        hdf5_layout.addWidget(self.export_variance_button, 4, 4, 1, 2)
        hdf5_layout.addWidget(self.residual_map_combo, 4, 6, 1, 2)
        controls.addWidget(self.hdf5_controls, 1, 0, 1, 7)
        self.hdf5_controls.setVisible(False)
        root_layout.addLayout(controls)
        self.image_video_workspace_slot = QVBoxLayout()
        root_layout.addLayout(self.image_video_workspace_slot, 1)

    def _build_camera_screen(self) -> None:
        root_layout = QVBoxLayout(self.camera_screen)

        controls = QGridLayout()
        controls.addWidget(self.camera_back_button, 0, 0)
        controls.addWidget(QLabel("Camera"), 0, 1)
        controls.addWidget(self.camera_spin, 0, 2)
        controls.addWidget(QLabel("Backend"), 0, 3)
        controls.addWidget(self.backend_combo, 0, 4)
        controls.addWidget(QLabel("Format"), 0, 5)
        controls.addWidget(self.format_combo, 0, 6)
        controls.addWidget(self.open_camera_button, 0, 7)
        controls.addWidget(self.go_image_video_workflow_button, 0, 8)
        controls.addWidget(QLabel("Ender port"), 1, 0)
        controls.addWidget(self.ender_port_combo, 1, 1)
        controls.addWidget(self.refresh_ender_ports_button, 1, 2)
        controls.addWidget(QLabel("Baud"), 1, 3)
        controls.addWidget(self.ender_baud_combo, 1, 4)
        controls.addWidget(self.ender_connect_button, 1, 5)
        controls.addWidget(self.ender_disconnect_button, 1, 6)
        controls.addWidget(self.ender_position_label, 1, 7, 1, 2)
        controls.addWidget(QLabel("Step"), 2, 0)
        controls.addWidget(self.ender_step_spin, 2, 1)
        controls.addWidget(QLabel("Feed"), 2, 2)
        controls.addWidget(self.ender_feed_spin, 2, 3)
        controls.addWidget(self.ender_x_minus_button, 2, 4)
        controls.addWidget(self.ender_x_plus_button, 2, 5)
        controls.addWidget(self.ender_y_minus_button, 2, 6)
        controls.addWidget(self.ender_y_plus_button, 2, 7)
        controls.addWidget(self.ender_z_minus_button, 2, 8)
        controls.addWidget(self.ender_z_plus_button, 2, 9)
        controls.addWidget(QLabel("Soft limits X/Y/Z"), 3, 0)
        controls.addWidget(self.ender_max_x_spin, 3, 1)
        controls.addWidget(self.ender_max_y_spin, 3, 2)
        controls.addWidget(self.ender_max_z_spin, 3, 3)
        controls.addWidget(self.ender_home_button, 3, 4)
        controls.addWidget(self.ender_position_button, 3, 5)
        controls.addWidget(self.ender_soft_stop_button, 3, 6)
        controls.addWidget(self.ender_emergency_stop_button, 3, 7, 1, 2)
        controls.addWidget(self.ender_status_label, 4, 0, 1, 5)
        controls.addWidget(self.ender_log_label, 4, 5, 1, 5)
        root_layout.addLayout(controls)
        self.camera_workspace_slot = QVBoxLayout()
        root_layout.addLayout(self.camera_workspace_slot, 1)

    def _build_analysis_workspace(self) -> None:
        root_layout = QVBoxLayout(self.analysis_workspace)

        top_layout = QHBoxLayout()
        controls = QGridLayout()
        controls.addWidget(self.original_only_button, 0, 0)
        controls.addWidget(self.analysis_button, 0, 1)
        controls.addWidget(self.advanced_view_button, 0, 2)
        controls.addWidget(QLabel("Algorithm"), 1, 0)
        controls.addWidget(self.algorithm_combo, 1, 1, 1, 2)
        controls.addWidget(QLabel("Strength"), 1, 3)
        controls.addWidget(self.strength_slider, 1, 4)
        controls.addWidget(QLabel("Threshold"), 1, 5)
        controls.addWidget(self.threshold_slider, 1, 6)
        controls.addWidget(QLabel("Process every N frames"), 2, 0)
        controls.addWidget(self.process_every_spin, 2, 1)
        controls.addWidget(self.save_row_button, 2, 2)
        controls.addWidget(self.save_experiment_button, 2, 3)
        controls.addWidget(self.start_log_button, 2, 4, 1, 2)
        controls.addWidget(self.stop_log_button, 2, 6)
        controls.addWidget(QLabel("Image nav"), 3, 0)
        controls.addWidget(self.image_zoom_in_button, 3, 1)
        controls.addWidget(self.image_zoom_out_button, 3, 2)
        controls.addWidget(self.image_reset_view_button, 3, 3)
        controls.addWidget(self.annotation_pin_button, 3, 4)
        controls.addWidget(self.annotation_polygon_button, 3, 5)
        controls.addWidget(self.annotation_confirm_button, 3, 6)
        controls.addWidget(self.annotation_cancel_button, 3, 7)
        controls.addWidget(self.annotation_clear_draft_button, 3, 8)
        controls.addWidget(QLabel("Operator target"), 4, 0)
        controls.addWidget(self.operator_target_combo, 4, 1, 1, 2)
        controls.addWidget(QLabel("Align X/Y/Rot/Scale"), 4, 3)
        controls.addWidget(self.align_x_spin, 4, 4)
        controls.addWidget(self.align_y_spin, 4, 5)
        controls.addWidget(self.align_rotation_spin, 4, 6)
        controls.addWidget(self.align_scale_spin, 5, 4)
        controls.addWidget(self.auto_alignment_button, 5, 5)
        controls.addWidget(self.auto_tilt_button, 5, 6)
        controls.addWidget(self.reset_alignment_button, 5, 7)
        controls.addWidget(QLabel("Search base"), 6, 0)
        controls.addWidget(self.template_base_combo, 6, 1)
        controls.addWidget(QLabel("Filter source"), 6, 2)
        controls.addWidget(self.template_filter_source_combo, 6, 3)
        controls.addWidget(self.template_load_button, 6, 4)
        controls.addWidget(self.template_crop_button, 6, 5)
        controls.addWidget(self.template_polygon_button, 6, 6)
        controls.addWidget(self.template_help_button, 6, 7)
        controls.addWidget(QLabel("Stride / angle sweep"), 7, 0)
        controls.addWidget(self.template_stride_spin, 7, 1)
        controls.addWidget(self.template_sweep_mode_combo, 7, 2)
        controls.addWidget(self.template_angle_min_spin, 7, 3)
        controls.addWidget(self.template_angle_max_spin, 7, 4)
        controls.addWidget(self.template_angle_step_spin, 7, 5)
        controls.addWidget(self.template_run_button, 7, 6)
        controls.addWidget(self.template_clear_button, 7, 7)
        controls.addWidget(QLabel("Operation progress"), 8, 0)
        controls.addWidget(self.operation_progress, 8, 1, 1, 3)
        controls.addWidget(self.template_sweep_label, 8, 4, 1, 4)
        controls.addWidget(QLabel("Video frame"), 9, 0)
        controls.addWidget(self.frame_slider, 9, 1, 1, 6)
        top_layout.addLayout(controls, 3)

        metrics_tab_layout = QVBoxLayout(self.metrics_tab)
        metrics_tab_layout.addWidget(self.metrics_help_label)
        metrics_tab_layout.addWidget(self.metrics_table)

        annotations_tab_layout = QVBoxLayout(self.annotations_tab)
        annotations_tab_layout.addWidget(self.annotations_table)

        notes_tab_layout = QVBoxLayout(self.notes_tab)
        notes_tab_layout.addWidget(self.annotation_context_label)
        notes_tab_layout.addWidget(self.annotation_label_edit)
        format_row = QHBoxLayout()
        format_row.addWidget(self.annotation_bold_button)
        format_row.addWidget(self.annotation_italic_button)
        format_row.addWidget(self.annotation_underline_button)
        format_row.addWidget(self.annotation_bullet_button)
        format_row.addStretch(1)
        notes_tab_layout.addLayout(format_row)
        notes_tab_layout.addWidget(self.annotation_notes_edit, 1)

        self.evidence_tabs.addTab(self.metrics_tab, "Metrics")
        self.evidence_tabs.addTab(self.annotations_tab, "Annotations")
        self.evidence_tabs.addTab(self.notes_tab, "Notes")
        self.evidence_tabs.setMinimumWidth(520)
        self.evidence_tabs.setMinimumHeight(360)
        evidence_panel = QVBoxLayout()
        evidence_panel.addWidget(QLabel("Evidence"))
        evidence_panel.addWidget(self.evidence_tabs)
        top_layout.addLayout(evidence_panel, 2)
        root_layout.addLayout(top_layout)

        info = QHBoxLayout()
        info.addWidget(self.status_label, 2)
        info.addWidget(self.metrics_label, 1)
        root_layout.addLayout(info)
        root_layout.addWidget(self.algorithm_label)
        root_layout.addWidget(self.log_label)

        standard_views = QHBoxLayout(self.standard_views_widget)
        standard_views.setContentsMargins(0, 0, 0, 0)
        standard_views.setSpacing(8)
        self.input_panel = self.make_image_panel(self.input_title, self.input_label, self.left_pane_combo)
        standard_views.addWidget(self.input_panel, 1)
        standard_views.addWidget(self.make_image_panel(self.processed_title, self.processed_label, self.center_pane_combo), 1)
        standard_views.addWidget(self.make_image_panel(self.original_title, self.original_label, self.right_pane_combo), 1)
        self.variance_panel = self.make_image_panel(self.variance_heatmap_title, self.variance_heatmap_label)
        standard_views.addWidget(self.variance_panel, 1)
        self.input_panel.setVisible(True)
        self.variance_panel.setVisible(False)

        advanced_grid = QGridLayout(self.advanced_views_widget)
        advanced_grid.addWidget(self.advanced_input_canvas, 0, 0)
        advanced_grid.addWidget(self.advanced_reference_canvas, 0, 1)
        self.views_stack.addWidget(self.standard_views_widget)
        self.views_stack.addWidget(self.advanced_views_widget)
        root_layout.addWidget(self.views_stack, 1)

    def _build_experiments_screen(self) -> None:
        layout = QVBoxLayout(self.experiments_screen)
        header = QHBoxLayout()
        title = QLabel("Experiments")
        title.setObjectName("ScreenTitle")
        title.setStyleSheet("QLabel { font-size: 18pt; color: #ffffff; font-weight: 700; }")
        self.add_neon_glow(title, "#2563eb", blur=24, alpha=170)
        header.addWidget(self.experiments_back_button)
        header.addWidget(title, 1)
        header.addWidget(self.refresh_experiments_button)
        layout.addLayout(header)

        content = QHBoxLayout()
        self.experiments_list.setMinimumWidth(360)
        content.addWidget(self.experiments_list, 1)
        detail_panel = QVBoxLayout()
        detail_panel.addWidget(self.experiment_summary_label)
        detail_panel.addWidget(self.experiment_details, 1)
        content.addLayout(detail_panel, 2)
        layout.addLayout(content, 1)

    def _connect(self) -> None:
        self.start_image_video_button.clicked.connect(self.show_image_video_workflow)
        self.start_camera_button.clicked.connect(self.show_camera_workflow)
        self.open_experiments_button.clicked.connect(self.show_experiments)
        self.image_video_back_button.clicked.connect(self.show_home_from_workflow)
        self.camera_back_button.clicked.connect(self.show_home_from_workflow)
        self.go_camera_workflow_button.clicked.connect(self.show_camera_workflow)
        self.go_image_video_workflow_button.clicked.connect(self.show_image_video_workflow)
        self.experiments_back_button.clicked.connect(self.show_home)
        self.refresh_experiments_button.clicked.connect(self.refresh_experiments)
        self.experiments_list.currentItemChanged.connect(self.show_selected_experiment)
        self.open_camera_button.clicked.connect(self.open_camera)
        self.open_image_button.clicked.connect(self.open_image)
        self.open_idrt_button.clicked.connect(self.open_idrt_pair)
        self.open_video_button.clicked.connect(self.open_video)
        self.refresh_ender_ports_button.clicked.connect(self.refresh_ender_ports)
        self.ender_connect_button.clicked.connect(self.connect_ender)
        self.ender_disconnect_button.clicked.connect(self.disconnect_ender)
        self.ender_home_button.clicked.connect(self.home_ender)
        self.ender_position_button.clicked.connect(self.request_ender_position)
        self.ender_soft_stop_button.clicked.connect(self.soft_stop_ender)
        self.ender_emergency_stop_button.clicked.connect(self.emergency_stop_ender)
        self.ender_x_minus_button.clicked.connect(lambda: self.jog_ender("X", -1.0))
        self.ender_x_plus_button.clicked.connect(lambda: self.jog_ender("X", 1.0))
        self.ender_y_minus_button.clicked.connect(lambda: self.jog_ender("Y", -1.0))
        self.ender_y_plus_button.clicked.connect(lambda: self.jog_ender("Y", 1.0))
        self.ender_z_minus_button.clicked.connect(lambda: self.jog_ender("Z", -1.0))
        self.ender_z_plus_button.clicked.connect(lambda: self.jog_ender("Z", 1.0))
        self.play_button.clicked.connect(self.toggle_video)
        self.original_only_button.clicked.connect(lambda: self.set_display_mode(False))
        self.analysis_button.clicked.connect(lambda: self.set_display_mode(True))
        self.advanced_view_button.clicked.connect(self.toggle_advanced_view)
        self.frame_slider.valueChanged.connect(self.show_video_frame)
        self.algorithm_combo.currentTextChanged.connect(lambda _text: self.schedule_processing_update())
        self.strength_slider.valueChanged.connect(lambda _value: self.schedule_processing_update())
        self.threshold_slider.valueChanged.connect(lambda _value: self.schedule_processing_update())
        self.process_every_spin.valueChanged.connect(lambda _value: self.schedule_processing_update())
        self.operator_target_combo.currentTextChanged.connect(self.update_operator_target)
        self.template_base_combo.currentTextChanged.connect(lambda _text: self.update_template_preview_pane())
        self.template_sweep_mode_combo.currentTextChanged.connect(lambda _text: self.update_template_sweep_controls())
        self.template_angle_min_spin.valueChanged.connect(lambda _value: self.update_template_sweep_label())
        self.template_angle_max_spin.valueChanged.connect(lambda _value: self.update_template_sweep_label())
        self.template_angle_step_spin.valueChanged.connect(lambda _value: self.update_template_sweep_label())
        self.left_pane_combo.currentTextChanged.connect(lambda _text: self.refresh_pane_views())
        self.center_pane_combo.currentTextChanged.connect(lambda _text: self.refresh_pane_views())
        self.right_pane_combo.currentTextChanged.connect(lambda _text: self.refresh_pane_views())
        self.align_x_spin.valueChanged.connect(lambda _value: self.schedule_satellite_alignment_update())
        self.align_y_spin.valueChanged.connect(lambda _value: self.schedule_satellite_alignment_update())
        self.align_rotation_spin.valueChanged.connect(lambda _value: self.schedule_satellite_alignment_update())
        self.align_scale_spin.valueChanged.connect(lambda _value: self.schedule_satellite_alignment_update())
        self.auto_alignment_button.clicked.connect(self.auto_align_satellite_pair)
        self.auto_tilt_button.clicked.connect(self.auto_tilt_satellite_pair)
        self.reset_alignment_button.clicked.connect(self.reset_satellite_alignment)
        self.template_load_button.clicked.connect(self.load_template_filter)
        self.template_crop_button.clicked.connect(self.arm_template_crop)
        self.template_polygon_button.clicked.connect(self.arm_template_polygon)
        self.template_run_button.clicked.connect(self.run_template_search)
        self.template_help_button.clicked.connect(self.show_template_help)
        self.template_clear_button.clicked.connect(self.clear_template_search)
        self.input_label.roi_selected.connect(lambda rect: self.finish_template_crop(self.template_crop_base_label, rect))
        self.processed_label.roi_selected.connect(lambda rect: self.finish_template_crop(self.template_crop_base_label, rect))
        self.original_label.roi_selected.connect(lambda rect: self.finish_template_crop(self.template_crop_base_label, rect))
        self.input_label.polygon_selected.connect(lambda points: self.finish_template_polygon(self.template_crop_base_label, points))
        self.processed_label.polygon_selected.connect(lambda points: self.finish_template_polygon(self.template_crop_base_label, points))
        self.original_label.polygon_selected.connect(lambda points: self.finish_template_polygon(self.template_crop_base_label, points))
        self.annotation_pin_button.clicked.connect(self.arm_annotation_pin)
        self.annotation_polygon_button.clicked.connect(self.arm_annotation_polygon)
        self.annotation_confirm_button.clicked.connect(self.confirm_annotation)
        self.annotation_cancel_button.clicked.connect(self.cancel_annotation)
        self.annotation_clear_draft_button.clicked.connect(self.clear_annotation_draft)
        self.annotation_bold_button.clicked.connect(self.toggle_annotation_bold)
        self.annotation_italic_button.clicked.connect(self.toggle_annotation_italic)
        self.annotation_underline_button.clicked.connect(self.toggle_annotation_underline)
        self.annotation_bullet_button.clicked.connect(self.insert_annotation_bullets)
        self.annotations_table.currentCellChanged.connect(self.show_annotation_from_table)
        self.input_label.annotation_pin_selected.connect(lambda point: self.finish_annotation_pin(self.source_for_view(self.input_label), point))
        self.processed_label.annotation_pin_selected.connect(lambda point: self.finish_annotation_pin(self.source_for_view(self.processed_label), point))
        self.original_label.annotation_pin_selected.connect(lambda point: self.finish_annotation_pin(self.source_for_view(self.original_label), point))
        self.variance_heatmap_label.annotation_pin_selected.connect(lambda point: self.finish_annotation_pin(self.source_for_view(self.variance_heatmap_label), point))
        self.input_label.annotation_polygon_selected.connect(lambda points: self.finish_annotation_polygon(self.source_for_view(self.input_label), points))
        self.processed_label.annotation_polygon_selected.connect(lambda points: self.finish_annotation_polygon(self.source_for_view(self.processed_label), points))
        self.original_label.annotation_polygon_selected.connect(lambda points: self.finish_annotation_polygon(self.source_for_view(self.original_label), points))
        self.variance_heatmap_label.annotation_polygon_selected.connect(lambda points: self.finish_annotation_polygon(self.source_for_view(self.variance_heatmap_label), points))
        self.save_row_button.clicked.connect(self.save_current_row)
        self.save_experiment_button.clicked.connect(self.save_current_experiment)
        self.start_log_button.clicked.connect(self.start_csv_log)
        self.stop_log_button.clicked.connect(self.stop_csv_log)
        self.hdf5_sample_combo.currentTextChanged.connect(self.on_hdf5_sample_changed)
        self.hdf5_input_combo.currentTextChanged.connect(self.on_hdf5_input_changed)
        self.hdf5_reference_combo.currentTextChanged.connect(self.load_current_hdf5_selection)
        self.hdf5_z_spin.valueChanged.connect(self.load_current_hdf5_selection)
        self.hdf5_z_down_button.clicked.connect(lambda: self.step_spinbox(self.hdf5_z_spin, -1))
        self.hdf5_z_up_button.clicked.connect(lambda: self.step_spinbox(self.hdf5_z_spin, 1))
        self.hdf5_prev_button.clicked.connect(lambda: self.step_hdf5_sample(-1))
        self.hdf5_next_button.clicked.connect(lambda: self.step_hdf5_sample(1))
        self.surface_downsample_spin.valueChanged.connect(self.refresh_advanced_view)
        self.surface_downsample_down_button.clicked.connect(lambda: self.step_spinbox(self.surface_downsample_spin, -1))
        self.surface_downsample_up_button.clicked.connect(lambda: self.step_spinbox(self.surface_downsample_spin, 1))
        self.surface_home_button.clicked.connect(self.reset_surface_views)
        self.image_zoom_in_button.clicked.connect(self.zoom_standard_images_in)
        self.image_zoom_out_button.clicked.connect(self.zoom_standard_images_out)
        self.image_reset_view_button.clicked.connect(self.reset_standard_image_views)
        self.qa_mode_combo.currentTextChanged.connect(lambda _text: self.render_variance_heatmap())
        self.residual_map_combo.currentTextChanged.connect(lambda _text: self.render_variance_heatmap())
        self.variance_norm_combo.currentTextChanged.connect(lambda _text: self.render_variance_heatmap())
        self.variance_map_combo.currentTextChanged.connect(lambda _text: self.render_variance_heatmap())
        self.variance_clip_spin.valueChanged.connect(lambda _value: self.render_variance_heatmap())
        self.variance_threshold_check.stateChanged.connect(lambda _state: self.render_variance_heatmap())
        self.export_variance_button.clicked.connect(self.export_variance_heatmap)
        self.refresh_ender_ports()

    def show_home(self) -> None:
        self.stack.setCurrentWidget(self.home_screen)

    def show_home_from_workflow(self) -> None:
        self.stop_sources()
        self.stack.setCurrentWidget(self.home_screen)

    def attach_analysis_workspace(self, slot: QVBoxLayout) -> None:
        for existing_slot in (self.image_video_workspace_slot, self.camera_workspace_slot):
            index = existing_slot.indexOf(self.analysis_workspace)
            if index >= 0:
                item = existing_slot.takeAt(index)
                if item is not None:
                    item.widget().setParent(None)
        parent = self.analysis_workspace.parentWidget()
        if parent is not None:
            self.analysis_workspace.setParent(None)
        slot.addWidget(self.analysis_workspace, 1)

    def show_image_video_workflow(self) -> None:
        if self.camera_worker is not None:
            self.stop_sources()
        self.attach_analysis_workspace(self.image_video_workspace_slot)
        self.stack.setCurrentWidget(self.image_video_screen)

    def show_camera_workflow(self) -> None:
        self.stop_sources()
        self.clear_hdf5_dataset()
        self.attach_analysis_workspace(self.camera_workspace_slot)
        self.stack.setCurrentWidget(self.camera_screen)

    def show_experiments(self) -> None:
        self.refresh_experiments()
        self.stack.setCurrentWidget(self.experiments_screen)

    def refresh_ender_ports(self) -> None:
        current = self.ender_port_combo.currentText().strip()
        self.ender_port_combo.clear()
        if list_ports is None:
            self.ender_port_combo.addItem("pyserial missing")
            self.ender_status_label.setText("Install pyserial to enable Ender controls.")
            self.ender_connect_button.setEnabled(False)
            return
        ports = list(list_ports.comports())
        for port_info in ports:
            label = f"{port_info.device} | {port_info.description}"
            self.ender_port_combo.addItem(label, port_info.device)
        if current and current != "pyserial missing":
            index = self.ender_port_combo.findText(current)
            if index >= 0:
                self.ender_port_combo.setCurrentIndex(index)
            else:
                self.ender_port_combo.setEditText(current.split("|")[0].strip())
        elif ports:
            self.ender_port_combo.setCurrentIndex(0)
        else:
            self.ender_port_combo.setEditText("COM3")
        self.ender_connect_button.setEnabled(True)

    def selected_ender_port(self) -> str:
        data = self.ender_port_combo.currentData()
        if data:
            return str(data)
        return self.ender_port_combo.currentText().split("|")[0].strip()

    def connect_ender(self) -> None:
        port = self.selected_ender_port()
        if not port or port == "pyserial missing":
            QMessageBox.warning(self, "Ender connection", "Choose or type a serial port such as COM3.")
            return
        self.disconnect_ender()
        self.set_ender_connected(False, connecting=True)
        self.ender_status_label.setText(f"Connecting to {port} ...")
        self.ender_worker = Ender3Worker(
            port,
            int(self.ender_baud_combo.currentText()),
            self.ender_max_x_spin.value(),
            self.ender_max_y_spin.value(),
            self.ender_max_z_spin.value(),
        )
        self.ender_worker.status.connect(self.ender_status_label.setText)
        self.ender_worker.connected_changed.connect(self.set_ender_connected)
        self.ender_worker.position_changed.connect(self.update_ender_position)
        self.ender_worker.log_line.connect(self.update_ender_log)
        self.ender_worker.start()

    def disconnect_ender(self) -> None:
        if self.ender_worker is not None:
            self.ender_worker.stop()
            self.ender_worker = None
        self.set_ender_connected(False)

    def set_ender_connected(self, connected: bool, connecting: bool = False) -> None:
        self.ender_connect_button.setEnabled(not connected and not connecting and serial is not None)
        self.ender_disconnect_button.setEnabled(connected or connecting)
        self.ender_port_combo.setEnabled(not connected and not connecting)
        self.ender_baud_combo.setEnabled(not connected and not connecting)
        for button in self.ender_motion_buttons:
            button.setEnabled(connected)
        if not connected and not connecting:
            self.ender_status_label.setText("Ender disconnected")

    def update_ender_position(self, x_pos: float, y_pos: float, z_pos: float) -> None:
        self.ender_position_label.setText(f"Position X {x_pos:0.2f} | Y {y_pos:0.2f} | Z {z_pos:0.2f}")

    def update_ender_log(self, line: str) -> None:
        self.ender_log_label.setText(f"Motion: {line[:110]}")

    def jog_ender(self, axis: str, direction: float) -> None:
        if self.ender_worker is None:
            return
        delta = direction * self.ender_step_spin.value()
        self.ender_worker.enqueue_jog(axis, delta, self.ender_feed_spin.value())

    def home_ender(self) -> None:
        if self.ender_worker is not None:
            self.ender_worker.enqueue_home()

    def request_ender_position(self) -> None:
        if self.ender_worker is not None:
            self.ender_worker.enqueue_position_request()

    def soft_stop_ender(self) -> None:
        if self.ender_worker is not None:
            self.ender_worker.enqueue_soft_stop()

    def emergency_stop_ender(self) -> None:
        if self.ender_worker is not None:
            self.ender_worker.enqueue_emergency_stop()

    def step_spinbox(self, spinbox: QSpinBox, delta: int) -> None:
        spinbox.setValue(min(spinbox.maximum(), max(spinbox.minimum(), spinbox.value() + delta)))

    def satellite_aligned_pre_frame(self) -> np.ndarray | None:
        if self.satellite_pre_frame is None:
            return None
        pre = ensure_rgb(self.satellite_pre_frame)
        target = self.satellite_post_frame if self.satellite_post_frame is not None else self.current_frame
        if target is None:
            return pre
        target = ensure_rgb(target)
        pre = resize_like(pre, target)
        height, width = target.shape[:2]
        pre = transform_frame_projective(pre, (height, width), self.satellite_homography)
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, self.align_rotation_spin.value(), self.align_scale_spin.value() / 100.0)
        matrix[0, 2] += self.align_x_spin.value()
        matrix[1, 2] += self.align_y_spin.value()
        return cv2.warpAffine(
            pre,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def schedule_processing_update(self) -> None:
        self.algorithm_label.setText(
            f"Algorithm: {self.algorithm_combo.currentText()} | "
            f"strength {self.strength_slider.value()} | threshold {self.threshold_slider.value()} | "
            f"process every {self.process_every_spin.value()} frame(s)"
        )
        self.processing_update_timer.start()

    def schedule_satellite_alignment_update(self) -> None:
        if self.satellite_pre_frame is None:
            return
        self.alignment_update_timer.start()

    def auto_align_satellite_pair(self) -> None:
        if self.satellite_pre_frame is None or self.current_frame is None:
            self.status_label.setText("Open an IDRT satellite pair before running Auto Align")
            return
        self.status_label.setText("Auto Align running: coarse affine search on downsampled GxH maps...")
        QApplication.processEvents()
        try:
            result = estimate_satellite_affine_alignment(self.current_frame, self.satellite_pre_frame)
        except Exception as exc:
            QMessageBox.warning(self, "Auto Align failed", str(exc))
            return
        self.align_x_spin.blockSignals(True)
        self.align_y_spin.blockSignals(True)
        self.align_rotation_spin.blockSignals(True)
        self.align_scale_spin.blockSignals(True)
        self.align_x_spin.setValue(int(round(result["x"])))
        self.align_y_spin.setValue(int(round(result["y"])))
        self.align_rotation_spin.setValue(float(result["rotation"]))
        self.align_scale_spin.setValue(float(result["scale"]))
        self.align_x_spin.blockSignals(False)
        self.align_y_spin.blockSignals(False)
        self.align_rotation_spin.blockSignals(False)
        self.align_scale_spin.blockSignals(False)
        self.update_satellite_alignment()
        self.status_label.setText(
            f"Auto Align: x {result['x']:0.1f}px | y {result['y']:0.1f}px | "
            f"rot {result['rotation']:0.2f} deg | scale {result['scale']:0.2f}% | score {result['score']:0.3f}"
        )

    def auto_tilt_satellite_pair(self) -> None:
        if self.satellite_pre_frame is None or self.current_frame is None:
            self.status_label.setText("Open an IDRT satellite pair before running Auto Tilt")
            return
        self.status_label.setText("Auto Tilt running: feature homography from downsampled GxH maps...")
        QApplication.processEvents()
        try:
            result = estimate_satellite_projective_alignment(self.current_frame, self.satellite_pre_frame)
        except Exception as exc:
            QMessageBox.warning(self, "Auto Tilt failed", str(exc))
            return
        self.satellite_homography = np.asarray(result["homography"], dtype=np.float64)
        self.satellite_homography_summary = (
            f"homography score {float(result['score']):0.3f}, "
            f"inliers {int(result['inliers'])}/{int(result['matches'])}"
        )
        self.satellite_homography_score = float(result["score"])
        self.satellite_homography_inliers = int(result["inliers"])
        self.update_satellite_alignment()
        self.status_label.setText(
            f"Auto Tilt: {self.satellite_homography_summary}. "
            "Manual X/Y/rotation/scale can still be used for cleanup."
        )

    def satellite_operator_frame(self, post_frame: np.ndarray) -> np.ndarray:
        if self.satellite_pre_frame is None:
            return post_frame
        if self.operator_target_combo.currentText() == "Pre/reference image":
            aligned = self.satellite_aligned_pre_frame()
            if aligned is not None:
                return aligned
        return post_frame

    def refresh_satellite_shared_gxh(self) -> None:
        if self.current_frame is None or self.satellite_pre_frame is None:
            self.satellite_shared_gxh_frame = None
            self.satellite_shared_gxh_mask = None
            return
        shared, support = satellite_shared_gxh_reference(
            self.current_frame,
            self.satellite_aligned_pre_frame(),
            self.threshold_slider.value(),
        )
        self.satellite_shared_gxh_frame = shared
        self.satellite_shared_gxh_mask = support

    def update_operator_target(self) -> None:
        if self.current_frame is not None:
            self.last_processed = None
            self.render_processed(self.current_frame)
            self.refresh_current_metrics()
        self.update_algorithm_label()

    def update_satellite_alignment(self) -> None:
        if self.satellite_pre_frame is None or self.current_frame is None:
            return
        aligned = self.satellite_aligned_pre_frame()
        if aligned is not None:
            self.hdf5_reference_frame = aligned
        self.comparison_frame = aligned
        self.refresh_satellite_shared_gxh()
        self.last_processed = None
        self.set_progress(8, "Processing")
        self.render_processed(self.current_frame)
        self.set_progress(100, "Processing complete")
        self.refresh_current_metrics()
        self.status_label.setText(
            f"IDRT satellite pair: {self.satellite_pair_name} | "
            f"align x {self.align_x_spin.value()} y {self.align_y_spin.value()} "
            f"rot {self.align_rotation_spin.value():0.2f} scale {self.align_scale_spin.value():0.2f}%"
        )
        QTimer.singleShot(900, self.reset_progress)

    def reset_satellite_alignment(self) -> None:
        self.align_x_spin.blockSignals(True)
        self.align_y_spin.blockSignals(True)
        self.align_rotation_spin.blockSignals(True)
        self.align_scale_spin.blockSignals(True)
        self.align_x_spin.setValue(0)
        self.align_y_spin.setValue(0)
        self.align_rotation_spin.setValue(0.0)
        self.align_scale_spin.setValue(100.0)
        self.align_x_spin.blockSignals(False)
        self.align_y_spin.blockSignals(False)
        self.align_rotation_spin.blockSignals(False)
        self.align_scale_spin.blockSignals(False)
        self.satellite_homography = None
        self.satellite_homography_summary = ""
        self.satellite_homography_score = ""
        self.satellite_homography_inliers = ""
        self.update_satellite_alignment()

    def template_base_frame(self, base_label: str | None = None) -> tuple[np.ndarray | None, ZoomableImageView | None, str]:
        label = base_label or self.template_base_combo.currentText()
        frame, title = self.frame_for_source(label)
        return frame, self.view_for_source(label), title

    def set_combo_items_preserving(self, combo: QComboBox, items: tuple[str, ...], fallback: str) -> None:
        current = combo.currentText()
        if current == "Raw/Input" and "Raw/Input" not in items and "Post/Event" in items:
            current = "Post/Event"
        elif current == "Post/Event" and "Post/Event" not in items and "Raw/Input" in items:
            current = "Raw/Input"
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.setCurrentText(current if current in items else fallback)
        combo.blockSignals(False)

    def update_source_options_for_mode(self) -> None:
        satellite_mode = self.satellite_pre_frame is not None
        pane_sources = SATELLITE_SOURCES if satellite_mode else GENERIC_SOURCES
        search_sources = tuple(source for source in pane_sources if source != "CIS Filter")
        crop_sources = tuple(source for source in search_sources if source != "CIS Heatmap")
        primary = "Post/Event" if satellite_mode else "Raw/Input"
        self.set_combo_items_preserving(self.left_pane_combo, pane_sources, primary)
        self.set_combo_items_preserving(self.center_pane_combo, pane_sources, "Processed Output")
        self.set_combo_items_preserving(self.right_pane_combo, pane_sources, "Reference/Pre")
        self.set_combo_items_preserving(self.template_base_combo, search_sources, primary)
        self.set_combo_items_preserving(self.template_filter_source_combo, crop_sources, primary)

    def view_for_source(self, source: str) -> ZoomableImageView:
        for combo, view in (
            (self.left_pane_combo, self.input_label),
            (self.center_pane_combo, self.processed_label),
            (self.right_pane_combo, self.original_label),
        ):
            if combo.currentText() == source:
                return view
        if source == "Processed Output" or source == "CIS Heatmap":
            return self.processed_label
        if source == "Raw/Input" or source == "Post/Event":
            return self.input_label
        return self.original_label

    def reference_frame_for_display(self) -> np.ndarray | None:
        if self.satellite_pre_frame is not None:
            reference = self.satellite_aligned_pre_frame()
            if reference is not None:
                return reference
        if self.hdf5_reference_frame is not None:
            return self.hdf5_reference_frame
        return self.comparison_frame

    def frame_for_source(self, source: str) -> tuple[np.ndarray | None, str]:
        if source == "Post/Event":
            return self.current_frame, "Post/Event"
        if source == "Raw/Input":
            title = "Post/Event" if self.satellite_pre_frame is not None else "Raw/Input"
            return self.current_frame, title
        if source == "Processed Output":
            return self.last_processed, "Processed Output"
        if source == "Reference/Pre":
            return self.reference_frame_for_display(), "Reference/Pre"
        if source == "CIS Heatmap":
            heatmap = self.template_match_result.get("heatmap") if self.template_match_result else None
            return heatmap if isinstance(heatmap, np.ndarray) else None, "CIS Heatmap"
        if source == "CIS Filter":
            return self.template_match_template, "CIS Filter"
        if source == "Shared GxH":
            return self.satellite_shared_gxh_frame, "Shared GxH"
        if source == "Current GxH":
            return biobridge_gradient_hessian_operator(self.current_frame) if self.current_frame is not None else None, "Current GxH"
        return None, source

    def source_for_view(self, view: ZoomableImageView) -> str:
        if view is self.input_label:
            return self.frame_for_source(self.left_pane_combo.currentText())[1]
        if view is self.processed_label:
            return self.frame_for_source(self.center_pane_combo.currentText())[1]
        if view is self.original_label:
            return self.frame_for_source(self.right_pane_combo.currentText())[1]
        if view is self.variance_heatmap_label:
            return self.variance_heatmap_title.text()
        return "Unknown"

    def current_annotation_context(self) -> str:
        return self.current_source or "idle"

    def annotation_matches_current_context(self, annotation: dict[str, object]) -> bool:
        context = self.current_annotation_context()
        annotation_context = annotation.get("image_context")
        if annotation_context is not None:
            return annotation_context == context
        return annotation.get("source") == context

    def annotations_for_source(self, source: str) -> list[dict[str, object]]:
        overlays = [
            annotation
            for annotation in self.annotations
            if self.annotation_matches_current_context(annotation) and annotation.get("pane_source") == source
        ]
        if self.pending_annotation is not None and self.pending_annotation.get("pane_source") == source:
            overlays.append(self.pending_annotation)
        return overlays

    def refresh_annotation_overlays(self) -> None:
        for combo, view in (
            (self.left_pane_combo, self.input_label),
            (self.center_pane_combo, self.processed_label),
            (self.right_pane_combo, self.original_label),
        ):
            view.set_annotation_overlays(self.annotations_for_source(self.frame_for_source(combo.currentText())[1]))
        self.variance_heatmap_label.set_annotation_overlays(self.annotations_for_source(self.variance_heatmap_title.text()))

    def stop_annotation_capture_modes(self) -> None:
        for view in self.standard_image_views():
            view.set_annotation_pin_mode(False)
            view.set_annotation_polygon_mode(False)

    def arm_annotation_pin(self) -> None:
        if self.current_frame is None and self.last_processed is None and self.reference_frame_for_display() is None:
            self.status_label.setText("Open an image-like frame before adding an annotation.")
            return
        self.stop_annotation_capture_modes()
        for view in self.standard_image_views():
            view.set_annotation_pin_mode(True)
        self.status_label.setText("Pin Note armed: click a visible image pane to place an evidence pin.")

    def arm_annotation_polygon(self) -> None:
        if self.current_frame is None and self.last_processed is None and self.reference_frame_for_display() is None:
            self.status_label.setText("Open an image-like frame before adding an annotation.")
            return
        self.stop_annotation_capture_modes()
        for view in self.standard_image_views():
            view.set_annotation_polygon_mode(True)
        self.status_label.setText("Polygon Note armed: left-click vertices on a visible pane, then right-click to close.")

    def begin_annotation_draft(self, annotation_type: str, pane_source: str, points: list[tuple[int, int]]) -> None:
        if not points:
            self.status_label.setText("Annotation needs at least one point.")
            return
        center_x = int(round(sum(point[0] for point in points) / len(points)))
        center_y = int(round(sum(point[1] for point in points) / len(points)))
        self.pending_annotation = {
            "annotation_id": "draft",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.current_source,
            "image_context": self.current_annotation_context(),
            "pane_source": pane_source,
            "artifact_label": pane_source,
            "annotation_type": annotation_type,
            "points": [[int(x), int(y)] for x, y in points],
            "center_x": center_x,
            "center_y": center_y,
            "label": "",
            "notes_html": "",
            "notes_plaintext": "",
        }
        self.annotation_label_edit.setText(f"{annotation_type} @ {center_x}, {center_y}")
        self.annotation_notes_edit.clear()
        self.annotation_context_label.setText(
            f"Draft {annotation_type} on {pane_source} | center ({center_x}, {center_y}) | {len(points)} point(s)"
        )
        self.evidence_tabs.setCurrentWidget(self.notes_tab)
        self.refresh_annotation_overlays()
        self.status_label.setText("Draft annotation ready. Write notes, then Confirm Note.")

    def finish_annotation_pin(self, pane_source: str, point: object) -> None:
        try:
            x, y = point  # type: ignore[misc]
        except (TypeError, ValueError):
            self.status_label.setText("Could not place annotation pin.")
            return
        self.stop_annotation_capture_modes()
        self.begin_annotation_draft("pin", pane_source, [(int(x), int(y))])

    def finish_annotation_polygon(self, pane_source: str, points: object) -> None:
        cleaned: list[tuple[int, int]] = []
        if isinstance(points, list):
            for point in points:
                try:
                    x, y = point  # type: ignore[misc]
                except (TypeError, ValueError):
                    continue
                cleaned.append((int(x), int(y)))
        self.stop_annotation_capture_modes()
        if len(cleaned) < 3:
            self.status_label.setText("Polygon annotation needs at least three points.")
            return
        self.begin_annotation_draft("polygon", pane_source, cleaned)

    def annotation_settings_snapshot(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm_combo.currentText() if self.show_processed else "Original Only",
            "strength": self.strength_slider.value(),
            "threshold": self.threshold_slider.value(),
            "process_every_n": self.process_every_spin.value(),
            "operator_target": self.operator_target_combo.currentText() if self.satellite_pre_frame is not None else "Current image",
            "alignment_x_px": self.align_x_spin.value() if self.satellite_pre_frame is not None else "",
            "alignment_y_px": self.align_y_spin.value() if self.satellite_pre_frame is not None else "",
            "alignment_rotation_deg": self.align_rotation_spin.value() if self.satellite_pre_frame is not None else "",
            "alignment_scale_percent": self.align_scale_spin.value() if self.satellite_pre_frame is not None else "",
            "template_search_base": self.template_base_combo.currentText(),
            "template_filter_source": self.template_filter_source_combo.currentText(),
            "template_sweep_mode": self.template_sweep_mode_combo.currentText(),
            "template_sweep_angles": [float(angle) for angle in self.template_sweep_angles()],
            "template_stride_px": self.template_stride_spin.value(),
        }

    def confirm_annotation(self) -> None:
        if self.pending_annotation is None:
            self.status_label.setText("No draft annotation to confirm.")
            return
        label = self.annotation_label_edit.text().strip() or "Untitled annotation"
        annotation = dict(self.pending_annotation)
        annotation["annotation_id"] = "ann_" + time.strftime("%Y%m%d_%H%M%S") + f"_{len(self.annotations) + 1:03d}"
        annotation["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        annotation["label"] = label
        annotation["notes_html"] = self.annotation_notes_edit.toHtml()
        annotation["notes_plaintext"] = self.annotation_notes_edit.toPlainText()
        annotation["settings"] = self.annotation_settings_snapshot()
        annotation["metrics"] = self.csv_ready_row(self.latest_metrics) if self.latest_metrics else {}
        annotation["template_match"] = template_match_metrics(self.template_match_result)
        safe_annotation = json_safe(annotation)
        if not isinstance(safe_annotation, dict):
            self.status_label.setText("Could not serialize annotation.")
            return
        try:
            append_jsonl(ANNOTATIONS_PATH, safe_annotation)
        except OSError as exc:
            QMessageBox.warning(self, "Annotation save failed", str(exc))
            return
        self.annotations.append(safe_annotation)
        self.pending_annotation = None
        self.refresh_annotations_table()
        self.refresh_annotation_overlays()
        self.evidence_tabs.setCurrentWidget(self.annotations_tab)
        self.status_label.setText(f"Saved annotation: {label}")

    def cancel_annotation(self) -> None:
        self.stop_annotation_capture_modes()
        self.pending_annotation = None
        self.refresh_annotation_overlays()
        self.status_label.setText("Annotation capture canceled.")

    def clear_annotation_draft(self) -> None:
        self.pending_annotation = None
        self.annotation_label_edit.clear()
        self.annotation_notes_edit.clear()
        self.annotation_context_label.setText("No active annotation.")
        self.refresh_annotation_overlays()
        self.status_label.setText("Cleared draft annotation.")

    def load_annotations(self) -> None:
        self.annotations = read_jsonl(ANNOTATIONS_PATH)
        self.refresh_annotations_table()
        self.refresh_annotation_overlays()

    def refresh_annotations_table(self) -> None:
        self.visible_annotations = [
            annotation for annotation in self.annotations if self.annotation_matches_current_context(annotation)
        ]
        self.annotations_table.blockSignals(True)
        self.annotations_table.setRowCount(len(self.visible_annotations))
        for row, annotation in enumerate(self.visible_annotations):
            values = (
                annotation.get("annotation_id", ""),
                annotation.get("annotation_type", ""),
                annotation.get("pane_source", ""),
                annotation.get("center_x", ""),
                annotation.get("center_y", ""),
                annotation.get("label", ""),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, annotation.get("annotation_id", ""))
                self.annotations_table.setItem(row, column, item)
        self.annotations_table.blockSignals(False)

    def show_annotation_from_table(self, current_row: int, current_column: int, previous_row: int, previous_column: int) -> None:
        if current_row < 0 or current_row >= len(self.visible_annotations):
            return
        annotation = self.visible_annotations[current_row]
        self.pending_annotation = None
        self.annotation_label_edit.setText(str(annotation.get("label", "")))
        self.annotation_notes_edit.setHtml(str(annotation.get("notes_html", "")))
        self.annotation_context_label.setText(
            f"Saved {annotation.get('annotation_type', '')} on {annotation.get('pane_source', '')} | "
            f"center ({annotation.get('center_x', '')}, {annotation.get('center_y', '')}) | "
            f"{annotation.get('created_at', '')}"
        )
        self.evidence_tabs.setCurrentWidget(self.notes_tab)

    def toggle_annotation_bold(self) -> None:
        weight = self.annotation_notes_edit.fontWeight()
        self.annotation_notes_edit.setFontWeight(QFont.Normal if weight > QFont.Normal else QFont.Bold)

    def toggle_annotation_italic(self) -> None:
        self.annotation_notes_edit.setFontItalic(not self.annotation_notes_edit.fontItalic())

    def toggle_annotation_underline(self) -> None:
        self.annotation_notes_edit.setFontUnderline(not self.annotation_notes_edit.fontUnderline())

    def insert_annotation_bullets(self) -> None:
        cursor = self.annotation_notes_edit.textCursor()
        cursor.insertList(QTextListFormat.ListDisc)
        self.annotation_notes_edit.setTextCursor(cursor)

    def set_progress(self, value: int, label: str) -> None:
        value = max(0, min(100, int(value)))
        self.operation_progress.setValue(value)
        self.operation_progress.setFormat(f"{label} | {value}%")
        QApplication.processEvents()

    def reset_progress(self, label: str = "Idle") -> None:
        self.operation_progress.setValue(0)
        self.operation_progress.setFormat(label)

    def render_pane_source(self, combo: QComboBox, title_label: QLabel, image_label: ZoomableImageView) -> None:
        frame, title = self.frame_for_source(combo.currentText())
        title_label.setText(title)
        if frame is None:
            image_label.setText(f"No {title} available")
            image_label.set_annotation_overlays([])
            return
        if frame.ndim == 3:
            image_label.set_image_rgb(frame)
        else:
            image_label.set_image_gray(frame)
        image_label.set_annotation_overlays(self.annotations_for_source(title))

    def refresh_pane_views(self) -> None:
        if self._pane_refreshing:
            return
        self._pane_refreshing = True
        try:
            self.render_pane_source(self.left_pane_combo, self.input_title, self.input_label)
            self.render_pane_source(self.center_pane_combo, self.processed_title, self.processed_label)
            self.render_pane_source(self.right_pane_combo, self.original_title, self.original_label)
        finally:
            self._pane_refreshing = False

    def template_sweep_angles(self) -> np.ndarray:
        mode = self.template_sweep_mode_combo.currentText()
        if mode == "Light 90 deg":
            return np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
        if mode == "Deep 45 deg":
            return np.arange(0.0, 360.0, 45.0, dtype=np.float32)
        angle_min = self.template_angle_min_spin.value()
        angle_max = self.template_angle_max_spin.value()
        angle_step = max(0.25, abs(self.template_angle_step_spin.value()))
        if angle_min > angle_max:
            angle_min, angle_max = angle_max, angle_min
        angles = np.arange(float(angle_min), float(angle_max) + angle_step * 0.5, angle_step)
        return angles if angles.size else np.array([0.0], dtype=np.float32)

    def update_template_sweep_controls(self) -> None:
        custom = self.template_sweep_mode_combo.currentText() == "Custom fine"
        for spinbox in (self.template_angle_min_spin, self.template_angle_max_spin, self.template_angle_step_spin):
            spinbox.setEnabled(custom)
        self.update_template_sweep_label()

    def update_template_sweep_label(self) -> None:
        angles = self.template_sweep_angles()
        preview = ", ".join(f"{float(angle):0.1f}" for angle in angles[:12])
        if angles.size > 12:
            preview += ", ..."
        self.template_sweep_label.setText(f"{self.template_sweep_mode_combo.currentText()} | Angles [{angles.size}]: {preview} deg")

    def show_template_help(self) -> None:
        angles = self.template_sweep_angles()
        angle_text = ", ".join(f"{float(angle):0.1f}" for angle in angles)
        QMessageBox.information(
            self,
            "Convolutional Image Search",
            "Convolutional Image Search is GAIA's template-matching tool.\n\n"
            "1. Search base: the image GAIA searches inside.\n"
            "2. Filter source: the image used to crop a rectangular or polygon filter.\n"
            "3. Load/Crop/Polygon Filter: creates the kernel/template.\n"
            "4. Run Image Search: sweeps the template over the base image using grayscale normalized cross-correlation.\n\n"
            f"Current stride: {self.template_stride_spin.value()} px\n"
            f"Current rotation mode: {self.template_sweep_mode_combo.currentText()}\n"
            f"Current rotation sweep: {angle_text} deg\n\n"
            "Polygon filters are closed automatically on right-click. Pixels outside the polygon are masked out and ignored.\n\n"
            "Best match is selected by maximum correlation. Mean absolute difference is reported as an interpretability check.\n"
            "The metrics table is the evidence/export table: it records search base, filter source, best x/y, correlation score, "
            "difference score, best rotation, template size, stride, alignment settings, and processing parameters."
        )

    def load_template_filter(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Load search filter",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path_text:
            return
        try:
            template = load_image(Path(path_text))
        except Exception as exc:
            QMessageBox.warning(self, "Filter load failed", str(exc))
            return
        if min(template.shape[:2]) < 3:
            QMessageBox.warning(self, "Filter too small", "Use at least a 3 x 3 pixel filter image.")
            return
        self.template_match_template = template
        self.template_match_mask = None
        self.template_match_source = Path(path_text).name
        self.template_match_result = {}
        self.update_template_preview_pane()
        self.status_label.setText(f"Loaded search filter: {self.template_match_source} ({template.shape[1]} x {template.shape[0]} px)")

    def arm_template_crop(self) -> None:
        frame, label = self.frame_for_source(self.template_filter_source_combo.currentText())
        view = self.view_for_source(self.template_filter_source_combo.currentText())
        if frame is None or view is None:
            self.status_label.setText("Open an image-like frame before cropping a search filter.")
            return
        self.template_crop_base_label = label
        for image_view in (self.input_label, self.processed_label, self.original_label):
            image_view.set_roi_mode(False)
        if frame.ndim == 3:
            view.set_image_rgb(frame)
        else:
            view.set_image_gray(frame)
        if view is self.original_label:
            self.original_title.setText(f"Crop Filter From {label}")
        elif view is self.processed_label:
            self.processed_title.setText(f"Crop Filter From {label}")
        elif view is self.input_label:
            self.input_title.setText(f"Crop Filter From {label}")
        view.set_roi_mode(True)
        self.status_label.setText(f"Draw a crop box on {label} to create the search filter.")

    def arm_template_polygon(self) -> None:
        frame, label = self.frame_for_source(self.template_filter_source_combo.currentText())
        view = self.view_for_source(self.template_filter_source_combo.currentText())
        if frame is None or view is None:
            self.status_label.setText("Open an image-like frame before drawing a polygon search filter.")
            return
        self.template_crop_base_label = label
        for image_view in (self.input_label, self.processed_label, self.original_label):
            image_view.set_roi_mode(False)
            image_view.set_polygon_mode(False)
        if frame.ndim == 3:
            view.set_image_rgb(frame)
        else:
            view.set_image_gray(frame)
        if view is self.original_label:
            self.original_title.setText(f"Polygon Filter From {label}")
        elif view is self.processed_label:
            self.processed_title.setText(f"Polygon Filter From {label}")
        elif view is self.input_label:
            self.input_title.setText(f"Polygon Filter From {label}")
        view.set_polygon_mode(True)
        self.status_label.setText(f"Left-click polygon points on {label}; right-click to close the filter.")

    def finish_template_crop(self, base_label: str, rect: object) -> None:
        frame, label = self.frame_for_source(base_label)
        if frame is None or rect is None:
            self.status_label.setText("No image available for the selected search-filter crop.")
            return
        x, y, width, height = rect  # type: ignore[misc]
        if width < 3 or height < 3:
            self.status_label.setText("Search filter crop is too small.")
            return
        template = ensure_rgb(frame)[y:y + height, x:x + width].copy()
        self.template_match_template = template
        self.template_match_mask = None
        self.template_match_source = f"{label} crop x{x} y{y} w{width} h{height}"
        self.template_match_result = {}
        self.update_template_preview_pane()
        self.status_label.setText(f"Cropped search filter from {label}: {width} x {height} px")

    def finish_template_polygon(self, base_label: str, points: object) -> None:
        frame, label = self.frame_for_source(base_label)
        if frame is None or points is None:
            self.status_label.setText("No image available for the selected polygon search filter.")
            return
        polygon = np.asarray(points, dtype=np.int32)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            self.status_label.setText("Polygon filter needs at least three points.")
            return
        x, y, width, height = cv2.boundingRect(polygon)
        if width < 3 or height < 3:
            self.status_label.setText("Polygon search filter is too small.")
            return
        source = ensure_rgb(frame)
        crop = source[y:y + height, x:x + width].copy()
        local_polygon = polygon - np.array([x, y], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [local_polygon], 255)
        template = np.zeros_like(crop)
        template[mask > 0] = crop[mask > 0]
        self.template_match_template = template
        self.template_match_mask = mask
        self.template_match_source = f"{label} polygon x{x} y{y} w{width} h{height} points{polygon.shape[0]}"
        self.template_match_result = {}
        self.update_template_preview_pane()
        self.status_label.setText(f"Created polygon search filter from {label}: {width} x {height} px, {polygon.shape[0]} points")

    def clear_template_search(self) -> None:
        self.template_match_template = None
        self.template_match_mask = None
        self.template_match_source = ""
        self.template_match_result = {}
        for image_view in (self.input_label, self.processed_label, self.original_label):
            image_view.set_roi_mode(False)
            image_view.set_polygon_overlay(None)
        self.reset_progress()
        self.refresh_pane_views()
        if self.current_frame is not None and self.last_processed is not None:
            self.render_processed_image(self.last_processed)
        self.refresh_current_metrics()
        self.status_label.setText("Cleared Convolutional Image Search state.")

    def run_template_search(self) -> None:
        base, _view, base_label = self.template_base_frame()
        if base is None:
            self.status_label.setText("Open an image-like frame before running Convolutional Image Search.")
            return
        if self.template_match_template is None:
            self.status_label.setText("Load or crop a search filter before running Convolutional Image Search.")
            return
        try:
            self.set_progress(0, "CIS starting")
            result = template_match_search(
                base,
                self.template_match_template,
                self.template_stride_spin.value(),
                self.template_angle_min_spin.value(),
                self.template_angle_max_spin.value(),
                self.template_angle_step_spin.value(),
                self.set_progress,
                self.template_sweep_angles(),
                self.template_match_mask,
            )
        except Exception as exc:
            self.reset_progress("CIS failed")
            self.status_label.setText(f"Image search failed: {exc}")
            return
        self.template_match_result = result
        self.last_processed = result["heatmap"]  # type: ignore[assignment]
        self.update_template_preview_pane()
        self.refresh_current_metrics()
        self.status_label.setText(
            f"Search {base_label}: best {result['best_vector']} | "
            f"diff {float(result['difference_score']):0.2f} | rot {float(result['rotation_deg']):0.1f} deg"
        )

    def update_template_preview_pane(self, force_reference: bool = False) -> None:
        if not force_reference:
            self.refresh_pane_views()
            return
        if force_reference or self.template_base_combo.currentText() == "Reference/Pre" or self.template_match_template is None:
            if self.satellite_pre_frame is not None:
                reference = self.satellite_aligned_pre_frame()
                if reference is not None:
                    self.original_label.set_image_rgb(reference)
                    self.original_title.setText("Pre/reference image")
                    return
            if self.hdf5_reference_frame is not None:
                self.original_label.set_image_rgb(self.hdf5_reference_frame)
                self.original_title.setText("Original / Reference")
                return
            if self.current_frame is not None and self.satellite_pre_frame is None:
                self.original_label.set_image_rgb(self.current_frame)
                self.original_title.setText("Original")
            return
        self.original_label.set_image_rgb(self.template_match_template)
        self.original_title.setText(f"Search Filter | {self.template_match_template.shape[1]} x {self.template_match_template.shape[0]} px")

    def set_paired_hdf5_view_enabled(self, enabled: bool) -> None:
        if self.input_panel is not None:
            self.input_panel.setVisible(enabled)
        if enabled:
            return
        self.refresh_pane_views()

    def add_neon_glow(self, widget: QWidget, color: str = "#2563eb", blur: int = 18, alpha: int = 160) -> None:
        effect = QGraphicsDropShadowEffect(widget)
        glow = QColor(color)
        glow.setAlpha(alpha)
        effect.setColor(glow)
        effect.setBlurRadius(blur)
        effect.setOffset(0, 0)
        widget.setGraphicsEffect(effect)

    def apply_instrument_glow(self) -> None:
        for button in (
            self.start_image_video_button,
            self.start_camera_button,
            self.open_experiments_button,
        ):
            self.add_neon_glow(button, "#2563eb", blur=22, alpha=150)

        for button in (
            self.image_video_back_button,
            self.camera_back_button,
            self.go_camera_workflow_button,
            self.go_image_video_workflow_button,
            self.experiments_back_button,
            self.refresh_experiments_button,
            self.open_camera_button,
            self.open_image_button,
            self.open_video_button,
            self.refresh_ender_ports_button,
            self.ender_connect_button,
            self.ender_disconnect_button,
            self.ender_home_button,
            self.ender_position_button,
            self.ender_soft_stop_button,
            self.ender_emergency_stop_button,
            self.ender_x_minus_button,
            self.ender_x_plus_button,
            self.ender_y_minus_button,
            self.ender_y_plus_button,
            self.ender_z_minus_button,
            self.ender_z_plus_button,
            self.play_button,
            self.original_only_button,
            self.analysis_button,
            self.advanced_view_button,
            self.save_experiment_button,
            self.save_row_button,
            self.start_log_button,
            self.stop_log_button,
            self.hdf5_z_down_button,
            self.hdf5_z_up_button,
            self.hdf5_prev_button,
            self.hdf5_next_button,
            self.surface_downsample_down_button,
            self.surface_downsample_up_button,
            self.surface_home_button,
            self.image_zoom_in_button,
            self.image_zoom_out_button,
            self.image_reset_view_button,
            self.export_variance_button,
        ):
            self.add_neon_glow(button, "#2563eb", blur=10, alpha=95)

        for label in (
            self.input_title,
            self.processed_title,
            self.original_title,
            self.variance_heatmap_title,
        ):
            self.add_neon_glow(label, "#22c55e", blur=10, alpha=110)

    def make_stepper(self, down_button: QPushButton, spinbox: QSpinBox, up_button: QPushButton) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(down_button)
        layout.addWidget(spinbox, 1)
        layout.addWidget(up_button)
        return widget

    def make_image_panel(self, title_label: QLabel, image_label: QLabel, source_combo: QComboBox | None = None) -> QWidget:
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(24)
        title_label.setStyleSheet("QLabel { color: #86efac; font-size: 10pt; font-weight: 600; }")
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if source_combo is not None:
            layout.addWidget(source_combo)
        layout.addWidget(title_label)
        layout.addWidget(image_label, 1)
        return panel

    def create_surface_canvas(self, title: str) -> QWidget:
        if scene is not None and get_colormap is not None:
            try:
                return VispySurfaceWidget(title)
            except Exception:
                pass
        if FigureCanvas is None or Figure is None:
            label = QLabel(f"{title}\nMatplotlib is not available.")
            label.setAlignment(Qt.AlignCenter)
            return label
        figure = Figure(figsize=(5, 4), facecolor="#05070a")
        canvas = FigureCanvas(figure)
        canvas.setStyleSheet("background: #05070a; border: 1px solid #1f2937;")
        axis = figure.add_subplot(111, projection="3d")
        axis.set_title(title, color="#86efac")
        axis.set_facecolor("#05070a")
        canvas.figure = figure  # type: ignore[attr-defined]
        canvas.surface_axis = axis  # type: ignore[attr-defined]
        return canvas

    def clear_surface_canvas(self, canvas: QWidget, title: str) -> None:
        if FigureCanvas is None or not hasattr(canvas, "figure"):
            if isinstance(canvas, QLabel):
                canvas.setText(f"{title}\nNo surface available")
            return
        figure = canvas.figure  # type: ignore[attr-defined]
        figure.clear()
        axis = figure.add_subplot(111, projection="3d")
        axis.set_title(title, color="#86efac")
        axis.set_facecolor("#05070a")
        canvas.surface_axis = axis  # type: ignore[attr-defined]
        canvas.draw_idle()  # type: ignore[attr-defined]

    def plot_surface_canvas(self, canvas: QWidget, frame_rgb: np.ndarray | None, title: str) -> None:
        if isinstance(canvas, VispySurfaceWidget):
            try:
                canvas.set_surface(frame_rgb, self.surface_downsample_spin.value(), title)
            except Exception as exc:
                self.status_label.setText(f"GPU surface render failed: {exc}")
            return
        if frame_rgb is None:
            self.clear_surface_canvas(canvas, title)
            return
        if FigureCanvas is None or not hasattr(canvas, "figure"):
            if isinstance(canvas, QLabel):
                canvas.setText(f"{title}\nMatplotlib is not available.")
            return

        gray = rgb_to_luminance(ensure_rgb(frame_rgb))
        step = max(1, self.surface_downsample_spin.value())
        surface = gray[::step, ::step].astype(np.float32)
        y_values, x_values = np.mgrid[0 : gray.shape[0] : step, 0 : gray.shape[1] : step]

        figure = canvas.figure  # type: ignore[attr-defined]
        figure.clear()
        axis = figure.add_subplot(111, projection="3d")
        axis.plot_surface(
            x_values,
            y_values,
            surface,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
            rstride=1,
            cstride=1,
        )
        axis.set_title(f"{title} | step {step}", color="#86efac")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("v")
        axis.set_zlim(0, 255)
        axis.view_init(elev=28, azim=-135)
        axis.set_facecolor("#05070a")
        axis.tick_params(colors="#e5e7eb")
        axis.xaxis.label.set_color("#e5e7eb")
        axis.yaxis.label.set_color("#e5e7eb")
        axis.zaxis.label.set_color("#e5e7eb")
        figure.tight_layout()
        canvas.draw_idle()  # type: ignore[attr-defined]

    def reset_surface_views(self) -> None:
        for canvas in (self.advanced_input_canvas, self.advanced_reference_canvas):
            if isinstance(canvas, VispySurfaceWidget):
                canvas.reset_view()
                continue
            if hasattr(canvas, "surface_axis"):
                axis = canvas.surface_axis  # type: ignore[attr-defined]
                axis.view_init(elev=28, azim=-135)
                canvas.draw_idle()  # type: ignore[attr-defined]

    def standard_image_views(self) -> tuple[ZoomableImageView, ZoomableImageView, ZoomableImageView, ZoomableImageView]:
        return self.input_label, self.processed_label, self.original_label, self.variance_heatmap_label

    def zoom_standard_images_in(self) -> None:
        for view in self.standard_image_views():
            view.zoom_in()

    def zoom_standard_images_out(self) -> None:
        for view in self.standard_image_views():
            view.zoom_out()

    def reset_standard_image_views(self) -> None:
        for view in self.standard_image_views():
            view.reset_view()

    def update_qa_control_states(self) -> None:
        denoising_mode = self.qa_mode_combo.currentText() == "Denoising QA"
        self.residual_map_combo.setEnabled(denoising_mode)
        for widget in (
            self.variance_norm_combo,
            self.variance_map_combo,
            self.variance_clip_spin,
            self.variance_threshold_check,
        ):
            widget.setEnabled(not denoising_mode)

    def render_residual_heatmap(self) -> None:
        self.update_qa_control_states()
        self.export_variance_button.setText("Export Residual")
        residual_mode = self.residual_map_combo.currentText()
        candidate = self.current_frame if residual_mode == "Input - Reference Residual" else self.last_processed
        heatmap_rgb = residual_heatmap_rgb(candidate, self.hdf5_reference_frame)
        if heatmap_rgb is None:
            if self.variance_panel is not None:
                self.variance_panel.setVisible(False)
            self.variance_heatmap_label.setText("No residual map available")
            self.variance_heatmap_label.clear()
            self.variance_heatmap_title.setText("Residual Error")
            self.last_residual_heatmap_rgb = None
            return
        if self.variance_panel is not None:
            self.variance_panel.setVisible(True)
        self.last_residual_heatmap_rgb = heatmap_rgb
        self.variance_heatmap_label.set_image_rgb(heatmap_rgb)
        residual_metrics = (
            compare_to_reference(self.current_frame, self.hdf5_reference_frame)
            if residual_mode == "Input - Reference Residual"
            else {
                "reference_mse": self.latest_metrics.get("reference_mse", ""),
                "reference_psnr": self.latest_metrics.get("reference_psnr", ""),
                "reference_ssim": self.latest_metrics.get("reference_ssim", ""),
            }
        )
        mse = residual_metrics.get("reference_mse", "")
        psnr = residual_metrics.get("reference_psnr", "")
        ssim = residual_metrics.get("reference_ssim", "")
        summary = ""
        if isinstance(mse, (int, float)) and isinstance(psnr, (int, float)) and isinstance(ssim, (int, float)):
            summary = f" | MSE {mse:0.2f} | PSNR {psnr:0.2f} | SSIM {ssim:0.3f}"
        label = "input vs clean" if residual_mode == "Input - Reference Residual" else "processed vs clean"
        self.variance_heatmap_title.setText(f"Residual Error | {label}{summary}")

    def build_variance_heatmap_rgb(self) -> np.ndarray | None:
        variance = self.selected_variance_display_map()
        if variance is None:
            self.last_variance_heatmap_rgb = None
            return None

        variance = np.asarray(variance, dtype=np.float32)
        if self.variance_norm_combo.currentText() == "Percentile Clip":
            high = float(np.percentile(variance, self.variance_clip_spin.value()))
            low = float(np.percentile(variance, 1.0))
            if high <= low:
                heatmap_gray = normalize_to_uint8(variance)
            else:
                clipped = np.clip(variance, low, high)
                heatmap_gray = np.clip((clipped - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        else:
            heatmap_gray = normalize_to_uint8(variance)

        heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_MAGMA)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        if self.variance_threshold_check.isChecked():
            threshold = float(np.percentile(variance, self.variance_clip_spin.value()))
            mask = variance >= threshold
            heatmap_rgb[mask] = np.array([54, 249, 246], dtype=np.uint8)

        self.last_variance_heatmap_rgb = heatmap_rgb
        return heatmap_rgb

    def get_processed_z_variance_map(self) -> np.ndarray | None:
        if self.hdf5_path is None:
            return None
        group = str(self.hdf5_input_combo.currentData())
        sample = str(self.hdf5_sample_combo.currentData())
        key = (
            str(self.hdf5_path),
            group,
            sample,
            self.algorithm_combo.currentText(),
            self.strength_slider.value(),
            self.threshold_slider.value(),
        )
        if self.processed_z_variance_cache_key == key:
            return self.processed_z_variance_map
        self.status_label.setText("Computing processed z-variance QA map...")
        self.processed_z_variance_map = compute_processed_z_variance_map(
            self.hdf5_path,
            group,
            sample,
            self.algorithm_combo.currentText(),
            self.strength_slider.value(),
            self.threshold_slider.value(),
        )
        self.processed_z_variance_cache_key = key
        return self.processed_z_variance_map

    def selected_variance_display_map(self) -> np.ndarray | None:
        mode = self.variance_map_combo.currentText()
        if mode == "Reference Variance":
            self.variance_qa_metrics = compare_variance_maps(self.reference_z_variance_map, self.reference_z_variance_map)
            return self.reference_z_variance_map
        if mode == "Input - Reference Difference":
            self.variance_qa_metrics = compare_variance_maps(self.z_variance_map, self.reference_z_variance_map)
            return variance_difference_display_map(self.z_variance_map, self.reference_z_variance_map)
        if mode == "Processed - Reference Difference":
            processed_variance = self.get_processed_z_variance_map()
            self.variance_qa_metrics = compare_variance_maps(processed_variance, self.reference_z_variance_map)
            return variance_difference_display_map(processed_variance, self.reference_z_variance_map)

        self.variance_qa_metrics = compare_variance_maps(self.z_variance_map, self.reference_z_variance_map)
        return self.z_variance_map

    def export_variance_heatmap(self) -> None:
        if self.qa_mode_combo.currentText() == "Denoising QA":
            residual_mode = self.residual_map_combo.currentText()
            candidate = self.current_frame if residual_mode == "Input - Reference Residual" else self.last_processed
            heatmap_rgb = self.last_residual_heatmap_rgb if self.last_residual_heatmap_rgb is not None else residual_heatmap_rgb(
                candidate,
                self.hdf5_reference_frame,
            )
            if heatmap_rgb is None:
                QMessageBox.information(self, "No residual map", "Open a paired HDF5 sample and process it before exporting.")
                return
            default_name = "residual_error_map.png"
            if self.hdf5_path is not None:
                sample = str(self.hdf5_sample_combo.currentData())
                group = str(self.hdf5_input_combo.currentData())
                mode_name = "input_reference" if residual_mode == "Input - Reference Residual" else "processed_reference"
                default_name = f"{self.hdf5_path.stem}_{group}_{sample}_{mode_name}_residual_error_map.png"
            path_text, _ = QFileDialog.getSaveFileName(
                self,
                "Export residual error map",
                str(APP_DIR / default_name),
                "PNG images (*.png);;All files (*.*)",
            )
            if not path_text:
                return
            path = Path(path_text)
            if path.suffix.lower() != ".png":
                path = path.with_suffix(".png")
            cv2.imwrite(str(path), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
            self.status_label.setText(f"Exported residual error map: {path.name}")
            return

        heatmap_rgb = self.last_variance_heatmap_rgb if self.last_variance_heatmap_rgb is not None else self.build_variance_heatmap_rgb()
        if heatmap_rgb is None:
            QMessageBox.information(self, "No variance heatmap", "Open a stack-shaped HDF5 sample before exporting.")
            return
        default_name = "variance_heatmap.png"
        if self.hdf5_path is not None:
            sample = str(self.hdf5_sample_combo.currentData())
            group = str(self.hdf5_input_combo.currentData())
            default_name = f"{self.hdf5_path.stem}_{group}_{sample}_variance_heatmap.png"
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Export variance heatmap",
            str(APP_DIR / default_name),
            "PNG images (*.png);;All files (*.*)",
        )
        if not path_text:
            return
        path = Path(path_text)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        cv2.imwrite(str(path), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
        self.status_label.setText(f"Exported variance heatmap: {path.name}")

    def toggle_advanced_view(self) -> None:
        if not self.show_advanced_view:
            if self.current_frame is None:
                self.status_label.setText("Open an image, video frame, HDF5 sample, or satellite pair before using Image Space View")
                return
            self.show_advanced_view = True
        else:
            self.show_advanced_view = False
        self.views_stack.setCurrentWidget(self.advanced_views_widget if self.show_advanced_view else self.standard_views_widget)
        self.advanced_view_button.setText("Standard View" if self.show_advanced_view else "Image Space View")
        self.refresh_advanced_view()

    def refresh_advanced_view(self) -> None:
        if not self.show_advanced_view:
            return
        self.surface_render_timer.start()

    def render_image_space_surfaces(self) -> None:
        if not self.show_advanced_view:
            return
        processed_frame = self.last_processed if self.last_processed is not None else self.current_frame
        algorithm = self.algorithm_combo.currentText() if self.show_processed else "Original"
        self.plot_surface_canvas(self.advanced_input_canvas, processed_frame, f"{algorithm} x/y/v")
        reference_frame = self.hdf5_reference_frame
        reference_title = "Clean/Reference x/y/v"
        if self.satellite_pre_frame is not None:
            reference_frame = self.satellite_aligned_pre_frame()
            reference_title = "Aligned Pre/Reference x/y/v"
        elif reference_frame is None:
            reference_frame = self.current_frame
            reference_title = "Current Image x/y/v"
        self.plot_surface_canvas(self.advanced_reference_canvas, reference_frame, reference_title)

    def render_variance_heatmap(self) -> None:
        self.update_qa_control_states()
        if self.qa_mode_combo.currentText() == "Denoising QA":
            self.render_residual_heatmap()
            return

        self.export_variance_button.setText("Export Variance")
        selected_map = self.selected_variance_display_map()
        if selected_map is None:
            if self.variance_panel is not None:
                self.variance_panel.setVisible(False)
            self.variance_heatmap_label.setText("No z-stack variance map loaded")
            self.variance_heatmap_label.clear()
            self.variance_heatmap_title.setText("Variance Heatmap")
            if self.latest_metrics:
                self.latest_metrics.update(self.variance_qa_metrics)
                self.update_metrics_table(self.latest_metrics)
            return
        if self.variance_panel is not None:
            self.variance_panel.setVisible(True)
        heatmap_rgb = self.build_variance_heatmap_rgb()
        if heatmap_rgb is not None:
            self.variance_heatmap_label.set_image_rgb(heatmap_rgb)
        sample = str(self.hdf5_sample_combo.currentData()) if self.hdf5_path is not None else ""
        group = str(self.hdf5_input_combo.currentData()) if self.hdf5_path is not None else ""
        mode = self.variance_norm_combo.currentText()
        overlay = " | overlay" if self.variance_threshold_check.isChecked() else ""
        self.variance_heatmap_title.setText(
            f"{self.variance_map_combo.currentText()} | {group}/{sample} | {mode} {self.variance_clip_spin.value()}%{overlay}"
        )
        if self.latest_metrics:
            self.latest_metrics.update(self.variance_qa_metrics)
            self.update_metrics_table(self.latest_metrics)

    def render_advanced_images(self) -> None:
        if self.current_frame is not None:
            self.advanced_input_label.setPixmap(
                pixmap_from_rgb(self.current_frame, self.advanced_input_label.width(), self.advanced_input_label.height())
            )
        else:
            self.advanced_input_label.setText("No input frame")
        if self.hdf5_reference_frame is not None:
            self.advanced_reference_label.setPixmap(
                pixmap_from_rgb(
                    self.hdf5_reference_frame,
                    self.advanced_reference_label.width(),
                    self.advanced_reference_label.height(),
                )
            )
        else:
            self.advanced_reference_label.setText("No reference frame")

    def clear_hdf5_dataset(self) -> None:
        self.hdf5_path = None
        self.hdf5_catalog = None
        self.hdf5_reference_frame = None
        self.z_variance_metrics = empty_z_variance_metrics()
        self.variance_qa_metrics = empty_variance_qa_metrics()
        self.z_variance_map = None
        self.reference_z_variance_map = None
        self.processed_z_variance_map = None
        self.processed_z_variance_cache_key = None
        self.show_advanced_view = False
        self.advanced_view_button.setEnabled(False)
        self.advanced_view_button.setText("Image Space View")
        self.variance_heatmap_label.clear()
        self.variance_heatmap_label.setText("No z-stack variance map loaded")
        self.variance_heatmap_title.setText("Variance Heatmap")
        self.last_variance_heatmap_rgb = None
        self.last_residual_heatmap_rgb = None
        if self.variance_panel is not None:
            self.variance_panel.setVisible(False)
        self.set_paired_hdf5_view_enabled(False)
        self.views_stack.setCurrentWidget(self.standard_views_widget)
        self.hdf5_controls.setVisible(False)
        self.update_source_options_for_mode()

    def populate_hdf5_controls(self, path: Path, catalog: dict[str, object], selected: dict[str, object]) -> None:
        self.hdf5_loading_controls = True
        self.hdf5_path = path
        self.hdf5_catalog = catalog
        groups: dict[str, dict[str, dict[str, object]]] = catalog["groups"]  # type: ignore[assignment]
        samples: list[str] = catalog["common_samples"] or catalog["all_samples"]  # type: ignore[assignment]

        for combo in (self.hdf5_sample_combo, self.hdf5_input_combo, self.hdf5_reference_combo):
            combo.clear()
        for sample in samples:
            self.hdf5_sample_combo.addItem(sample, sample)
        for group in sorted(groups):
            self.hdf5_input_combo.addItem(group, group)
            self.hdf5_reference_combo.addItem(group, group)

        self.hdf5_sample_combo.setCurrentText(str(selected["sample"]))
        self.hdf5_input_combo.setCurrentText(str(selected["input_group"]))
        self.hdf5_reference_combo.setCurrentText(str(selected["reference_group"]))
        self.hdf5_file_label.setText(f"HDF5: {path.name}")
        self.update_hdf5_z_range(set_middle=False)
        self.hdf5_z_spin.setValue(int(selected["z_index"]))
        self.hdf5_controls.setVisible(True)
        self.hdf5_loading_controls = False

    def update_hdf5_z_range(self, set_middle: bool = True) -> None:
        if self.hdf5_catalog is None:
            return
        groups: dict[str, dict[str, dict[str, object]]] = self.hdf5_catalog["groups"]  # type: ignore[assignment]
        sample = str(self.hdf5_sample_combo.currentData())
        group = str(self.hdf5_input_combo.currentData())
        info = groups.get(group, {}).get(sample)
        shape = tuple(info["shape"]) if info else ()  # type: ignore[index]
        self.hdf5_z_spin.blockSignals(True)
        if len(shape) == 3 and shape[-1] not in (3, 4):
            self.hdf5_z_spin.setEnabled(True)
            self.hdf5_z_spin.setRange(0, int(shape[0]) - 1)
            if set_middle:
                self.hdf5_z_spin.setValue(int(shape[0]) // 2)
        else:
            self.hdf5_z_spin.setEnabled(False)
            self.hdf5_z_spin.setRange(0, 0)
            self.hdf5_z_spin.setValue(0)
        self.hdf5_z_spin.blockSignals(False)

    def on_hdf5_input_changed(self) -> None:
        if self.hdf5_loading_controls:
            return
        self.update_hdf5_z_range(set_middle=True)
        self.load_current_hdf5_selection()

    def on_hdf5_sample_changed(self) -> None:
        if self.hdf5_loading_controls:
            return
        self.update_hdf5_z_range(set_middle=True)
        self.load_current_hdf5_selection()

    def step_hdf5_sample(self, offset: int) -> None:
        count = self.hdf5_sample_combo.count()
        if count <= 0:
            return
        self.hdf5_sample_combo.setCurrentIndex((self.hdf5_sample_combo.currentIndex() + offset) % count)

    def load_current_hdf5_selection(self) -> None:
        if self.hdf5_loading_controls or self.hdf5_path is None or self.hdf5_catalog is None:
            return
        groups: dict[str, dict[str, dict[str, object]]] = self.hdf5_catalog["groups"]  # type: ignore[assignment]
        sample = str(self.hdf5_sample_combo.currentData())
        input_group = str(self.hdf5_input_combo.currentData())
        reference_group = str(self.hdf5_reference_combo.currentData())
        z_index = self.hdf5_z_spin.value()
        try:
            frame = load_hdf5_sample_frame(self.hdf5_path, input_group, sample, z_index)
            self.hdf5_reference_frame = (
                load_hdf5_sample_frame(self.hdf5_path, reference_group, sample, z_index)
                if reference_group in groups and sample in groups[reference_group]
                else None
            )
            self.z_variance_map = compute_z_variance_map(self.hdf5_path, input_group, sample)
            self.reference_z_variance_map = compute_z_variance_map(self.hdf5_path, reference_group, sample)
            self.processed_z_variance_map = None
            self.processed_z_variance_cache_key = None
            self.z_variance_metrics = z_variance_metrics_from_map(self.z_variance_map)
            self.variance_qa_metrics = compare_variance_maps(self.z_variance_map, self.reference_z_variance_map)
        except Exception as exc:
            QMessageBox.warning(self, "HDF5 load failed", str(exc))
            return

        surface_backend_available = scene is not None or FigureCanvas is not None
        self.advanced_view_button.setEnabled(self.hdf5_reference_frame is not None and surface_backend_available)
        if self.show_advanced_view and self.hdf5_reference_frame is None:
            self.toggle_advanced_view()
        self.fps.reset()
        self.current_source = f"hdf5:{self.hdf5_path.name}:{input_group}/{sample}:z{z_index}"
        self.frame_counter = 0
        self.last_processed = None
        self.previous_frame = None
        self.comparison_frame = None
        self.set_paired_hdf5_view_enabled(True)
        self.update_source_options_for_mode()
        self.input_title.setText(f"Raw Input | {input_group}")
        self.processed_title.setText(f"Processed Input | {self.algorithm_combo.currentText()}")
        self.original_title.setText(f"Clean Reference | {reference_group}")
        self.input_label.set_image_rgb(frame)
        self.set_frame(frame)
        self.refresh_advanced_view()
        self.render_variance_heatmap()
        ref_text = f" | reference {reference_group}/{sample}" if self.hdf5_reference_frame is not None else ""
        self.status_label.setText(f"HDF5: {input_group}/{sample} z{z_index}{ref_text}")

    def set_display_mode(self, processed: bool) -> None:
        self.show_processed = processed
        self.processed_label.setVisible(processed)
        self.original_only_button.setEnabled(processed)
        self.analysis_button.setEnabled(not processed)
        self.update_algorithm_label()

    def update_algorithm_label(self) -> None:
        self.algorithm_label.setText(
            f"Algorithm: {self.algorithm_combo.currentText()} | "
            f"strength {self.strength_slider.value()} | threshold {self.threshold_slider.value()} | "
            f"process every {self.process_every_spin.value()} frame(s)"
        )
        if self.satellite_pre_frame is not None:
            self.processed_title.setText(f"Change Output | {self.algorithm_combo.currentText()}")
        elif self.hdf5_path is not None:
            self.processed_title.setText(f"Processed Input | {self.algorithm_combo.currentText()}")
        self.last_processed = None
        self.processed_z_variance_map = None
        self.processed_z_variance_cache_key = None
        if self.satellite_pre_frame is not None:
            self.refresh_satellite_shared_gxh()
        if self.show_processed and self.current_frame is not None:
            self.render_processed(self.current_frame)
            self.refresh_current_metrics()
        if self.hdf5_path is not None:
            self.render_variance_heatmap()
        self.refresh_pane_views()

    def open_camera(self) -> None:
        self.stop_sources()
        self.clear_hdf5_dataset()
        self.clear_satellite_pair()
        self.update_source_options_for_mode()
        self.fps.reset()
        self.current_source = "camera"
        self.frame_counter = 0
        self.last_processed = None
        self.previous_frame = None
        self.comparison_frame = None
        self.camera_worker = CameraWorker(
            self.camera_spin.value(),
            self.backend_combo.currentText(),
            self.format_combo.currentText(),
            self.requested_width,
            self.requested_height,
            self.capture_fps,
        )
        self.camera_worker.frame_ready.connect(self.set_frame)
        self.camera_worker.failed_frame.connect(self.mark_failed_frame)
        self.camera_worker.status.connect(self.status_label.setText)
        self.camera_worker.start()

    def open_image(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images and HDF5 (*.bmp *.gif *.h5 *.hdf5 *.jpg *.jpeg *.png *.tif *.tiff);;All files (*.*)",
        )
        if not path_text:
            return
        path = Path(path_text)
        self.stop_sources()
        self.clear_satellite_pair()
        self.update_source_options_for_mode()
        try:
            if path.suffix.lower() in HDF5_EXTENSIONS:
                catalog = hdf5_dataset_catalog(path)
                dialog = Hdf5DatasetDialog(path, catalog, self)
                if dialog.exec_() != QDialog.Accepted:
                    self.clear_hdf5_dataset()
                    return
                selected = dialog.selected_values()
                self.populate_hdf5_controls(path, catalog, selected)
                self.load_current_hdf5_selection()
                return
            else:
                self.clear_hdf5_dataset()
                self.clear_satellite_pair()
                frame = load_image(path)
                self.current_source = f"image:{path.name}"
        except Exception as exc:
            QMessageBox.warning(self, "Image open failed", str(exc))
            return
        self.fps.reset()
        self.frame_counter = 0
        self.last_processed = None
        self.previous_frame = None
        self.comparison_frame = None
        self.set_frame(frame)
        self.status_label.setText(f"Image: {self.current_source}")

    def clear_satellite_pair(self) -> None:
        self.satellite_pre_frame = None
        self.satellite_post_frame = None
        self.satellite_shared_gxh_frame = None
        self.satellite_shared_gxh_mask = None
        self.satellite_pair_name = ""
        if self.hdf5_path is None:
            self.hdf5_reference_frame = None
        self.update_source_options_for_mode()

    def show_idrt_dataset_help(self, detail: str) -> None:
        message = (
            f"{detail}\n\n"
            "The IDRT satellite preview dataset is intentionally not included in this repository. "
            "This keeps GAIA's public package free of external source imagery and generated report artifacts.\n\n"
            "To run this workflow, set IDRT_DATASET_DIR to a local folder containing the supplied IDRT image files, "
            "then restart GAIA.\n\n"
            "Example:\n"
            '$env:IDRT_DATASET_DIR=\"C:\\path\\to\\IDRT application dataset\"\n'
            ".\\run_gaia.bat\n\n"
            "Without that dataset, use Open Image or Open Video to analyze your own local imagery."
        )
        self.status_label.setText("IDRT dataset unavailable. Set IDRT_DATASET_DIR or use Open Image.")
        QMessageBox.information(self, "IDRT dataset unavailable", message)

    def open_idrt_pair(self) -> None:
        if not DEFAULT_IDRT_DATASET_DIR.exists():
            self.show_idrt_dataset_help(f"Expected dataset folder was not found:\n{DEFAULT_IDRT_DATASET_DIR}")
            return
        labels = [pair[0] for pair in IDRT_PAIRS]
        label, accepted = QInputDialog.getItem(
            self,
            "Open IDRT image pair",
            "Pair",
            labels,
            0,
            False,
        )
        if not accepted or not label:
            return
        pair_lookup = {pair[0]: pair for pair in IDRT_PAIRS}
        _pair_label, pre_name, post_name = pair_lookup[label]
        pre_path = DEFAULT_IDRT_DATASET_DIR / pre_name
        post_path = DEFAULT_IDRT_DATASET_DIR / post_name
        missing_files = [path.name for path in (pre_path, post_path) if not path.exists()]
        if missing_files:
            self.show_idrt_dataset_help(
                "The selected IDRT pair is missing required files:\n"
                + "\n".join(f"- {name}" for name in missing_files)
                + f"\n\nConfigured dataset folder:\n{DEFAULT_IDRT_DATASET_DIR}"
            )
            return
        try:
            pre_frame = load_image(pre_path)
            post_frame = load_image(post_path)
        except Exception as exc:
            QMessageBox.warning(self, "IDRT pair open failed", str(exc))
            return

        self.stop_sources()
        self.clear_hdf5_dataset()
        self.satellite_pre_frame = pre_frame
        self.satellite_post_frame = post_frame
        self.satellite_pair_name = label
        self.hdf5_reference_frame = pre_frame
        self.update_source_options_for_mode()
        self.fps.reset()
        self.current_source = f"satellite:{label}"
        self.frame_counter = 0
        self.last_processed = None
        self.previous_frame = pre_frame
        self.comparison_frame = pre_frame
        self.set_paired_hdf5_view_enabled(True)
        self.input_title.setText("Post-event image")
        self.processed_title.setText(f"Change Output | {self.algorithm_combo.currentText()}")
        self.original_title.setText("Pre/reference image")
        self.input_label.set_image_rgb(post_frame)
        self.original_label.set_image_rgb(pre_frame)
        self.algorithm_combo.setCurrentText("Satellite Change Heatmap")
        self.set_frame(post_frame)
        self.status_label.setText(f"IDRT satellite pair: {label}")

    def open_video(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Videos (*.avi *.m4v *.mov *.mp4 *.mpeg *.mpg *.wmv);;All files (*.*)",
        )
        if not path_text:
            return
        self.stop_sources()
        self.clear_hdf5_dataset()
        self.clear_satellite_pair()
        try:
            self.video_frames = load_video_frames(Path(path_text))
        except Exception as exc:
            QMessageBox.warning(self, "Video open failed", str(exc))
            return
        self.video_index = 0
        self.fps.reset()
        self.current_source = f"video:{Path(path_text).name}"
        self.frame_counter = 0
        self.last_processed = None
        self.previous_frame = None
        self.comparison_frame = None
        self.frame_slider.setEnabled(True)
        self.frame_slider.setRange(0, len(self.video_frames) - 1)
        self.frame_slider.setValue(0)
        self.set_frame(self.video_frames[0])
        self.status_label.setText(f"Video: {Path(path_text).name} ({len(self.video_frames)} frames)")

    def set_frame(self, frame_rgb: np.ndarray) -> None:
        frame = ensure_rgb(frame_rgb)
        previous_for_processing = self.satellite_pre_frame if self.satellite_pre_frame is not None else self.previous_frame
        self.comparison_frame = previous_for_processing
        self.current_frame = frame
        self.frame_counter += 1
        instant, rolling = self.fps.mark_frame()
        if self.satellite_pre_frame is not None:
            aligned_reference = self.satellite_aligned_pre_frame()
            if aligned_reference is not None:
                self.hdf5_reference_frame = aligned_reference
            self.refresh_satellite_shared_gxh()
        else:
            pass
        if self.show_processed:
            should_process = self.last_processed is None or self.frame_counter % self.process_every_spin.value() == 0
            if should_process:
                self.set_progress(10, "Processing")
                self.last_processed = self.compute_processed(frame, previous_for_processing)
                self.set_progress(100, "Processing complete")
                QTimer.singleShot(700, self.reset_progress)
            if self.last_processed is not None:
                self.render_processed_image(self.last_processed)
        else:
            self.refresh_pane_views()
        gray = rgb_to_luminance(frame)
        mean = float(gray.mean())
        self.latest_metrics = self.build_metrics(gray, self.last_processed, instant, rolling)
        self.update_metrics_table(self.latest_metrics)
        self.refresh_annotations_table()
        self.refresh_annotation_overlays()
        self.write_log_row_if_active()
        self.metrics_label.setText(
            f"FPS inst {instant:0.1f} | avg {rolling:0.1f} | failed {self.fps.failed_frames} | brightness {mean:0.1f}/255"
        )
        self.advanced_view_button.setEnabled((scene is not None or FigureCanvas is not None) and self.current_frame is not None)
        if self.satellite_pre_frame is None:
            self.previous_frame = frame

    def render_processed(self, frame: np.ndarray) -> None:
        processed = self.compute_processed(frame, self.comparison_frame)
        self.last_processed = processed
        self.render_processed_image(processed)

    def compute_processed(self, frame: np.ndarray, previous_frame: np.ndarray | None) -> np.ndarray:
        if self.algorithm_combo.currentText() == "Satellite Change Heatmap":
            reference_frame = self.satellite_aligned_pre_frame() if self.satellite_pre_frame is not None else previous_frame
            processed, similarity = satellite_change_heatmap(
                frame,
                reference_frame,
                self.strength_slider.value(),
                self.threshold_slider.value(),
                self.satellite_shared_gxh_mask,
            )
            self.adjacent_similarity_percent = similarity
            return processed
        if self.algorithm_combo.currentText() == "Satellite Change Mask":
            reference_frame = self.satellite_aligned_pre_frame() if self.satellite_pre_frame is not None else previous_frame
            processed, similarity = satellite_change_mask(
                frame,
                reference_frame,
                self.strength_slider.value(),
                self.threshold_slider.value(),
                self.satellite_shared_gxh_mask,
            )
            self.adjacent_similarity_percent = similarity
            return processed
        if self.algorithm_combo.currentText() == "Satellite Shared GxH Overlap":
            reference_frame = self.satellite_aligned_pre_frame() if self.satellite_pre_frame is not None else previous_frame
            if self.satellite_shared_gxh_frame is not None and self.satellite_shared_gxh_mask is not None:
                processed = self.satellite_shared_gxh_frame
                support = self.satellite_shared_gxh_mask
            else:
                processed, support = satellite_shared_gxh_reference(frame, reference_frame, self.threshold_slider.value())
            self.adjacent_similarity_percent = float(np.mean(support) * 100.0) if support is not None else 100.0
            return processed
        if self.algorithm_combo.currentText() == "Frame Similarity Heatmap":
            processed, similarity = frame_similarity_heatmap(frame, previous_frame, self.strength_slider.value())
            self.adjacent_similarity_percent = similarity
            return processed
        self.adjacent_similarity_percent = 100.0
        target_frame = self.satellite_operator_frame(frame)
        return process_frame(
            target_frame,
            self.algorithm_combo.currentText(),
            self.strength_slider.value(),
            self.threshold_slider.value(),
        )

    def render_processed_image(self, processed: np.ndarray) -> None:
        self.refresh_pane_views()
        self.refresh_advanced_view()

    def refresh_current_metrics(self) -> None:
        if self.current_frame is None:
            return
        gray = rgb_to_luminance(self.current_frame)
        self.latest_metrics = self.build_metrics(gray, self.last_processed, 0.0, 0.0)
        self.update_metrics_table(self.latest_metrics)

    def build_metrics(
        self,
        gray: np.ndarray,
        processed: np.ndarray | None,
        instant_fps: float,
        rolling_fps: float,
    ) -> dict[str, object]:
        original = image_metrics(gray)
        processed_values = (
            processed_metrics(processed)
            if processed is not None
            else {"processed_mean": 0.0, "processed_std": 0.0, "processed_active_percent": 0.0}
        )
        reference_values = compare_to_reference(processed, self.hdf5_reference_frame)
        algorithm = self.algorithm_combo.currentText() if self.show_processed else "Original Only"
        sensor_values = sensor_domain_metrics(self.current_frame, processed, algorithm, self.threshold_slider.value())
        template_values = template_match_metrics(self.template_match_result)
        satellite_values = satellite_change_metrics(
            self.current_frame,
            self.satellite_aligned_pre_frame(),
            self.threshold_slider.value(),
        )
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.current_source,
            "algorithm": algorithm,
            "operator_target": self.operator_target_combo.currentText() if self.satellite_pre_frame is not None else "Current image",
            "strength": self.strength_slider.value(),
            "threshold": self.threshold_slider.value(),
            "process_every_n": self.process_every_spin.value(),
            "alignment_x_px": self.align_x_spin.value() if self.satellite_pre_frame is not None else "",
            "alignment_y_px": self.align_y_spin.value() if self.satellite_pre_frame is not None else "",
            "alignment_rotation_deg": self.align_rotation_spin.value() if self.satellite_pre_frame is not None else "",
            "alignment_scale_percent": self.align_scale_spin.value() if self.satellite_pre_frame is not None else "",
            "projective_alignment": "enabled" if self.satellite_homography is not None else "off",
            "projective_alignment_score": self.satellite_homography_score,
            "projective_alignment_inliers": self.satellite_homography_inliers,
            "template_search_base": self.template_base_combo.currentText(),
            "template_filter_source": self.template_filter_source_combo.currentText(),
            "instant_fps": instant_fps,
            "rolling_fps": rolling_fps,
            "failed_frames": self.fps.failed_frames,
            "original_mean": original["mean"],
            "original_median": original["median"],
            "original_min": original["min"],
            "original_max": original["max"],
            "original_std": original["std"],
            "original_range": original["range"],
            "black_clip_percent": original["black_clip_percent"],
            "white_clip_percent": original["white_clip_percent"],
            "focus_score": original["focus_score"],
            "adjacent_similarity_percent": self.adjacent_similarity_percent,
            **processed_values,
            **sensor_values,
            **template_values,
            **reference_values,
            **satellite_values,
            **self.z_variance_metrics,
            **self.variance_qa_metrics,
        }

    def update_metrics_table(self, values: dict[str, object]) -> None:
        for row, name in enumerate(METRIC_NAMES):
            value = values.get(name, "--")
            if isinstance(value, float):
                text = f"{value:0.3f}"
            else:
                text = str(value)
            self.metrics_table.item(row, 1).setText(text)

    def mark_failed_frame(self) -> None:
        self.fps.failed_frames += 1

    def save_current_row(self) -> None:
        if not self.latest_metrics:
            self.status_label.setText("No metrics available to save yet")
            return
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Save current metrics row",
            "gaia_metrics_row.csv",
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METRIC_NAMES)
                writer.writeheader()
                writer.writerow(self.csv_ready_row(self.latest_metrics))
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.log_label.setText(f"Saved metrics row: {path.name}")

    def save_current_experiment(self) -> None:
        if self.current_frame is None or not self.latest_metrics:
            self.status_label.setText("No frame or metrics available to save yet")
            return

        run_id = experiment_id()
        run_dir = EXPERIMENTS_DIR / run_id
        snapshots_dir = run_dir / "snapshots"
        try:
            snapshots_dir.mkdir(parents=True, exist_ok=False)
            experiment_annotations = self.current_relevant_annotations()
            original_path = snapshots_dir / "original.png"
            cv2.imwrite(str(original_path), cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))

            snapshots = ["snapshots/original.png"]
            if self.last_processed is not None:
                processed_path = snapshots_dir / "processed.png"
                if self.last_processed.ndim == 3:
                    processed_bgr = cv2.cvtColor(self.last_processed, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(processed_path), processed_bgr)
                else:
                    cv2.imwrite(str(processed_path), self.last_processed)
                snapshots.append("snapshots/processed.png")
            if self.hdf5_reference_frame is not None:
                reference_path = snapshots_dir / "reference.png"
                cv2.imwrite(str(reference_path), cv2.cvtColor(self.hdf5_reference_frame, cv2.COLOR_RGB2BGR))
                snapshots.append("snapshots/reference.png")
            if self.satellite_pre_frame is not None:
                pre_path = snapshots_dir / "satellite_pre.png"
                cv2.imwrite(str(pre_path), cv2.cvtColor(self.satellite_pre_frame, cv2.COLOR_RGB2BGR))
                snapshots.append("snapshots/satellite_pre.png")
            if self.satellite_post_frame is not None:
                post_path = snapshots_dir / "satellite_post.png"
                cv2.imwrite(str(post_path), cv2.cvtColor(self.satellite_post_frame, cv2.COLOR_RGB2BGR))
                snapshots.append("snapshots/satellite_post.png")
            if self.satellite_pre_frame is not None:
                overlap_rgb, _support = satellite_shared_gxh_reference(
                    self.current_frame,
                    self.satellite_aligned_pre_frame(),
                    self.threshold_slider.value(),
                )
                overlap_path = snapshots_dir / "shared_gxh_overlap.png"
                cv2.imwrite(str(overlap_path), cv2.cvtColor(overlap_rgb, cv2.COLOR_RGB2BGR))
                snapshots.append("snapshots/shared_gxh_overlap.png")

            manifest = {
                "experiment_id": run_id,
                "created_at": self.latest_metrics.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                "source": self.current_source,
                "algorithm": self.latest_metrics.get("algorithm", ""),
                "settings": {
                    "strength": self.strength_slider.value(),
                    "threshold": self.threshold_slider.value(),
                    "process_every_n": self.process_every_spin.value(),
                    "operator_target": self.operator_target_combo.currentText() if self.satellite_pre_frame is not None else "Current image",
                    "alignment_x_px": self.align_x_spin.value() if self.satellite_pre_frame is not None else "",
                    "alignment_y_px": self.align_y_spin.value() if self.satellite_pre_frame is not None else "",
                    "alignment_rotation_deg": self.align_rotation_spin.value() if self.satellite_pre_frame is not None else "",
                    "alignment_scale_percent": self.align_scale_spin.value() if self.satellite_pre_frame is not None else "",
                    "projective_alignment": "enabled" if self.satellite_homography is not None else "off",
                    "projective_alignment_score": self.satellite_homography_score,
                    "projective_alignment_inliers": self.satellite_homography_inliers,
                    "display_mode": "processed" if self.show_processed else "original_only",
                    "image_space_view": self.show_advanced_view,
                    "surface_downsample": self.surface_downsample_spin.value(),
                },
                "frame_shape": list(self.current_frame.shape),
                "snapshots": snapshots,
                "annotation_count": len(experiment_annotations),
            }
            if self.hdf5_path is not None:
                manifest["hdf5"] = {
                    "path": str(self.hdf5_path),
                    "sample": str(self.hdf5_sample_combo.currentData()),
                    "input_group": str(self.hdf5_input_combo.currentData()),
                    "reference_group": str(self.hdf5_reference_combo.currentData()),
                    "z_index": self.hdf5_z_spin.value(),
                }
            if self.satellite_pre_frame is not None:
                manifest["satellite_analysis"] = {
                    "pair": self.satellite_pair_name,
                    "dataset_dir": str(DEFAULT_IDRT_DATASET_DIR),
                    "projective_homography": self.satellite_homography.tolist() if self.satellite_homography is not None else None,
                    "note": "Preview-image analysis for rapid visual screening; confirm with georeferenced source data before operational use.",
                }
            write_json(run_dir / "manifest.json", manifest)
            with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METRIC_NAMES)
                writer.writeheader()
                writer.writerow(self.csv_ready_row(self.latest_metrics))
            if experiment_annotations:
                with (run_dir / "annotations.jsonl").open("w", encoding="utf-8") as handle:
                    for annotation in experiment_annotations:
                        handle.write(json.dumps(json_safe(annotation), ensure_ascii=True) + "\n")
            if self.satellite_pre_frame is not None:
                self.write_satellite_report(run_dir, manifest, experiment_annotations)
        except OSError as exc:
            QMessageBox.warning(self, "Experiment save failed", str(exc))
            return

        self.active_experiment_dir = run_dir
        self.log_label.setText(f"Saved experiment: {run_id}")

    def write_satellite_report(
        self,
        run_dir: Path,
        manifest: dict[str, object],
        annotations: list[dict[str, object]] | None = None,
    ) -> None:
        metrics = self.latest_metrics
        annotations = annotations or []
        lines = [
            "# Satellite Change Analysis Draft",
            "",
            f"- Pair: {self.satellite_pair_name}",
            f"- Created: {manifest.get('created_at', '')}",
            f"- Algorithm: {metrics.get('algorithm', '')}",
            f"- Operator target: {metrics.get('operator_target', '')}",
            f"- Manual alignment: x {metrics.get('alignment_x_px', '')} px, y {metrics.get('alignment_y_px', '')} px, rotation {metrics.get('alignment_rotation_deg', '')} deg, scale {metrics.get('alignment_scale_percent', '')}%",
            f"- Projective alignment: {metrics.get('projective_alignment', '')}; score {metrics.get('projective_alignment_score', '')}; inliers {metrics.get('projective_alignment_inliers', '')}",
            f"- Source: {metrics.get('source', '')}",
            "",
            "## Quantitative Screening Metrics",
            "",
            f"- Mean RGB change delta: {metrics.get('change_mean_delta', '')}",
            f"- 95th percentile RGB change delta: {metrics.get('change_p95_delta', '')}",
            f"- Active change area above screening threshold: {metrics.get('change_active_percent', '')}%",
            f"- Water proxy change: {metrics.get('water_change_percent', '')} percentage points",
            f"- Green/vegetation proxy change: {metrics.get('green_change_mean', '')}",
            f"- Shared GxH overlap support: {metrics.get('shared_gxh_overlap_percent', '')}%",
            "",
            "## Shared-Overlap Method",
            "",
            "- The pre/reference and post/current images are each converted to a Gradient x Hessian structural map.",
            "- The tool computes the geometric mean `sqrt(GxH_pre * GxH_post)` to identify image regions with shared structural support.",
            "- Black crop bands and non-overlapping cut regions are suppressed before candidate change heatmaps are calculated.",
            "",
            "## Analyst Notes",
            "",
            "- This is preview-image analysis, not a georeferenced operational product.",
            "- Pixel-difference heatmaps are candidate change maps only; image crop, viewing angle, sensor response, tide, clouds, and shadows can create false change.",
            "- Manual alignment values record the visual correction applied before paired change detection.",
            "- Recommended next pass: annotate shoreline, marina/canal, roof/debris, flood/water-discoloration, and cloud-obscured areas separately.",
            "",
            "## Analyst Annotations",
            "",
        ]
        if annotations:
            for annotation in annotations:
                label = str(annotation.get("label", "Untitled annotation")).strip() or "Untitled annotation"
                annotation_type = annotation.get("annotation_type", "")
                pane_source = annotation.get("pane_source", "")
                center_x = annotation.get("center_x", "")
                center_y = annotation.get("center_y", "")
                notes = str(annotation.get("notes_plaintext", "")).strip().replace("\r\n", "\n")
                first_note_line = notes.split("\n", 1)[0] if notes else "No note text entered."
                lines.append(f"- {label} ({annotation_type}, {pane_source}, center {center_x}, {center_y}): {first_note_line}")
        else:
            lines.append("- No analyst annotations were saved with this experiment.")
        lines.extend(
            [
                "",
                "## Saved Evidence",
                "",
                "- `snapshots/satellite_pre.png`",
                "- `snapshots/satellite_post.png`",
                "- `snapshots/processed.png`",
                "- `snapshots/shared_gxh_overlap.png`",
                "- `metrics.csv`",
            ]
        )
        if annotations:
            lines.append("- `annotations.jsonl`")
        (run_dir / "satellite_report_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def refresh_experiments(self) -> None:
        self.experiments_list.blockSignals(True)
        self.experiments_list.clear()
        for path in list_experiments():
            try:
                manifest = read_json(path / "manifest.json")
            except (OSError, json.JSONDecodeError):
                continue
            summary = summarize_metrics_csv(path / "metrics.csv")
            created = manifest.get("created_at", "unknown time")
            source = manifest.get("source", "unknown source")
            algorithm = manifest.get("algorithm", "unknown algorithm")
            rows = summary.get("metric_rows", 0)
            annotation_count = manifest.get("annotation_count", 0)
            item = QListWidgetItem(f"{path.name}\n{created} | {source} | {algorithm} | {rows} row(s) | {annotation_count} annotation(s)")
            item.setData(Qt.UserRole, str(path))
            self.experiments_list.addItem(item)
        self.experiments_list.blockSignals(False)

        if self.experiments_list.count():
            self.experiments_list.setCurrentRow(0)
            self.show_selected_experiment(self.experiments_list.currentItem())
        else:
            self.experiment_summary_label.setText("No experiments saved yet.")
            self.experiment_details.setRowCount(0)

    def show_selected_experiment(self, current: QListWidgetItem | None, previous: QListWidgetItem | None = None) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.UserRole))
        try:
            manifest = read_json(path / "manifest.json")
            summary = summarize_metrics_csv(path / "metrics.csv")
        except (OSError, json.JSONDecodeError) as exc:
            self.experiment_summary_label.setText(f"Could not open experiment: {exc}")
            self.experiment_details.setRowCount(0)
            return

        settings = manifest.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        rows = [
            ("experiment_id", manifest.get("experiment_id", path.name)),
            ("created_at", manifest.get("created_at", "")),
            ("source", manifest.get("source", "")),
            ("algorithm", manifest.get("algorithm", "")),
            ("strength", settings.get("strength", "")),
            ("threshold", settings.get("threshold", "")),
            ("process_every_n", settings.get("process_every_n", "")),
            ("annotation_count", manifest.get("annotation_count", 0)),
            ("metric_rows", summary.get("metric_rows", 0)),
            ("avg_fps", summary.get("rolling_fps_avg", "")),
            ("brightness_avg", summary.get("original_mean_avg", "")),
            ("contrast_avg", summary.get("original_std_avg", "")),
            ("focus_min", summary.get("focus_score_min", "")),
            ("focus_max", summary.get("focus_score_max", "")),
            ("similarity_avg", summary.get("adjacent_similarity_percent_avg", "")),
            ("reference_mse_avg", summary.get("reference_mse_avg", "")),
            ("reference_psnr_avg", summary.get("reference_psnr_avg", "")),
            ("reference_ssim_avg", summary.get("reference_ssim_avg", "")),
            ("residual_mean_avg", summary.get("residual_mean_avg", "")),
            ("z_variance_mean_avg", summary.get("z_variance_mean_avg", "")),
            ("z_variance_max_avg", summary.get("z_variance_max_avg", "")),
            ("z_variance_p95_avg", summary.get("z_variance_p95_avg", "")),
            ("z_high_variance_percent_avg", summary.get("z_high_variance_percent_avg", "")),
        ]
        self.experiment_details.setRowCount(len(rows))
        for row_index, (name, value) in enumerate(rows):
            if isinstance(value, float):
                value_text = f"{value:0.3f}"
            else:
                value_text = str(value)
            self.experiment_details.setItem(row_index, 0, QTableWidgetItem(name))
            self.experiment_details.setItem(row_index, 1, QTableWidgetItem(value_text))
        self.experiment_summary_label.setText(f"Viewing {path.name}")

    def start_csv_log(self) -> None:
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Start CSV metrics log",
            "gaia_metrics_log.csv",
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path_text:
            return
        self.stop_csv_log()
        try:
            self.log_file = Path(path_text).open("w", newline="", encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Log open failed", str(exc))
            self.log_file = None
            return
        self.log_writer = csv.DictWriter(self.log_file, fieldnames=METRIC_NAMES)
        self.log_writer.writeheader()
        self.start_log_button.setEnabled(False)
        self.stop_log_button.setEnabled(True)
        self.log_label.setText(f"Logging CSV: {Path(path_text).name}")

    def stop_csv_log(self) -> None:
        if self.log_file is not None:
            self.log_file.close()
        self.log_file = None
        self.log_writer = None
        self.start_log_button.setEnabled(True)
        self.stop_log_button.setEnabled(False)
        self.log_label.setText("CSV log idle")

    def write_log_row_if_active(self) -> None:
        if self.log_writer is None or not self.latest_metrics:
            return
        self.log_writer.writerow(self.csv_ready_row(self.latest_metrics))
        if self.log_file is not None:
            self.log_file.flush()

    def csv_ready_row(self, values: dict[str, object]) -> dict[str, object]:
        return {name: values.get(name, "") for name in METRIC_NAMES}

    def current_relevant_annotations(self) -> list[dict[str, object]]:
        sources = {
            self.current_source,
            self.frame_for_source(self.left_pane_combo.currentText())[1],
            self.frame_for_source(self.center_pane_combo.currentText())[1],
            self.frame_for_source(self.right_pane_combo.currentText())[1],
            self.variance_heatmap_title.text(),
        }
        return [
            annotation
            for annotation in self.annotations
            if self.annotation_matches_current_context(annotation)
            and (annotation.get("source") in sources or annotation.get("pane_source") in sources)
        ]

    def show_video_frame(self, index: int) -> None:
        if not self.video_frames:
            return
        self.video_index = index
        self.set_frame(self.video_frames[index])

    def toggle_video(self) -> None:
        if not self.video_frames:
            return
        if self.video_timer.isActive():
            self.video_timer.stop()
            self.play_button.setText("Play")
        else:
            self.video_timer.start()
            self.play_button.setText("Pause")

    def advance_video(self) -> None:
        if not self.video_frames:
            return
        self.video_index = (self.video_index + 1) % len(self.video_frames)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.video_index)
        self.frame_slider.blockSignals(False)
        self.set_frame(self.video_frames[self.video_index])

    def stop_sources(self) -> None:
        self.video_timer.stop()
        self.play_button.setText("Play")
        self.video_frames = []
        self.frame_slider.setEnabled(False)
        self.current_source = "idle"
        self.previous_frame = None
        self.comparison_frame = None
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker = None

    def closeEvent(self, event) -> None:
        self.stop_sources()
        self.disconnect_ender()
        self.stop_csv_log()
        super().closeEvent(event)


