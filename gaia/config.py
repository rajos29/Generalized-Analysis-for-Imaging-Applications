from __future__ import annotations

import os
from pathlib import Path

import cv2

BACKENDS = {
    "DirectShow": cv2.CAP_DSHOW,
    "MSMF": cv2.CAP_MSMF,
    "Default": None,
}

FORMATS = {
    "Auto": None,
    "MJPG": "MJPG",
    "YUY2": "YUY2",
}

ALGORITHMS = (
    "Radar Range-Doppler Map",
    "LiDAR Depth Relief",
    "Ultrasonic B-Scan Envelope",
    "Satellite Change Heatmap",
    "Satellite Change Mask",
    "Satellite Shared GxH Overlap",
    "Satellite Water Emphasis",
    "Satellite Vegetation Proxy",
    "Gradient x Hessian",
    "Gradient Magnitude",
    "Hessian Magnitude",
    "CLAHE Contrast",
    "Otsu Threshold",
    "Adaptive Threshold",
    "Canny Edges",
    "Edge Overlay",
    "Frame Similarity Heatmap",
    "Denoise - Gaussian",
    "Denoise - Median",
    "Denoise - Bilateral",
    "Denoise - Non-Local Means",
    "Sharpen - Unsharp Mask",
    "Sharpen - Laplacian",
    "Sharpen - High Boost",
    "Morph Open",
    "Morph Close",
    "Focus Map",
    "Background Subtract",
)

PANE_SOURCES = (
    "Post/Event",
    "Raw/Input",
    "Processed Output",
    "Reference/Pre",
    "CIS Heatmap",
    "CIS Filter",
    "Shared GxH",
    "Current GxH",
)

SATELLITE_SOURCES = tuple(source for source in PANE_SOURCES if source != "Raw/Input")
GENERIC_SOURCES = tuple(source for source in PANE_SOURCES if source != "Post/Event")

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}
HDF5_EXTENSIONS = {".h5", ".hdf5"}
IDRT_PAIRS = (
    (
        "Maxar pre/post Treasure Island",
        "Maxar_pre_20231104_TreasureIsland_preview.png",
        "Maxar_post_20241010_TreasureIsland_preview.png",
    ),
    (
        "NOAA post-Helene/post-Milton Treasure Island",
        "NOAA_postHelene_20240930_TreasureIsland_preview.png",
        "NOAA_postMilton_20241011_TreasureIsland_preview.png",
    ),
    (
        "Sentinel-2 pre/post regional context",
        "S2_pre_20240919_preview.png",
        "S2_post_20241014_preview.png",
    ),
)
ROLLING_FPS_FRAMES = 90
APP_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = APP_DIR / "experiments"
ANNOTATIONS_PATH = EXPERIMENTS_DIR / "annotations.jsonl"
DEFAULT_IDRT_DATASET_DIR = Path(os.environ.get("IDRT_DATASET_DIR", APP_DIR / "data" / "idrt_application_dataset"))
METRIC_NAMES = (
    "timestamp",
    "source",
    "algorithm",
    "operator_target",
    "strength",
    "threshold",
    "process_every_n",
    "alignment_x_px",
    "alignment_y_px",
    "alignment_rotation_deg",
    "alignment_scale_percent",
    "projective_alignment",
    "projective_alignment_score",
    "projective_alignment_inliers",
    "template_search_base",
    "template_filter_source",
    "instant_fps",
    "rolling_fps",
    "failed_frames",
    "original_mean",
    "original_median",
    "original_min",
    "original_max",
    "original_std",
    "original_range",
    "black_clip_percent",
    "white_clip_percent",
    "focus_score",
    "processed_mean",
    "processed_std",
    "processed_active_percent",
    "radar_peak_energy",
    "radar_spectral_entropy",
    "lidar_relief_std",
    "lidar_edge_density_percent",
    "ultrasonic_echo_density_percent",
    "ultrasonic_echo_mean",
    "template_match_x",
    "template_match_y",
    "template_match_corr_score",
    "template_match_difference_score",
    "template_match_rotation_deg",
    "template_width_px",
    "template_height_px",
    "template_stride_px",
    "adjacent_similarity_percent",
    "reference_mse",
    "reference_psnr",
    "reference_ssim",
    "residual_mean",
    "residual_std",
    "z_variance_mean",
    "z_variance_max",
    "z_variance_p95",
    "z_high_variance_percent",
    "variance_reference_corr",
    "variance_high_overlap_percent",
    "variance_difference_mean",
    "variance_difference_p95",
    "change_mean_delta",
    "change_p95_delta",
    "change_active_percent",
    "water_pre_percent",
    "water_post_percent",
    "water_change_percent",
    "green_pre_mean",
    "green_post_mean",
    "green_change_mean",
    "shared_gxh_overlap_percent",
    "excluded_area_percent",
    "valid_post_area_percent",
    "valid_pre_area_percent",
    "structural_similarity_score",
    "change_median_delta",
    "change_p90_delta",
    "change_max_delta",
    "change_component_count",
    "filtered_change_region_count",
    "filtered_change_region_area_percent",
    "largest_change_region_area_px",
    "largest_change_component_percent",
    "mean_change_region_area_px",
    "median_change_region_area_px",
    "change_region_density_per_megapixel",
    "change_severity_index",
)

