from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
try:
    import h5py
except ImportError:
    h5py = None

from gaia.core.image_ops import ensure_rgb, normalize_to_uint8, process_frame, rgb_to_luminance
from gaia.core.satellite import affine_similarity_score, resize_like, satellite_change_diff, satellite_shared_gxh_reference, satellite_water_score, valid_coverage_mask

def image_metrics(gray: np.ndarray) -> dict[str, float]:
    gray = np.asarray(gray, dtype=np.uint8)
    gray_float = gray.astype(np.float32)
    return {
        "mean": float(gray_float.mean()),
        "median": float(np.median(gray_float)),
        "min": float(gray.min()),
        "max": float(gray.max()),
        "std": float(gray_float.std()),
        "range": float(gray.max() - gray.min()),
        "black_clip_percent": float(np.mean(gray <= 2) * 100.0),
        "white_clip_percent": float(np.mean(gray >= 253) * 100.0),
        "focus_score": float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var()),
    }


def processed_metrics(processed: np.ndarray) -> dict[str, float]:
    processed = np.asarray(processed, dtype=np.uint8)
    if processed.ndim == 3:
        processed = rgb_to_luminance(processed)
    processed_float = processed.astype(np.float32)
    return {
        "processed_mean": float(processed_float.mean()),
        "processed_std": float(processed_float.std()),
        "processed_active_percent": float(np.mean(processed > 0) * 100.0),
    }


def sensor_domain_metrics(
    frame_rgb: np.ndarray | None,
    processed: np.ndarray | None,
    algorithm: str,
    threshold: int,
) -> dict[str, float | str]:
    empty = {
        "radar_peak_energy": "",
        "radar_spectral_entropy": "",
        "lidar_relief_std": "",
        "lidar_edge_density_percent": "",
        "ultrasonic_echo_density_percent": "",
        "ultrasonic_echo_mean": "",
    }
    if frame_rgb is None or processed is None:
        return empty

    gray = rgb_to_luminance(ensure_rgb(frame_rgb)).astype(np.float32)
    output = ensure_rgb(processed) if np.asarray(processed).ndim == 3 else np.repeat(np.asarray(processed, dtype=np.uint8)[:, :, None], 3, axis=2)
    output_gray = rgb_to_luminance(output).astype(np.float32)

    if algorithm == "Radar Range-Doppler Map":
        centered = gray - float(np.mean(gray))
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
        energy = np.log1p(spectrum).astype(np.float32)
        probability = energy / max(float(energy.sum()), 1e-6)
        entropy = -float(np.sum(probability * np.log2(probability + 1e-12)))
        entropy /= float(np.log2(probability.size))
        return {
            **empty,
            "radar_peak_energy": float(np.percentile(energy, 99.5)),
            "radar_spectral_entropy": entropy,
        }

    if algorithm == "LiDAR Depth Relief":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        relief = cv2.magnitude(gx, gy)
        return {
            **empty,
            "lidar_relief_std": float(relief.std()),
            "lidar_edge_density_percent": float(np.mean(relief >= max(10, threshold)) * 100.0),
        }

    if algorithm == "Ultrasonic B-Scan Envelope":
        echo_mask = output_gray >= max(10, threshold)
        return {
            **empty,
            "ultrasonic_echo_density_percent": float(np.mean(echo_mask) * 100.0),
            "ultrasonic_echo_mean": float(output_gray[echo_mask].mean()) if echo_mask.any() else 0.0,
        }

    return empty


def template_match_metrics(result: dict[str, object]) -> dict[str, float | int | str]:
    if not result:
        return {
            "template_match_x": "",
            "template_match_y": "",
            "template_match_corr_score": "",
            "template_match_difference_score": "",
            "template_match_rotation_deg": "",
            "template_width_px": "",
            "template_height_px": "",
            "template_stride_px": "",
        }
    return {
        "template_match_x": int(result.get("best_x", 0)),
        "template_match_y": int(result.get("best_y", 0)),
        "template_match_corr_score": float(result.get("corr_score", 0.0)),
        "template_match_difference_score": float(result.get("difference_score", 0.0)),
        "template_match_rotation_deg": float(result.get("rotation_deg", 0.0)),
        "template_width_px": int(result.get("template_width", 0)),
        "template_height_px": int(result.get("template_height", 0)),
        "template_stride_px": int(result.get("stride_px", 0)),
    }


def compare_to_reference(processed: np.ndarray | None, reference_rgb: np.ndarray | None) -> dict[str, float | str]:
    if processed is None or reference_rgb is None:
        return {
            "reference_mse": "",
            "reference_psnr": "",
            "reference_ssim": "",
            "residual_mean": "",
            "residual_std": "",
        }

    candidate = np.asarray(processed)
    if candidate.ndim == 3:
        candidate_gray = rgb_to_luminance(ensure_rgb(candidate))
    else:
        candidate_gray = candidate.astype(np.uint8, copy=False)
    reference_gray = rgb_to_luminance(ensure_rgb(reference_rgb))
    if candidate_gray.shape != reference_gray.shape:
        candidate_gray = cv2.resize(
            candidate_gray,
            (reference_gray.shape[1], reference_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    candidate_float = candidate_gray.astype(np.float32)
    reference_float = reference_gray.astype(np.float32)
    residual = candidate_float - reference_float
    mse = float(np.mean(residual * residual))
    psnr = 99.0 if mse <= 1e-9 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = cv2.GaussianBlur(candidate_float, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(reference_float, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(candidate_float * candidate_float, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(reference_float * reference_float, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(candidate_float * reference_float, (11, 11), 1.5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )

    return {
        "reference_mse": mse,
        "reference_psnr": psnr,
        "reference_ssim": float(np.mean(ssim_map)),
        "residual_mean": float(np.mean(np.abs(residual))),
        "residual_std": float(np.std(residual)),
    }


def residual_heatmap_rgb(processed: np.ndarray | None, reference_rgb: np.ndarray | None) -> np.ndarray | None:
    if processed is None or reference_rgb is None:
        return None
    if processed.ndim == 3:
        candidate_gray = rgb_to_luminance(ensure_rgb(processed))
    else:
        candidate_gray = processed.astype(np.uint8, copy=False)
    reference_gray = rgb_to_luminance(ensure_rgb(reference_rgb))
    if candidate_gray.shape != reference_gray.shape:
        candidate_gray = cv2.resize(
            candidate_gray,
            (reference_gray.shape[1], reference_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    residual = np.abs(candidate_gray.astype(np.float32) - reference_gray.astype(np.float32))
    residual_gray = normalize_to_uint8(residual)
    residual_bgr = cv2.applyColorMap(residual_gray, cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(residual_bgr, cv2.COLOR_BGR2RGB)


def empty_z_variance_metrics() -> dict[str, float | str]:
    return {
        "z_variance_mean": "",
        "z_variance_max": "",
        "z_variance_p95": "",
        "z_high_variance_percent": "",
    }


def empty_variance_qa_metrics() -> dict[str, float | str]:
    return {
        "variance_reference_corr": "",
        "variance_high_overlap_percent": "",
        "variance_difference_mean": "",
        "variance_difference_p95": "",
    }


from gaia.core.hdf5_io import frame_from_array

def compute_z_variance_map(path: Path, group: str, sample: str) -> np.ndarray | None:
    if h5py is None:
        return None
    dataset_name = f"{group}/{sample}"
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            return None
        dataset = handle[dataset_name]
        shape = tuple(int(value) for value in dataset.shape)
        if len(shape) != 3 or shape[-1] in (3, 4):
            return None
        stack = dataset[()].astype(np.float32)
    return np.var(stack, axis=0)


def load_hdf5_stack(path: Path, group: str, sample: str) -> np.ndarray | None:
    if h5py is None:
        return None
    dataset_name = f"{group}/{sample}"
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            return None
        dataset = handle[dataset_name]
        shape = tuple(int(value) for value in dataset.shape)
        if len(shape) != 3 or shape[-1] in (3, 4):
            return None
        return dataset[()]


def compute_processed_z_variance_map(
    path: Path,
    group: str,
    sample: str,
    algorithm: str,
    strength: int,
    threshold: int,
) -> np.ndarray | None:
    stack = load_hdf5_stack(path, group, sample)
    if stack is None:
        return None

    processed_slices = []
    for slice_data in stack:
        frame = frame_from_array(slice_data)
        processed = process_frame(frame, algorithm, strength, threshold)
        if processed.ndim == 3:
            gray = rgb_to_luminance(ensure_rgb(processed))
        else:
            gray = processed.astype(np.uint8, copy=False)
        processed_slices.append(gray.astype(np.float32))
    return np.var(np.stack(processed_slices, axis=0), axis=0)


def z_variance_metrics_from_map(variance: np.ndarray | None) -> dict[str, float | str]:
    if variance is None:
        return empty_z_variance_metrics()

    p95 = float(np.percentile(variance, 95))
    high_variance_percent = float(np.mean(variance >= p95) * 100.0) if p95 > 0 else 0.0
    return {
        "z_variance_mean": float(np.mean(variance)),
        "z_variance_max": float(np.max(variance)),
        "z_variance_p95": p95,
        "z_high_variance_percent": high_variance_percent,
    }


def normalize_float_map(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    data = np.asarray(values, dtype=np.float32)
    low = float(np.min(data))
    high = float(np.max(data))
    if high <= low:
        return np.zeros(data.shape, dtype=np.float32)
    return (data - low) / (high - low)


def variance_difference_display_map(candidate: np.ndarray | None, reference: np.ndarray | None) -> np.ndarray | None:
    candidate_norm = normalize_float_map(candidate)
    reference_norm = normalize_float_map(reference)
    if candidate_norm is None or reference_norm is None:
        return None
    if candidate_norm.shape != reference_norm.shape:
        reference_norm = cv2.resize(
            reference_norm,
            (candidate_norm.shape[1], candidate_norm.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return np.abs(candidate_norm - reference_norm)


def compare_variance_maps(candidate: np.ndarray | None, reference: np.ndarray | None) -> dict[str, float | str]:
    if candidate is None or reference is None:
        return empty_variance_qa_metrics()
    candidate = normalize_float_map(candidate)
    reference = normalize_float_map(reference)
    if candidate is None or reference is None:
        return empty_variance_qa_metrics()
    if candidate.shape != reference.shape:
        reference = cv2.resize(reference, (candidate.shape[1], candidate.shape[0]), interpolation=cv2.INTER_AREA)

    candidate_flat = candidate.ravel()
    reference_flat = reference.ravel()
    if float(np.std(candidate_flat)) <= 1e-9 or float(np.std(reference_flat)) <= 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(candidate_flat, reference_flat)[0, 1])

    candidate_threshold = float(np.percentile(candidate, 95))
    reference_threshold = float(np.percentile(reference, 95))
    candidate_mask = candidate >= candidate_threshold
    reference_mask = reference >= reference_threshold
    union = np.logical_or(candidate_mask, reference_mask)
    overlap = 0.0 if not np.any(union) else float(np.mean(np.logical_and(candidate_mask, reference_mask)[union]) * 100.0)
    difference = np.abs(candidate - reference)
    return {
        "variance_reference_corr": corr,
        "variance_high_overlap_percent": overlap,
        "variance_difference_mean": float(np.mean(difference)),
        "variance_difference_p95": float(np.percentile(difference, 95)),
    }


def compute_z_variance_metrics(path: Path, group: str, sample: str) -> dict[str, float | str]:
    return z_variance_metrics_from_map(compute_z_variance_map(path, group, sample))


