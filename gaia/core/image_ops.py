from __future__ import annotations

import cv2
import numpy as np

from gaia.config import ALGORITHMS

def ensure_rgb(frame: np.ndarray, source_order: str = "RGB") -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2).astype(np.uint8, copy=False)
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected grayscale or RGB frame, got {frame.shape}.")
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            max_value = 1.0 if float(np.nanmax(frame)) <= 1.0 else 255.0
            frame = np.clip(frame, 0.0, max_value) * (255.0 / max_value)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if source_order.upper() == "BGR":
        frame = frame[:, :, ::-1]
    return np.ascontiguousarray(frame)


def rgb_to_luminance(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = cv2.minMaxLoc(image)[:2]
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def edge_padded_smooth_3x3(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    padded = np.pad(data, ((1, 1), (1, 1)), mode="edge")
    return (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0


def biobridge_gradient(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    padded = np.pad(data, ((0, 1), (0, 1)), mode="edge")
    gx = padded[0:-1, 1:] - padded[0:-1, 0:-1]
    gy = padded[1:, 0:-1] - padded[0:-1, 0:-1]
    return np.abs(gx) + np.abs(gy)


def biobridge_hessian(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    padded = np.pad(data, ((1, 1), (1, 1)), mode="edge")
    center = padded[1:-1, 1:-1]
    left = padded[1:-1, 0:-2]
    right = padded[1:-1, 2:]
    up = padded[0:-2, 1:-1]
    down = padded[2:, 1:-1]
    hxx = right - 2 * center + left
    hyy = down - 2 * center + up
    return np.abs(hxx) + np.abs(hyy)


def biobridge_gradient_hessian_operator(frame_rgb: np.ndarray) -> np.ndarray:
    gray = rgb_to_luminance(frame_rgb).astype(np.float32)
    smoothed = edge_padded_smooth_3x3(gray)
    gradient = biobridge_gradient(smoothed)
    hessian = biobridge_hessian(smoothed)
    combined = gradient * hessian
    return normalize_to_uint8(edge_padded_smooth_3x3(combined))


def gradient_hessian_fast(frame_rgb: np.ndarray) -> np.ndarray:
    return biobridge_gradient_hessian_operator(frame_rgb)


def gradient_and_hessian(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    hxx = cv2.Sobel(image, cv2.CV_32F, 2, 0, ksize=3)
    hyy = cv2.Sobel(image, cv2.CV_32F, 0, 2, ksize=3)
    hxy = cv2.Sobel(image, cv2.CV_32F, 1, 1, ksize=3)
    hessian = cv2.sqrt(hxx * hxx + 2.0 * hxy * hxy + hyy * hyy)

    return gradient, hessian


def odd_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def radar_range_doppler_map(frame_rgb: np.ndarray, strength: int) -> np.ndarray:
    gray = rgb_to_luminance(ensure_rgb(frame_rgb)).astype(np.float32)
    gray = gray - float(np.mean(gray))
    window_y = np.hanning(gray.shape[0]).astype(np.float32)[:, None]
    window_x = np.hanning(gray.shape[1]).astype(np.float32)[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(gray * window_y * window_x))
    magnitude = np.log1p(np.abs(spectrum))
    magnitude = cv2.GaussianBlur(magnitude.astype(np.float32), (0, 0), sigmaX=0.5 + strength / 90.0)
    spectrum_gray = normalize_to_uint8(magnitude)
    spectrum_bgr = cv2.applyColorMap(spectrum_gray, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(spectrum_bgr, cv2.COLOR_BGR2RGB)


def lidar_depth_relief(frame_rgb: np.ndarray, strength: int) -> np.ndarray:
    gray = rgb_to_luminance(ensure_rgb(frame_rgb)).astype(np.float32)
    depth = cv2.bilateralFilter(gray, 7, 35 + strength, 35 + strength)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    hillshade = normalize_to_uint8(0.55 * depth - 0.35 * gx - 0.25 * gy)
    relief_bgr = cv2.applyColorMap(hillshade, cv2.COLORMAP_OCEAN)
    return cv2.cvtColor(relief_bgr, cv2.COLOR_BGR2RGB)


def ultrasonic_bscan_envelope(frame_rgb: np.ndarray, strength: int) -> np.ndarray:
    gray = rgb_to_luminance(ensure_rgb(frame_rgb)).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8 + strength / 85.0)
    vertical_echo = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    horizontal_echo = 0.35 * np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    envelope = cv2.GaussianBlur(vertical_echo + horizontal_echo, (0, 0), sigmaX=1.0 + strength / 60.0)
    envelope_gray = normalize_to_uint8(envelope)
    envelope_bgr = cv2.applyColorMap(envelope_gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(envelope_bgr, cv2.COLOR_BGR2RGB)



def process_frame(frame_rgb: np.ndarray, algorithm: str, strength: int, threshold: int) -> np.ndarray:
    gray = rgb_to_luminance(frame_rgb)
    blur_kernel = odd_kernel(3 + 2 * max(0, strength // 35))
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    if algorithm == "Radar Range-Doppler Map":
        return radar_range_doppler_map(frame_rgb, strength)
    if algorithm == "LiDAR Depth Relief":
        return lidar_depth_relief(frame_rgb, strength)
    if algorithm == "Ultrasonic B-Scan Envelope":
        return ultrasonic_bscan_envelope(frame_rgb, strength)
    if algorithm == "Gradient x Hessian":
        return biobridge_gradient_hessian_operator(frame_rgb)
    if algorithm == "Gradient Magnitude":
        gradient, _ = gradient_and_hessian(blurred)
        return normalize_to_uint8(gradient)
    if algorithm == "Hessian Magnitude":
        _, hessian = gradient_and_hessian(blurred)
        return normalize_to_uint8(hessian)
    if algorithm == "CLAHE Contrast":
        clip = 1.0 + strength / 20.0
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if algorithm == "Otsu Threshold":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if algorithm == "Adaptive Threshold":
        block_size = odd_kernel(11 + int(strength / 4))
        c_value = max(1, int(threshold / 20))
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)
    if algorithm == "Canny Edges":
        low = max(1, threshold)
        high = max(low + 1, int(low * (1.5 + strength / 100.0)))
        return cv2.Canny(blurred, low, high)
    if algorithm == "Edge Overlay":
        return edge_overlay(frame_rgb, strength, threshold)
    if algorithm == "Denoise - Gaussian":
        kernel_size = odd_kernel(3 + int(strength / 12))
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    if algorithm == "Denoise - Median":
        kernel_size = odd_kernel(3 + int(strength / 16))
        return cv2.medianBlur(gray, kernel_size)
    if algorithm == "Denoise - Bilateral":
        diameter = odd_kernel(5 + int(strength / 12))
        sigma = 20 + strength * 2
        return cv2.bilateralFilter(gray, diameter, sigma, sigma)
    if algorithm == "Denoise - Non-Local Means":
        h_value = max(3, int(3 + strength / 4))
        return cv2.fastNlMeansDenoising(gray, None, h=h_value, templateWindowSize=7, searchWindowSize=21)
    if algorithm == "Sharpen - Unsharp Mask":
        sigma = 0.8 + strength / 35.0
        amount = 0.4 + strength / 45.0
        soft = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        return cv2.addWeighted(gray, 1.0 + amount, soft, -amount, 0)
    if algorithm == "Sharpen - Laplacian":
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        amount = 0.15 + strength / 140.0
        return np.clip(gray.astype(np.float32) - amount * laplacian, 0, 255).astype(np.uint8)
    if algorithm == "Sharpen - High Boost":
        sigma = 1.0 + strength / 25.0
        soft = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        high_frequency = gray.astype(np.float32) - soft.astype(np.float32)
        amount = 0.8 + strength / 30.0
        return np.clip(gray.astype(np.float32) + amount * high_frequency, 0, 255).astype(np.uint8)
    if algorithm == "Morph Open":
        kernel_size = odd_kernel(3 + int(strength / 25))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    if algorithm == "Morph Close":
        kernel_size = odd_kernel(3 + int(strength / 25))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    if algorithm == "Focus Map":
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        local_energy = cv2.GaussianBlur(laplacian * laplacian, (0, 0), sigmaX=1.0 + strength / 25.0)
        return normalize_to_uint8(local_energy)
    if algorithm == "Background Subtract":
        kernel_size = odd_kernel(21 + int(strength / 2))
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        corrected = cv2.subtract(gray, background)
        return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if algorithm == "Satellite Water Emphasis":
        from gaia.core.satellite import satellite_water_emphasis
        return satellite_water_emphasis(frame_rgb)
    if algorithm == "Satellite Vegetation Proxy":
        from gaia.core.satellite import satellite_vegetation_proxy
        return satellite_vegetation_proxy(frame_rgb)
    return gradient_hessian_fast(frame_rgb)



def edge_overlay(frame_rgb: np.ndarray, strength: int, threshold: int) -> np.ndarray:
    frame = ensure_rgb(frame_rgb)
    gray = rgb_to_luminance(frame)
    low = max(1, threshold)
    high = max(low + 1, int(low * (1.5 + strength / 100.0)))
    edges = cv2.Canny(gray, low, high)
    overlay = frame.copy()
    overlay[edges > 0] = np.array([255, 126, 219], dtype=np.uint8)
    glow_size = odd_kernel(3 + int(strength / 30))
    glow = cv2.GaussianBlur(edges, (glow_size, glow_size), 0)
    glow_rgb = np.zeros_like(frame)
    glow_rgb[:, :, 0] = glow
    glow_rgb[:, :, 2] = glow
    return cv2.addWeighted(overlay, 0.82, glow_rgb, 0.35, 0)


def frame_similarity_heatmap(
    current_rgb: np.ndarray,
    previous_rgb: np.ndarray | None,
    strength: int,
) -> tuple[np.ndarray, float]:
    current = ensure_rgb(current_rgb)
    if previous_rgb is None:
        return np.zeros_like(current), 100.0

    previous = ensure_rgb(previous_rgb)
    if previous.shape != current.shape:
        previous = cv2.resize(previous, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA)

    current_gray = rgb_to_luminance(current)
    previous_gray = rgb_to_luminance(previous)
    diff = cv2.absdiff(current_gray, previous_gray)
    blur_sigma = 1.0 + strength / 30.0
    local_diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=blur_sigma)
    similarity = 255 - local_diff
    similarity_percent = float(np.mean(similarity) * 100.0 / 255.0)

    activation = cv2.applyColorMap(local_diff, cv2.COLORMAP_MAGMA)
    activation_rgb = cv2.cvtColor(activation, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(current, 0.45, activation_rgb, 0.75, 0)
    return blended, similarity_percent


