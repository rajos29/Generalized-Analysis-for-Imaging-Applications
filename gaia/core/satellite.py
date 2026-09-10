from __future__ import annotations

import cv2
import numpy as np

from gaia.core.image_ops import biobridge_gradient_hessian_operator, ensure_rgb, normalize_to_uint8, rgb_to_luminance

def resize_like(frame_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    frame = ensure_rgb(frame_rgb)
    reference = ensure_rgb(reference_rgb)
    if frame.shape[:2] == reference.shape[:2]:
        return frame
    return cv2.resize(frame, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def valid_coverage_mask(frame_rgb: np.ndarray) -> np.ndarray:
    frame = ensure_rgb(frame_rgb)
    gray = rgb_to_luminance(frame)
    color_sum = frame.astype(np.int16).sum(axis=2)
    return (gray > 3) & (color_sum > 12)


def satellite_shared_gxh_reference(
    post_rgb: np.ndarray,
    pre_rgb: np.ndarray | None,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    post = ensure_rgb(post_rgb)
    if pre_rgb is None:
        return np.zeros(post.shape[:2], dtype=np.uint8), np.zeros(post.shape[:2], dtype=bool)
    pre = resize_like(pre_rgb, post)
    post_gxh = biobridge_gradient_hessian_operator(post).astype(np.float32)
    pre_gxh = biobridge_gradient_hessian_operator(pre).astype(np.float32)
    shared = np.sqrt(np.nan_to_num(post_gxh * pre_gxh, nan=0.0, posinf=0.0, neginf=0.0))
    finite_shared = shared[np.isfinite(shared)]
    if finite_shared.size:
        high = float(np.percentile(finite_shared, 99.5))
        if high > 0:
            shared = np.clip(shared, 0.0, high)
    shared_gray = normalize_to_uint8(shared)

    valid = valid_coverage_mask(post) & valid_coverage_mask(pre)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    support = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0

    display = cv2.cvtColor(shared_gray, cv2.COLOR_GRAY2RGB)
    display[~support] = np.array([0, 0, 0], dtype=np.uint8)
    return display, support


def transform_frame_affine(
    frame_rgb: np.ndarray,
    target_shape: tuple[int, int],
    x_shift: float,
    y_shift: float,
    rotation_deg: float,
    scale_percent: float,
) -> np.ndarray:
    frame = ensure_rgb(frame_rgb)
    target_height, target_width = target_shape
    if frame.shape[:2] != target_shape:
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    center = (target_width / 2.0, target_height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, scale_percent / 100.0)
    matrix[0, 2] += x_shift
    matrix[1, 2] += y_shift
    return cv2.warpAffine(
        frame,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def transform_frame_projective(
    frame_rgb: np.ndarray,
    target_shape: tuple[int, int],
    homography: np.ndarray | None,
) -> np.ndarray:
    frame = ensure_rgb(frame_rgb)
    target_height, target_width = target_shape
    if frame.shape[:2] != target_shape:
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if homography is None:
        return frame
    return cv2.warpPerspective(
        frame,
        homography,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def affine_similarity_score(post_gray: np.ndarray, pre_gray: np.ndarray, valid: np.ndarray) -> float:
    if not valid.any():
        return -1.0
    post_values = post_gray[valid].astype(np.float32)
    pre_values = pre_gray[valid].astype(np.float32)
    post_values -= float(post_values.mean())
    pre_values -= float(pre_values.mean())
    denominator = float(np.linalg.norm(post_values) * np.linalg.norm(pre_values))
    if denominator <= 1e-6:
        return -1.0
    return float(np.dot(post_values, pre_values) / denominator)


def scale_homography_to_full_resolution(h_small: np.ndarray, scale_factor: float) -> np.ndarray:
    small_to_full = np.array(
        [[1.0 / scale_factor, 0.0, 0.0], [0.0, 1.0 / scale_factor, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    full_to_small = np.array(
        [[scale_factor, 0.0, 0.0], [0.0, scale_factor, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return small_to_full @ h_small @ full_to_small


def estimate_satellite_projective_alignment(
    post_rgb: np.ndarray,
    pre_rgb: np.ndarray,
) -> dict[str, object]:
    post = ensure_rgb(post_rgb)
    pre = resize_like(pre_rgb, post)
    target_width = 720
    scale_factor = target_width / post.shape[1]
    target_height = max(1, int(post.shape[0] * scale_factor))
    post_small = cv2.resize(post, (target_width, target_height), interpolation=cv2.INTER_AREA)
    pre_small = cv2.resize(pre, (target_width, target_height), interpolation=cv2.INTER_AREA)

    post_gxh = biobridge_gradient_hessian_operator(post_small)
    pre_gxh = biobridge_gradient_hessian_operator(pre_small)
    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8, edgeThreshold=19, patchSize=31)
    post_keypoints, post_descriptors = orb.detectAndCompute(post_gxh, None)
    pre_keypoints, pre_descriptors = orb.detectAndCompute(pre_gxh, None)
    if post_descriptors is None or pre_descriptors is None or len(post_keypoints) < 12 or len(pre_keypoints) < 12:
        raise ValueError("Not enough shared structural features for projective Auto Tilt.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(pre_descriptors, post_descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.78 * second.distance:
            good_matches.append(first)
    if len(good_matches) < 10:
        raise ValueError("Not enough reliable feature matches for projective Auto Tilt.")

    pre_points = np.float32([pre_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    post_points = np.float32([post_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    h_small, inlier_mask = cv2.findHomography(pre_points, post_points, cv2.RANSAC, 4.0)
    if h_small is None or inlier_mask is None:
        raise ValueError("Projective Auto Tilt could not solve a stable homography.")
    inliers = int(inlier_mask.ravel().sum())
    if inliers < 8:
        raise ValueError("Projective Auto Tilt found too few inlier matches.")

    homography = scale_homography_to_full_resolution(h_small, scale_factor)
    transformed = transform_frame_projective(pre, post.shape[:2], homography)
    post_gxh_full = biobridge_gradient_hessian_operator(post)
    pre_gxh_full = biobridge_gradient_hessian_operator(transformed)
    valid = valid_coverage_mask(post) & valid_coverage_mask(transformed)
    score = affine_similarity_score(post_gxh_full, pre_gxh_full, valid)
    return {
        "homography": homography,
        "matches": len(good_matches),
        "inliers": inliers,
        "score": score,
    }


def phase_translation(reference_gray: np.ndarray, moving_gray: np.ndarray) -> tuple[float, float]:
    reference = reference_gray.astype(np.float32)
    moving = moving_gray.astype(np.float32)
    reference = cv2.GaussianBlur(reference, (0, 0), sigmaX=1.0)
    moving = cv2.GaussianBlur(moving, (0, 0), sigmaX=1.0)
    shift, response = cv2.phaseCorrelate(reference, moving)
    if response < 0.02:
        return 0.0, 0.0
    x_shift, y_shift = shift
    return float(x_shift), float(y_shift)


def estimate_satellite_affine_alignment(
    post_rgb: np.ndarray,
    pre_rgb: np.ndarray,
) -> dict[str, float]:
    post = ensure_rgb(post_rgb)
    pre = resize_like(pre_rgb, post)
    target_width = 360
    scale_factor = target_width / post.shape[1]
    target_height = max(1, int(post.shape[0] * scale_factor))
    post_small = cv2.resize(post, (target_width, target_height), interpolation=cv2.INTER_AREA)
    pre_small = cv2.resize(pre, (target_width, target_height), interpolation=cv2.INTER_AREA)
    post_gxh = biobridge_gradient_hessian_operator(post_small)
    post_valid = valid_coverage_mask(post_small)

    best = {"score": -1.0, "x": 0.0, "y": 0.0, "rotation": 0.0, "scale": 100.0}
    candidates: list[tuple[float, float]] = []
    for rotation in (-4.0, -2.0, 0.0, 2.0, 4.0):
        for scale in (97.0, 100.0, 103.0):
            candidates.append((rotation, scale))
    for rotation in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for scale in (99.0, 100.0, 101.0):
            candidates.append((rotation, scale))

    for rotation, scale in candidates:
        transformed_no_shift = transform_frame_affine(pre_small, post_small.shape[:2], 0.0, 0.0, rotation, scale)
        pre_gxh_no_shift = biobridge_gradient_hessian_operator(transformed_no_shift)
        x_shift, y_shift = phase_translation(post_gxh, pre_gxh_no_shift)
        for x_delta in (-4.0, 0.0, 4.0):
            for y_delta in (-4.0, 0.0, 4.0):
                trial_x = x_shift + x_delta
                trial_y = y_shift + y_delta
                transformed = transform_frame_affine(pre_small, post_small.shape[:2], trial_x, trial_y, rotation, scale)
                pre_gxh = biobridge_gradient_hessian_operator(transformed)
                valid = post_valid & valid_coverage_mask(transformed)
                score = affine_similarity_score(post_gxh, pre_gxh, valid)
                if score > best["score"]:
                    best = {"score": score, "x": trial_x, "y": trial_y, "rotation": rotation, "scale": scale}

    return {
        "x": best["x"] / scale_factor,
        "y": best["y"] / scale_factor,
        "rotation": best["rotation"],
        "scale": best["scale"],
        "score": best["score"],
    }


def satellite_change_diff(
    post_rgb: np.ndarray,
    pre_rgb: np.ndarray | None,
    blur_strength: int,
    threshold: int,
    support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    post = ensure_rgb(post_rgb)
    if pre_rgb is None:
        return np.zeros(post.shape[:2], dtype=np.float32), None, post, None
    pre = resize_like(pre_rgb, post)
    post_float = post.astype(np.float32)
    pre_float = pre.astype(np.float32)
    diff = np.nan_to_num(np.mean(np.abs(post_float - pre_float), axis=2), nan=0.0, posinf=0.0, neginf=0.0)
    if support_mask is not None and support_mask.shape == post.shape[:2]:
        support = support_mask.astype(bool)
    else:
        _shared_display, support = satellite_shared_gxh_reference(post, pre, threshold)
    diff = np.where(support, diff, 0.0)
    sigma = 0.4 + blur_strength / 45.0
    if sigma > 0:
        diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=sigma)
        diff = np.where(support, diff, 0.0)
    return diff, pre, post, support


def satellite_change_heatmap(
    post_rgb: np.ndarray,
    pre_rgb: np.ndarray | None,
    strength: int,
    threshold: int,
    support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    diff, _pre, post, support = satellite_change_diff(post_rgb, pre_rgb, strength, threshold, support_mask)
    if pre_rgb is None:
        return np.zeros_like(post), 100.0
    diff_gray = normalize_to_uint8(diff)
    heat_bgr = cv2.applyColorMap(diff_gray, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    if support is not None:
        heat_rgb[~support] = np.array([20, 18, 28], dtype=np.uint8)
    blended = cv2.addWeighted(post, 0.42, heat_rgb, 0.78, 0)
    if support is not None:
        blended[~support] = np.array([0, 0, 0], dtype=np.uint8)
    if support is not None and support.any():
        similarity = float(100.0 - min(100.0, np.mean(diff[support]) * 100.0 / 255.0))
    else:
        similarity = 100.0
    return blended, similarity


def satellite_change_mask(
    post_rgb: np.ndarray,
    pre_rgb: np.ndarray | None,
    strength: int,
    threshold: int,
    support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    diff, _pre, post, support = satellite_change_diff(post_rgb, pre_rgb, strength, threshold, support_mask)
    if pre_rgb is None:
        return np.zeros_like(post), 100.0
    mask = diff >= float(threshold)
    overlay = post.copy()
    if support is not None:
        overlay[~support] = np.array([20, 18, 28], dtype=np.uint8)
    overlay[mask] = np.array([255, 214, 92], dtype=np.uint8)
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 30, 120)
    overlay[edges > 0] = np.array([255, 80, 180], dtype=np.uint8)
    if support is not None and support.any():
        active = float(np.mean(mask[support]) * 100.0)
    else:
        active = 0.0
    return overlay, 100.0 - active


def satellite_water_score(frame_rgb: np.ndarray) -> np.ndarray:
    frame = ensure_rgb(frame_rgb).astype(np.float32)
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    score = (blue + green) * 0.5 - red
    return normalize_to_uint8(score)


def satellite_water_emphasis(frame_rgb: np.ndarray) -> np.ndarray:
    water = satellite_water_score(frame_rgb)
    water_bgr = cv2.applyColorMap(water, cv2.COLORMAP_OCEAN)
    water_rgb = cv2.cvtColor(water_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(ensure_rgb(frame_rgb), 0.45, water_rgb, 0.75, 0)


def satellite_vegetation_proxy(frame_rgb: np.ndarray) -> np.ndarray:
    frame = ensure_rgb(frame_rgb).astype(np.float32)
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    proxy = green - 0.5 * (red + blue)
    proxy_gray = normalize_to_uint8(proxy)
    proxy_bgr = cv2.applyColorMap(proxy_gray, cv2.COLORMAP_SUMMER)
    return cv2.cvtColor(proxy_bgr, cv2.COLOR_BGR2RGB)


def satellite_change_metrics(post_rgb: np.ndarray | None, pre_rgb: np.ndarray | None, threshold: int) -> dict[str, float | str]:
    if post_rgb is None or pre_rgb is None:
        return {
            "change_mean_delta": "",
            "change_p95_delta": "",
            "change_active_percent": "",
            "water_pre_percent": "",
            "water_post_percent": "",
            "water_change_percent": "",
            "green_pre_mean": "",
            "green_post_mean": "",
            "green_change_mean": "",
            "shared_gxh_overlap_percent": "",
            "excluded_area_percent": "",
            "valid_post_area_percent": "",
            "valid_pre_area_percent": "",
            "structural_similarity_score": "",
            "change_median_delta": "",
            "change_p90_delta": "",
            "change_max_delta": "",
            "change_component_count": "",
            "filtered_change_region_count": "",
            "filtered_change_region_area_percent": "",
            "largest_change_region_area_px": "",
            "largest_change_component_percent": "",
            "mean_change_region_area_px": "",
            "median_change_region_area_px": "",
            "change_region_density_per_megapixel": "",
            "change_severity_index": "",
        }
    post = ensure_rgb(post_rgb)
    pre = resize_like(pre_rgb, post)
    _shared_display, support = satellite_shared_gxh_reference(post, pre, threshold)
    diff = np.mean(np.abs(post.astype(np.float32) - pre.astype(np.float32)), axis=2)
    active = (diff >= 45.0) & support
    valid_post = valid_coverage_mask(post)
    valid_pre = valid_coverage_mask(pre)
    post_gxh = biobridge_gradient_hessian_operator(post)
    pre_gxh = biobridge_gradient_hessian_operator(pre)
    structural_score = affine_similarity_score(post_gxh, pre_gxh, support)
    active_uint8 = active.astype(np.uint8)
    component_count, component_labels, component_stats, _centroids = cv2.connectedComponentsWithStats(active_uint8, 8)
    component_areas = component_stats[1:, cv2.CC_STAT_AREA].astype(np.float32) if component_count > 1 else np.array([], dtype=np.float32)
    support_area = max(1.0, float(np.count_nonzero(support)))
    largest_component_percent = float(component_areas.max() * 100.0 / support_area) if component_areas.size else 0.0
    min_region_area_px = max(80.0, support_area * 0.00002)
    filtered_component_areas = component_areas[component_areas >= min_region_area_px]
    filtered_region_count = int(filtered_component_areas.size)
    filtered_region_area = float(filtered_component_areas.sum()) if filtered_component_areas.size else 0.0
    filtered_region_area_percent = filtered_region_area * 100.0 / support_area
    largest_region_area_px = float(filtered_component_areas.max()) if filtered_component_areas.size else 0.0
    mean_region_area_px = float(filtered_component_areas.mean()) if filtered_component_areas.size else 0.0
    median_region_area_px = float(np.median(filtered_component_areas)) if filtered_component_areas.size else 0.0
    support_megapixels = support_area / 1_000_000.0
    region_density = float(filtered_region_count / support_megapixels) if support_megapixels > 0 else 0.0

    pre_water = satellite_water_score(pre) >= 120
    post_water = satellite_water_score(post) >= 120
    pre_green = pre[:, :, 1].astype(np.float32) - 0.5 * (pre[:, :, 0].astype(np.float32) + pre[:, :, 2].astype(np.float32))
    post_green = post[:, :, 1].astype(np.float32) - 0.5 * (post[:, :, 0].astype(np.float32) + post[:, :, 2].astype(np.float32))

    return {
        "change_mean_delta": float(np.mean(diff[support])) if support.any() else 0.0,
        "change_p95_delta": float(np.percentile(diff[support], 95)) if support.any() else 0.0,
        "change_active_percent": float(np.mean(active[support]) * 100.0) if support.any() else 0.0,
        "water_pre_percent": float(np.mean(pre_water[support]) * 100.0) if support.any() else 0.0,
        "water_post_percent": float(np.mean(post_water[support]) * 100.0) if support.any() else 0.0,
        "water_change_percent": float((np.mean(post_water[support]) - np.mean(pre_water[support])) * 100.0) if support.any() else 0.0,
        "green_pre_mean": float(np.mean(pre_green[support])) if support.any() else 0.0,
        "green_post_mean": float(np.mean(post_green[support])) if support.any() else 0.0,
        "green_change_mean": float(np.mean((post_green - pre_green)[support])) if support.any() else 0.0,
        "shared_gxh_overlap_percent": float(np.mean(support) * 100.0),
        "excluded_area_percent": float((1.0 - np.mean(support)) * 100.0),
        "valid_post_area_percent": float(np.mean(valid_post) * 100.0),
        "valid_pre_area_percent": float(np.mean(valid_pre) * 100.0),
        "structural_similarity_score": structural_score,
        "change_median_delta": float(np.median(diff[support])) if support.any() else 0.0,
        "change_p90_delta": float(np.percentile(diff[support], 90)) if support.any() else 0.0,
        "change_max_delta": float(np.max(diff[support])) if support.any() else 0.0,
        "change_component_count": int(component_areas.size),
        "filtered_change_region_count": filtered_region_count,
        "filtered_change_region_area_percent": filtered_region_area_percent,
        "largest_change_region_area_px": largest_region_area_px,
        "largest_change_component_percent": largest_component_percent,
        "mean_change_region_area_px": mean_region_area_px,
        "median_change_region_area_px": median_region_area_px,
        "change_region_density_per_megapixel": region_density,
        "change_severity_index": float(np.mean(active[support]) * np.percentile(diff[support], 95)) if support.any() else 0.0,
    }


