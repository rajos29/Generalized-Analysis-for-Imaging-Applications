from __future__ import annotations

import cv2
import numpy as np

from gaia.core.image_ops import ensure_rgb, normalize_to_uint8, rgb_to_luminance

def rotate_template_gray(template_gray: np.ndarray, angle_deg: float) -> np.ndarray:
    template = np.asarray(template_gray, dtype=np.uint8)
    if abs(angle_deg) <= 1e-9:
        return template
    height, width = template.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        template,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def rotate_template_mask(template_mask: np.ndarray, angle_deg: float) -> np.ndarray:
    mask = np.asarray(template_mask, dtype=np.uint8)
    if abs(angle_deg) <= 1e-9:
        return (mask > 0).astype(np.uint8) * 255
    height, width = mask.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (rotated > 0).astype(np.uint8) * 255


def template_match_search(
    base_rgb: np.ndarray,
    template_rgb: np.ndarray,
    stride_px: int,
    angle_min: float,
    angle_max: float,
    angle_step: float,
    progress_callback: object | None = None,
    angle_values: object | None = None,
    template_mask: np.ndarray | None = None,
) -> dict[str, object]:
    base_gray = rgb_to_luminance(ensure_rgb(base_rgb))
    template_gray = rgb_to_luminance(ensure_rgb(template_rgb))
    template_height, template_width = template_gray.shape[:2]
    base_height, base_width = base_gray.shape[:2]
    if template_width < 3 or template_height < 3:
        raise ValueError("Search filter is too small. Use at least a 3 x 3 pixel template.")
    if template_width > base_width or template_height > base_height:
        raise ValueError("Search filter is larger than the selected base image.")
    stride_px = max(1, int(stride_px))
    mask_gray: np.ndarray | None = None
    if template_mask is not None:
        candidate_mask = np.asarray(template_mask, dtype=np.uint8)
        if candidate_mask.shape[:2] != template_gray.shape[:2]:
            raise ValueError("Polygon filter mask does not match the search filter size.")
        mask_gray = (candidate_mask > 0).astype(np.uint8) * 255
        if int(np.count_nonzero(mask_gray)) < 9:
            raise ValueError("Polygon filter is too small after masking.")
    if angle_values is not None:
        angles = np.asarray(angle_values, dtype=np.float32)
    else:
        angle_step = max(0.25, abs(float(angle_step)))
        if angle_min > angle_max:
            angle_min, angle_max = angle_max, angle_min
        angles = np.arange(float(angle_min), float(angle_max) + angle_step * 0.5, angle_step)
    if angles.size == 0:
        angles = np.array([0.0], dtype=np.float32)

    best: dict[str, object] | None = None
    best_corr_map: np.ndarray | None = None
    best_diff_map = np.full(
        (base_height - template_height + 1, base_width - template_width + 1),
        np.nan,
        dtype=np.float32,
    )
    base_float = base_gray.astype(np.float32)
    total_steps = int(angles.size) + 1
    for index, angle in enumerate(angles):
        rotated = rotate_template_gray(template_gray, float(angle))
        rotated_mask = rotate_template_mask(mask_gray, float(angle)) if mask_gray is not None else None
        if rotated_mask is not None and int(np.count_nonzero(rotated_mask)) < 9:
            continue
        if rotated_mask is not None:
            corr_map = cv2.matchTemplate(base_gray, rotated, cv2.TM_CCOEFF_NORMED, mask=rotated_mask)
        else:
            corr_map = cv2.matchTemplate(base_gray, rotated, cv2.TM_CCOEFF_NORMED)
        corr_map = np.nan_to_num(corr_map, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
        rotated_float = rotated.astype(np.float32)
        valid = rotated_mask > 0 if rotated_mask is not None else None
        for y in range(0, corr_map.shape[0], stride_px):
            for x in range(0, corr_map.shape[1], stride_px):
                corr_score = float(corr_map[y, x])
                patch = base_float[y:y + template_height, x:x + template_width]
                if valid is not None:
                    difference = float(np.mean(np.abs(patch[valid] - rotated_float[valid])))
                else:
                    difference = float(np.mean(np.abs(patch - rotated_float)))
                better_corr = best is None or corr_score > float(best["corr_score"]) + 1e-6
                same_corr_better_diff = (
                    best is not None
                    and abs(corr_score - float(best["corr_score"])) <= 1e-6
                    and difference < float(best["difference_score"])
                )
                if better_corr or same_corr_better_diff:
                    best = {
                        "x": int(x + template_width / 2),
                        "y": int(y + template_height / 2),
                        "corr_score": corr_score,
                        "difference_score": difference,
                        "rotation_deg": float(angle),
                        "top_left": (int(x), int(y)),
                    }
                    best_corr_map = corr_map
        if progress_callback is not None:
            progress_callback(
                int((index + 1) * 88 / total_steps),
                f"CIS angle {float(angle):0.1f} deg | sweep {index + 1}/{angles.size} | stride {stride_px} px",
            )

    if best is None or best_corr_map is None:
        raise ValueError("Template search did not produce a valid match.")

    best_template = rotate_template_gray(template_gray, float(best["rotation_deg"]))
    best_template_float = best_template.astype(np.float32)
    best_mask = rotate_template_mask(mask_gray, float(best["rotation_deg"])) if mask_gray is not None else None
    for y in range(0, best_diff_map.shape[0], stride_px):
        for x in range(0, best_diff_map.shape[1], stride_px):
            patch = base_float[y:y + template_height, x:x + template_width]
            if best_mask is not None:
                valid = best_mask > 0
                best_diff_map[y, x] = float(np.mean(np.abs(patch[valid] - best_template_float[valid])))
            else:
                best_diff_map[y, x] = float(np.mean(np.abs(patch - best_template_float)))
    if progress_callback is not None:
        progress_callback(92, f"CIS difference map | best angle {float(best['rotation_deg']):0.1f} deg")

    heat_gray = normalize_to_uint8(best_corr_map)
    heat_gray = cv2.resize(heat_gray, (base_width, base_height), interpolation=cv2.INTER_LINEAR)
    heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_gray, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(ensure_rgb(base_rgb), 0.62, heat_rgb, 0.45, 0)
    top_left = best["top_left"]
    x0, y0 = int(top_left[0]), int(top_left[1])
    x1, y1 = x0 + template_width, y0 + template_height
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (244, 210, 122), 2, cv2.LINE_AA)
    edge_source = np.where(best_mask > 0, best_template, 0).astype(np.uint8) if best_mask is not None else best_template
    template_edges = cv2.Canny(edge_source, 40, 120)
    edge_patch = overlay[y0:y1, x0:x1]
    if edge_patch.shape[:2] == template_edges.shape:
        edge_patch[template_edges > 0] = np.array([255, 255, 255], dtype=np.uint8)
        overlay[y0:y1, x0:x1] = edge_patch
    cv2.circle(overlay, (int(best["x"]), int(best["y"])), 4, (244, 210, 122), -1, cv2.LINE_AA)
    if progress_callback is not None:
        progress_callback(100, "CIS complete")
    return {
        "best_vector": [[int(best["x"])], [int(best["y"])], [float(best["corr_score"])]],
        "best_x": int(best["x"]),
        "best_y": int(best["y"]),
        "corr_score": float(best["corr_score"]),
        "difference_score": float(best["difference_score"]),
        "rotation_deg": float(best["rotation_deg"]),
        "template_width": int(template_width),
        "template_height": int(template_height),
        "stride_px": int(stride_px),
        "heatmap": overlay,
        "corr_map": best_corr_map,
        "difference_map": best_diff_map,
    }


