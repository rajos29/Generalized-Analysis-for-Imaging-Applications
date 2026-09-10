from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microscope_fast_qt_platform import (
    DEFAULT_IDRT_DATASET_DIR,
    IDRT_PAIRS,
    load_image,
    satellite_change_heatmap,
    satellite_change_mask,
    satellite_change_metrics,
    satellite_shared_gxh_reference,
)


REPORT_ROOT = Path("reports") / "idrt_satellite_analysis"
FIGURES_DIR = REPORT_ROOT / "figures"
VECTORS_DIR = REPORT_ROOT / "vectors"
THRESHOLD = 80
STRENGTH = 50
CLEAN_METRIC_COLUMNS = [
    "name",
    "shared_gxh_overlap_percent",
    "excluded_area_percent",
    "structural_similarity_score",
    "change_mean_delta",
    "change_median_delta",
    "change_p90_delta",
    "change_p95_delta",
    "change_active_percent",
    "filtered_change_region_count",
    "filtered_change_region_area_percent",
    "largest_change_region_area_px",
    "change_region_density_per_megapixel",
    "change_severity_index",
    "water_change_percent",
    "green_change_mean",
    "vector_feature_count",
    "analyst_use_grade",
]


def safe_name(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def resize_for_sheet(image_rgb: np.ndarray, width: int = 640) -> np.ndarray:
    scale = width / image_rgb.shape[1]
    height = max(1, int(image_rgb.shape[0] * scale))
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def caption_bar(width: int, text: str) -> np.ndarray:
    image = Image.new("RGB", (width, 48), (25, 31, 38))
    draw = ImageDraw.Draw(image)
    draw.text((16, 13), text, fill=(242, 246, 249), font=font(20, bold=True))
    return np.asarray(image)


def make_panel(title: str, image_rgb: np.ndarray, width: int = 640) -> np.ndarray:
    resized = resize_for_sheet(image_rgb, width)
    return np.vstack([caption_bar(width, title), resized])


def make_comparison_sheet(label: str, pre: np.ndarray, post: np.ndarray, overlap: np.ndarray, heat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    panels = [
        make_panel("Pre / reference", pre),
        make_panel("Post / current", post),
        make_panel("Shared GxH overlap", overlap),
        make_panel("Candidate change heatmap", heat),
        make_panel("Candidate change mask", mask),
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            pad[:, :] = np.array([18, 18, 24], dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    top = np.hstack(padded[:2])
    bottom = np.hstack(padded[2:])
    if bottom.shape[1] > top.shape[1]:
        pad = np.zeros((top.shape[0], bottom.shape[1] - top.shape[1], 3), dtype=np.uint8)
        pad[:, :] = np.array([18, 18, 24], dtype=np.uint8)
        top = np.hstack([top, pad])
    else:
        pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), dtype=np.uint8)
        pad[:, :] = np.array([18, 18, 24], dtype=np.uint8)
        bottom = np.hstack([bottom, pad])
    title = caption_bar(top.shape[1], label)
    return np.vstack([title, top, bottom])


def draw_vector_overlay(post: np.ndarray, features: list[dict[str, object]]) -> np.ndarray:
    overlay = post.copy()
    fill = overlay.copy()
    for feature in features:
        coords = np.array(feature["geometry"]["coordinates"][0], dtype=np.int32)  # type: ignore[index]
        cv2.fillPoly(fill, [coords], (255, 214, 92))
        cv2.polylines(overlay, [coords], True, (255, 80, 180), 3, cv2.LINE_AA)
    overlay = cv2.addWeighted(overlay, 0.78, fill, 0.22, 0)
    for feature in features[:12]:
        coords = np.array(feature["geometry"]["coordinates"][0], dtype=np.int32)  # type: ignore[index]
        cv2.polylines(overlay, [coords], True, (255, 80, 180), 2, cv2.LINE_AA)
    return overlay


def change_mask_from_pair(post: np.ndarray, pre: np.ndarray, support: np.ndarray) -> np.ndarray:
    pre = cv2.resize(pre, (post.shape[1], post.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.mean(np.abs(post.astype(np.float32) - pre.astype(np.float32)), axis=2)
    return (diff >= THRESHOLD) & support


def vectorize_mask(mask: np.ndarray) -> list[dict[str, object]]:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    contours, _hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features: list[dict[str, object]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 80:
            continue
        epsilon = 0.006 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2).astype(float).tolist()
        if len(points) < 3:
            continue
        points.append(points[0])
        x, y, w, h = cv2.boundingRect(approx)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "area_px": area,
                    "bbox_px": [int(x), int(y), int(w), int(h)],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [points],
                },
            }
        )
    features.sort(key=lambda item: item["properties"]["area_px"], reverse=True)  # type: ignore[index]
    return features[:60]


def save_geojson(path: Path, features: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "coordinate_system": "pixel coordinates, origin at upper-left",
        "features": features,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_svg_overlay(path: Path, width: int, height: int, features: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="none"/>',
    ]
    for feature in features:
        coords = feature["geometry"]["coordinates"][0]  # type: ignore[index]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(f'<polygon points="{points}" fill="rgba(255,214,92,0.22)" stroke="#ff50b4" stroke-width="2"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the IDRT satellite preview-image analysis report.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_IDRT_DATASET_DIR,
        help="Folder containing the external IDRT image files. Defaults to IDRT_DATASET_DIR or data/idrt_application_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_ROOT,
        help="Folder where generated reports, figures, metrics, and vectors are written.",
    )
    return parser.parse_args()


def metric_text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:0.2f}"
    return str(value)


def analyst_grade(row: dict[str, object]) -> str:
    overlap = float(row["shared_gxh_overlap_percent"])
    structural = float(row["structural_similarity_score"])
    active = float(row["change_active_percent"])
    if overlap >= 90.0 and structural >= 0.78 and active < 20.0:
        return "Strong comparison"
    if overlap >= 80.0 and structural >= 0.65:
        return "Usable with caution"
    return "High caution"


def build_reports(rows: list[dict[str, object]], report_root: Path) -> None:
    md_lines = [
        "# Satellite Image Comparison Report",
        "",
        "Prepared as a compact image-analytics demonstration for the Texas A&M IDRT application dataset.",
        "",
        "## Method Summary",
        "",
        "This analysis used a Python/OpenCV image-processing application originally built for microscopy workflows and adapted for satellite preview-image comparison. The tool loads paired images, supports manual and prompted automatic affine alignment, computes a shared valid-coverage support layer, generates candidate change maps, extracts vectorized change footprints, and saves reproducible figures and metrics.",
        "",
        "The shared support layer uses Gradient x Hessian structure from both images: `sqrt(GxH_pre * GxH_post)`. In this report it is used primarily to identify seams, crop bands, and non-overlap regions so the change map is not calculated in closed regions with no valid counterpart.",
        "",
        "## Key Limitations",
        "",
        "- These are preview images, not georeferenced analysis-ready rasters.",
        "- Pixel changes can reflect crop, sensor, lighting, clouds, shadows, tide, viewing geometry, and registration differences.",
        "- Vectorized polygons are candidate visual-change footprints in pixel coordinates, not GIS damage polygons.",
        "- Results should be interpreted as first-pass screening evidence.",
        "",
        "## Results",
        "",
    ]

    html_rows = []
    for row in rows:
        name = str(row["name"])
        md_lines.extend(
            [
                f"### {name}",
                "",
                f"![{name} comparison]({row['comparison_figure']})",
                "",
                f"- Shared valid overlap: {metric_text(row['shared_gxh_overlap_percent'])}%",
                f"- Mean supported RGB delta: {metric_text(row['change_mean_delta'])}",
                f"- Median / P90 / max supported RGB delta: {metric_text(row['change_median_delta'])} / {metric_text(row['change_p90_delta'])} / {metric_text(row['change_max_delta'])}",
                f"- 95th percentile supported RGB delta: {metric_text(row['change_p95_delta'])}",
                f"- Active candidate-change area: {metric_text(row['change_active_percent'])}%",
                f"- Filtered candidate-change regions: {row['filtered_change_region_count']}; filtered region area: {metric_text(row['filtered_change_region_area_percent'])}% of supported area",
                f"- Largest candidate region: {metric_text(row['largest_change_region_area_px'])} px; region density: {metric_text(row['change_region_density_per_megapixel'])} per supported megapixel",
                f"- Change severity index: {metric_text(row['change_severity_index'])}",
                f"- GxH structural similarity score: {metric_text(row['structural_similarity_score'])}",
                f"- Water proxy shift: {metric_text(row['water_change_percent'])} percentage points",
                f"- Green/vegetation proxy shift: {metric_text(row['green_change_mean'])}",
                f"- Vectorized candidate footprints: {row['vector_feature_count']}",
                f"- Excluded crop/non-overlap area: {metric_text(row['excluded_area_percent'])}%",
                "",
            ]
        )
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metric_text(row['shared_gxh_overlap_percent'])}%</td>"
            f"<td>{metric_text(row['change_mean_delta'])}</td>"
            f"<td>{metric_text(row['change_p95_delta'])}</td>"
            f"<td>{metric_text(row['change_active_percent'])}%</td>"
            f"<td>{row['filtered_change_region_count']}</td>"
            f"<td>{metric_text(row['filtered_change_region_area_percent'])}%</td>"
            f"<td>{metric_text(row['change_severity_index'])}</td>"
            f"<td>{analyst_grade(row)}</td>"
            "</tr>"
        )

    md_lines.extend(
        [
            "## Metric Dictionary",
            "",
            "- Shared overlap: percent of the image pair with valid comparable pixels after crop/no-data exclusion.",
            "- Structural similarity: correlation between pre/post GxH structure maps inside the shared support; higher means the comparison is better registered structurally.",
            "- Mean/median/P90/P95/max delta: RGB pixel-change magnitudes inside the shared support.",
            "- Active change area: percent of supported pixels above the screening threshold.",
            "- Filtered regions: connected candidate-change regions after removing small speckles.",
            "- Filtered region area: total area of filtered regions as a percent of supported pixels.",
            "- Region density: filtered candidate regions per supported megapixel.",
            "- Severity index: active-area fraction multiplied by P95 delta, a compact ranking signal rather than a damage probability.",
            "- Water/green shifts: rough RGB proxy changes for rapid environmental screening.",
            "",
            "## How The Tool Was Used",
            "",
            "I used my image-processing platform as a first-pass analytic workstation. The project began as a microscopy tool, but its core pipeline is image-array based: load imagery, run enhancement/structure operators, compare paired frames, calculate metrics, and save reproducible outputs. For this dataset I added satellite-specific pair loading, manual and prompted automatic affine alignment controls, shared GxH overlap support, candidate change heatmaps/masks, and vectorized pixel-coordinate change footprints.",
            "",
            "## Recommended Next Improvements",
            "",
            "- Add automated feature registration using roads, shorelines, and dock structures.",
            "- Add geospatial metadata support for true map coordinates.",
            "- Add cloud/shadow masking for Sentinel-2.",
            "- Add model-based similarity embeddings such as DINOv2 or CLIP for semantic comparison.",
            "- Add analyst annotation tools for labeled damage, flooding, shoreline, debris, and uncertainty zones.",
            "",
        ]
    )
    (report_root / "satellite_image_comparison_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Satellite Image Comparison Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #17202a; line-height: 1.42; }}
    h1, h2, h3 {{ color: #13293d; }}
    .note {{ background: #f3f6f8; border-left: 4px solid #2a6f97; padding: 12px 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #c9d3dc; padding: 8px 10px; text-align: left; }}
    th {{ background: #edf3f7; }}
    img {{ width: 100%; border: 1px solid #d5dde5; margin: 10px 0 24px; }}
    code {{ background: #eef2f5; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Satellite Image Comparison Report</h1>
  <p class="note">Prepared as a compact image-analytics demonstration using a Python/OpenCV platform originally built for microscopy image processing and adapted for satellite preview-image comparison.</p>
  <h2>Method Summary</h2>
  <p>The tool loads paired imagery, supports manual and prompted automatic affine alignment when needed, computes a shared valid-coverage support layer, generates candidate change maps, vectorizes candidate-change footprints, and saves reproducible figures and metrics.</p>
  <p>The shared support layer uses <code>sqrt(GxH_pre * GxH_post)</code> as a seam and structure diagnostic. Change heatmaps are not treated as definitive damage maps; they are first-pass screening layers constrained away from crop and non-overlap regions.</p>
  <h2>Summary Metrics</h2>
  <table>
    <tr><th>Pair</th><th>Shared overlap</th><th>Mean delta</th><th>P95 delta</th><th>Active change</th><th>Regions</th><th>Region area</th><th>Severity</th><th>Use</th></tr>
    {''.join(html_rows)}
  </table>
  <h2>Metric Dictionary</h2>
  <ul>
    <li><b>Shared overlap:</b> percent of the image pair with valid comparable pixels after crop/no-data exclusion.</li>
    <li><b>Structural similarity:</b> correlation between pre/post GxH structure maps inside the shared support; higher means the comparison is better registered structurally.</li>
    <li><b>Delta metrics:</b> RGB pixel-change magnitudes inside the shared support.</li>
    <li><b>Filtered regions:</b> connected candidate-change regions after removing small speckles.</li>
    <li><b>Severity index:</b> active-area fraction multiplied by P95 delta; useful as a compact ranking signal, not a damage probability.</li>
  </ul>
"""
    for row in rows:
        html_doc += f"<h2>{html.escape(str(row['name']))}</h2>\n"
        html_doc += f"<img src=\"{html.escape(str(row['comparison_figure']))}\" alt=\"{html.escape(str(row['name']))} comparison\">\n"
    html_doc += """
  <h2>Limitations</h2>
  <ul>
    <li>Preview images are not georeferenced analysis-ready rasters.</li>
    <li>Pixel differences can reflect crop, sensor, lighting, clouds, shadows, tide, viewing geometry, and registration differences.</li>
    <li>Vectorized polygons are candidate visual-change footprints in pixel coordinates, not operational GIS damage polygons.</li>
  </ul>
  <h2>How The Tool Was Used</h2>
  <p>I used my image-processing platform as a first-pass analytic workstation. The project began as a microscopy tool, but its core pipeline is image-array based: load imagery, run enhancement/structure operators, compare paired frames, calculate metrics, and save reproducible outputs. For this dataset I added satellite-specific pair loading, manual and prompted automatic affine alignment controls, shared GxH overlap support, candidate change heatmaps/masks, and vectorized pixel-coordinate change footprints.</p>
</body>
</html>
"""
    (report_root / "satellite_image_comparison_report.html").write_text(html_doc, encoding="utf-8")
    build_pdf_report(rows, report_root)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=text_font)
        if bbox[2] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def add_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, text_font: ImageFont.ImageFont, fill=(38, 48, 58), line_gap: int = 6) -> int:
    for line in wrap_text(draw, text, text_font, width):
        draw.text((x, y), line, fill=fill, font=text_font)
        y += text_font.size + line_gap if hasattr(text_font, "size") else 20
    return y


def image_for_pdf(path: Path, max_w: int, max_h: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return image


def build_pdf_report(rows: list[dict[str, object]], report_root: Path) -> None:
    pages: list[Image.Image] = []
    page_w, page_h = 1650, 2200
    margin = 110
    title_font = font(48, bold=True)
    h2_font = font(30, bold=True)
    body_font = font(24)
    small_font = font(20)

    cover = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(cover)
    draw.rectangle((0, 0, page_w, 170), fill=(19, 41, 61))
    draw.text((margin, 54), "Satellite Image Comparison Report", fill="white", font=title_font)
    y = 230
    y = add_wrapped(
        draw,
        "Prepared as an image-analytics demonstration for the Texas A&M IDRT application dataset. The workflow uses a Python/OpenCV platform originally built for microscopy image processing and adapted for paired satellite preview analysis.",
        margin,
        y,
        page_w - 2 * margin,
        body_font,
    )
    y += 45
    draw.text((margin, y), "Methods Used", fill=(19, 41, 61), font=h2_font)
    y += 48
    for item in [
        "Manual/visual pre-post comparison.",
        "Prompted affine alignment with low-resolution GxH maps, a small rotation/scale sweep, and phase-correlation translation.",
        "Shared GxH overlap support: sqrt(GxH_pre * GxH_post), used to identify seams, crop bands, and non-overlap regions.",
        "Candidate RGB change heatmaps and thresholded change masks.",
        "Vectorized candidate-change footprints exported as SVG and GeoJSON in pixel coordinates.",
        "Water and green-channel proxy metrics for rapid screening.",
    ]:
        y = add_wrapped(draw, f"- {item}", margin + 20, y, page_w - 2 * margin - 20, body_font)
        y += 8
    y += 40
    draw.text((margin, y), "Summary Metrics", fill=(19, 41, 61), font=h2_font)
    y += 56
    headers = ["Pair", "Overlap", "Struct.", "P95 Delta", "Active", "Regions", "Use"]
    widths = [520, 130, 120, 145, 125, 120, 240]
    x = margin
    for header, width in zip(headers, widths):
        draw.rectangle((x, y, x + width, y + 52), fill=(235, 241, 246), outline=(190, 203, 214))
        draw.text((x + 10, y + 14), header, fill=(19, 41, 61), font=small_font)
        x += width
    y += 52
    for row in rows:
        x = margin
        values = [
            str(row["name"]),
            f"{metric_text(row['shared_gxh_overlap_percent'])}%",
            metric_text(row["structural_similarity_score"]),
            metric_text(row["change_p95_delta"]),
            f"{metric_text(row['change_active_percent'])}%",
            str(row["filtered_change_region_count"]),
            analyst_grade(row),
        ]
        for index, (value, width) in enumerate(zip(values, widths)):
            draw.rectangle((x, y, x + width, y + 88), fill="white", outline=(210, 219, 226))
            if index == 0:
                yy = y + 10
                for line in wrap_text(draw, value, small_font, width - 18)[:2]:
                    draw.text((x + 10, yy), line, fill=(38, 48, 58), font=small_font)
                    yy += 25
            else:
                draw.text((x + 10, y + 15), value, fill=(38, 48, 58), font=small_font)
            x += width
        y += 88
    y += 60
    draw.text((margin, y), "Interpretation Guardrails", fill=(19, 41, 61), font=h2_font)
    y += 48
    y = add_wrapped(
        draw,
        "Outputs are first-pass screening evidence. Preview images are not georeferenced analysis-ready rasters, and pixel differences can reflect sensor, crop, illumination, tide, cloud, shadow, viewing geometry, and registration effects.",
        margin,
        y,
        page_w - 2 * margin,
        body_font,
    )
    pages.append(cover)

    for row in rows:
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        draw.rectangle((0, 0, page_w, 130), fill=(19, 41, 61))
        draw.text((margin, 42), str(row["name"]), fill="white", font=h2_font)
        y = 170
        figure = image_for_pdf(report_root / str(row["comparison_figure"]), page_w - 2 * margin, 1220)
        page.paste(figure, (margin, y))
        y += figure.height + 45
        bullets = [
            f"Shared valid overlap: {metric_text(row['shared_gxh_overlap_percent'])}%. Excluded crop/non-overlap area: {metric_text(row['excluded_area_percent'])}%.",
            f"Mean supported RGB delta: {metric_text(row['change_mean_delta'])}; median/P90/P95/max: {metric_text(row['change_median_delta'])}/{metric_text(row['change_p90_delta'])}/{metric_text(row['change_p95_delta'])}/{metric_text(row['change_max_delta'])}.",
            f"Active candidate-change area: {metric_text(row['change_active_percent'])}%; filtered candidate regions: {row['filtered_change_region_count']}; filtered region area: {metric_text(row['filtered_change_region_area_percent'])}% of supported area.",
            f"Largest candidate region: {metric_text(row['largest_change_region_area_px'])} px; region density: {metric_text(row['change_region_density_per_megapixel'])} per supported megapixel; severity index: {metric_text(row['change_severity_index'])}.",
            f"GxH structural similarity score: {metric_text(row['structural_similarity_score'])}.",
            f"Vectorized candidate-change footprints: {row['vector_feature_count']} features saved as SVG and GeoJSON.",
        ]
        for bullet in bullets:
            y = add_wrapped(draw, f"- {bullet}", margin, y, page_w - 2 * margin, body_font)
            y += 8
        pages.append(page)

    pdf_path = report_root / "satellite_image_comparison_report.pdf"
    png_dir = report_root / "pdf_pages"
    png_dir.mkdir(exist_ok=True)
    for index, page in enumerate(pages, start=1):
        page.save(png_dir / f"page_{index:02d}.png")
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])


def main() -> None:
    args = parse_args()
    report_root = args.output_dir
    figures_dir = report_root / "figures"
    vectors_dir = report_root / "vectors"
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset_dir}")
    figures_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for label, pre_name, post_name in IDRT_PAIRS:
        name = safe_name(label)
        pre = load_image(args.dataset_dir / pre_name)
        post = load_image(args.dataset_dir / post_name)
        overlap, support = satellite_shared_gxh_reference(post, pre, THRESHOLD)
        heat, _similarity = satellite_change_heatmap(post, pre, STRENGTH, THRESHOLD, support)
        mask_overlay, _mask_similarity = satellite_change_mask(post, pre, STRENGTH, THRESHOLD, support)
        metrics = satellite_change_metrics(post, pre, THRESHOLD)
        candidate_mask = change_mask_from_pair(post, pre, support)
        features = vectorize_mask(candidate_mask)
        vector_overlay = draw_vector_overlay(post, features)

        pre_path = figures_dir / f"{name}_pre.png"
        post_path = figures_dir / f"{name}_post.png"
        overlap_path = figures_dir / f"{name}_shared_gxh_overlap.png"
        heat_path = figures_dir / f"{name}_change_heatmap.png"
        mask_path = figures_dir / f"{name}_change_mask.png"
        comparison_path = figures_dir / f"{name}_comparison_sheet.png"
        vector_overlay_path = figures_dir / f"{name}_vectorized_change_overlay.png"
        save_rgb(pre_path, pre)
        save_rgb(post_path, post)
        save_rgb(overlap_path, overlap)
        save_rgb(heat_path, heat)
        save_rgb(mask_path, mask_overlay)
        save_rgb(vector_overlay_path, vector_overlay)
        save_rgb(comparison_path, make_comparison_sheet(label, pre, post, overlap, heat, mask_overlay))

        geojson_path = vectors_dir / f"{name}_candidate_change_footprints.geojson"
        svg_path = vectors_dir / f"{name}_candidate_change_footprints.svg"
        save_geojson(geojson_path, features)
        save_svg_overlay(svg_path, post.shape[1], post.shape[0], features)

        row: dict[str, object] = {
            "name": label,
            "pre_file": pre_name,
            "post_file": post_name,
            "comparison_figure": comparison_path.relative_to(report_root).as_posix(),
            "geojson": geojson_path.relative_to(report_root).as_posix(),
            "svg": svg_path.relative_to(report_root).as_posix(),
            "vector_overlay": vector_overlay_path.relative_to(report_root).as_posix(),
            "vector_feature_count": len(features),
            "excluded_area_percent": 100.0 - float(metrics["shared_gxh_overlap_percent"]),
            **metrics,
        }
        rows.append(row)

    with (report_root / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    clean_rows = [{**row, "analyst_use_grade": analyst_grade(row)} for row in rows]
    with (report_root / "metrics_summary_clean.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in CLEAN_METRIC_COLUMNS} for row in clean_rows)
    (report_root / "metrics_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    build_reports(rows, report_root)
    print(report_root.resolve())


if __name__ == "__main__":
    main()
