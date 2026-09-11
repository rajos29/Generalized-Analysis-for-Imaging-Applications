# GAIA: Generalized Analysis for Imaging Applications

GAIA is a Python/OpenCV/Qt workstation for exploratory analysis of image-like sensing data. It supports microscopy arrays, satellite image pairs, still images, videos, live camera frames, and sensor-inspired radar/LiDAR/ultrasonic operators from a single desktop interface.

The current MVP focuses on fast visual inspection, foundational image processing, template matching, annotation notes, metrics export, and experiment/report packaging.

<img width="1919" height="226" alt="ssc" src="https://github.com/user-attachments/assets/cb8cc3ee-d09d-4a8a-8120-783fc72db9a0" />

## Primary App

Run this file:

```powershell
python gaia.py
```

Or run the launcher:

```powershell
.\run_gaia.bat
```

If `run_gaia.bat` uses the wrong Python environment, point it at the environment with the required packages:

```powershell
$env:GAIA_PYTHON="path\to\python.exe"
.\run_gaia.bat
```

It opens a lightweight Qt viewer. The same display path works for:

- live camera frames
- still images
- HDF5 imaging arrays (`.h5`, `.hdf5`) through `Open Image`
- videos

The app now starts on a home screen:

- `Image / Video Analysis`: opens the high-FPS live/file analysis workspace.
- `Camera Acquisition`: opens the live acquisition workspace.
- `Open Experiment`: opens saved experiment runs without starting camera capture.

Use `Back` from either screen to return home. Leaving an analysis screen stops active camera/video sources.

The fast Qt app includes:

- Open Camera
- Open Image
- Open IDRT Pair for the Texas A&M IDRT application satellite preview dataset
- Open Video
- per-pane source selectors for choosing available data products in each display pane. Generic images use `Raw/Input`; satellite pairs use `Post/Event` and `Reference/Pre`.
- Play/Pause and frame slider for videos
- `Original Only` mode for maximum FPS
- `Gradient x Hessian` mode with processed view on the left and original on the right
- algorithm selector with live processing tools
- radar/LiDAR/ultrasonic-inspired operators for sensor-analysis demonstrations
- denoising and sharpening algorithms for scientific imaging exploration
- strength and threshold sliders for algorithm parameters
- evidence workspace with live metrics, annotation table, and rich-text analyst notes
- pin and polygon annotations on visible image panes
- `Process every N frames` control for FPS tuning
- operation progress bar for prompted image processing and template-search operations
- one-row CSV save and continuous CSV logging
- one-click experiment save into `experiments/`
- live FPS, failed-frame count, and brightness readout
- professional dark aerospace-style interface

Install dependencies in your preferred Python environment:

```powershell
python -m pip install -r requirements.txt
```

HDF5 datasets also require:

```powershell
python -m pip install h5py
```

Run the smoke-test suite before sharing or reviewing changes:

```powershell
python -m py_compile gaia.py idrt_satellite_report.py (Get-ChildItem -Path gaia,tests -Recurse -Filter *.py).FullName
python -m unittest discover -s tests -v
```

The tests cover core template matching behavior, metric flattening, annotation JSONL persistence, and public entrypoint importability.

The primary app defaults to `Camera 1`, `DirectShow`, `Auto`, `640x360` because this OpenCV build can read that stream reliably. `MSMF` remains selectable, and the camera opener falls back to other backends if the selected one cannot capture by index.

For live camera work, use `Original Only` first to see the best achievable capture/display FPS. Switch to `Gradient x Hessian` when you need the processed view.

Available processing algorithms:

- Radar Range-Doppler Map
- LiDAR Depth Relief
- Ultrasonic B-Scan Envelope
- Satellite Change Heatmap
- Satellite Change Mask
- Satellite Shared GxH Overlap
- Satellite Water Emphasis
- Satellite Vegetation Proxy
- Gradient x Hessian
- Gradient Magnitude
- Hessian Magnitude
- CLAHE Contrast
- Otsu Threshold
- Adaptive Threshold
- Canny Edges
- Edge Overlay
- Frame Similarity Heatmap
- Denoise - Gaussian
- Denoise - Median
- Denoise - Bilateral
- Denoise - Non-Local Means
- Sharpen - Unsharp Mask
- Sharpen - Laplacian
- Sharpen - High Boost
- Morph Open
- Morph Close
- Focus Map
- Background Subtract

`Radar Range-Doppler Map` converts the frame into a log-scaled 2D frequency-energy map using a windowed FFT. It is useful as a radar-inspired spectral view for periodic structure, texture, and motion-like spatial patterns in image/video data.

`LiDAR Depth Relief` treats luminance as a proxy depth surface, smooths it, then renders shaded relief from local gradients. It is useful for explaining terrain/height-map thinking and surface discontinuity detection.

`Ultrasonic B-Scan Envelope` emphasizes vertical echo-like boundaries using smoothed gradient energy and an envelope-style heatmap. It is useful for demonstrating how layered interfaces, inclusions, and reflectors can be screened in image-like acoustic data.

`Gradient x Hessian` uses the original low-frequency-noise microscopy operator: edge-padded 3 x 3 smoothing, gradient x Hessian, then edge-padded 3 x 3 smoothing again. The `Strength` slider controls blur/scale/kernel style parameters depending on the selected algorithm. For denoising, higher strength increases smoothing. For sharpening, higher strength increases enhancement intensity. The `Threshold` slider affects adaptive threshold and Canny edge detection.

`Edge Overlay` draws detected edges directly on top of the source image in an accent color. `Frame Similarity Heatmap` compares the current frame with the adjacent previous frame and renders an activation-style heatmap, with brighter regions indicating stronger frame-to-frame change. The table includes `adjacent_similarity_percent` for this comparison.

## Convolutional Image Search

GAIA includes a global template-matching workflow called `Convolutional Image Search`. It is available from the analysis toolbar whenever an image-like frame is loaded, including microscopy arrays, satellite pairs, live camera frames, still images, video frames, and processed radar/LiDAR/ultrasonic-style outputs.

The method is formal template matching using 2D cross-correlation across grayscale images. A desired object is treated as a filter/template, swept across the selected base image, and scored at each valid center coordinate. This is algorithmic and untrained; it is related to the way convolutional neural networks apply filters, but it does not use learned weights.

Workflow:

- Choose a search base. Generic images expose `Raw/Input`; satellite pairs expose `Post/Event`; both can also expose processed, reference, GxH, and CIS result sources when available.
- Choose a filter source, then create the filter with `Load Filter` or `Crop Filter`.
- Set stride and rotation sweep mode. `Light 90 deg` checks `0, 90, 180, 270`; `Deep 45 deg` checks the full circle every 45 degrees; `Custom fine` uses the min/max/step angle boxes for local correction.
- Press `Run Image Search`.

GAIA selects the best match by maximum normalized cross-correlation and creates a heatmap as a selectable pane source. Running CIS does not force the pane arrangement; the analyst chooses whether panes show pre, post, processed output, the CIS filter, the CIS heatmap, or other available sources. During the sweep, the progress bar reports the current angle, sweep index, and stride. The best location is recorded as `[[x], [y], [score]]`, where `x` and `y` are the center pixel coordinates in the selected base image and `score` is the correlation score. The metrics table also reports the search base, filter source, mean absolute pixel-difference score at the best match, best rotation, template size, and stride, preserving the original difference-based interpretation while making exported rows reproducible.

## Satellite / IDRT Preview Workflow

The Image / Video Analysis screen includes `Open IDRT Pair`, which opens paired preview images from an external dataset folder. The dataset is not included in this repository. Set `IDRT_DATASET_DIR` to the folder containing the supplied image files:

```powershell
$env:IDRT_DATASET_DIR="path\to\IDRT application dataset"
python gaia.py
```

If the folder or expected pair files are not present, GAIA shows a dataset policy/help message instead of crashing. Reviewers without the IDRT dataset can still use `Open Image`, `Open Video`, HDF5 input, or their own local imagery to exercise the general processing, annotation, metrics, and Convolutional Image Search workflows.

Available pairs:

- Maxar pre/post Treasure Island
- NOAA post-Helene/post-Milton Treasure Island
- Sentinel-2 pre/post regional context

The satellite pair view shows post-event imagery, a processed change output, and the pre/reference image side by side. `Operator target` chooses whether single-image operators such as `Gradient x Hessian`, `CLAHE Contrast`, or `Edge Overlay` run on the post/current image or the pre/reference image. `Satellite Shared GxH Overlap` computes `sqrt(GxH_pre * GxH_post)` after alignment to reveal the shared structural support where paired change analysis is most defensible. `Satellite Change Heatmap` overlays per-pixel RGB change intensity on the post image while suppressing non-shared crop/coverage regions. `Satellite Change Mask` highlights supported pixels above the threshold slider value. `Satellite Water Emphasis` and `Satellite Vegetation Proxy` are RGB-preview screening aids, not calibrated remote-sensing indices.

Use the alignment controls before trusting a change map:

- `Align X/Y`: shifts the pre/reference image in pixels.
- `Rot`: rotates the pre/reference image in degrees.
- `Scale`: scales the pre/reference image as a percentage.
- `Auto Align`: runs a prompted low-resolution affine search to estimate translation, rotation, and scale from shared GxH structure.
- `Auto Tilt`: disabled for the current demo; projective view-angle correction is tracked as future work.
- `Reset Align`: returns to the raw pair alignment.

These controls are intentionally lightweight. Manual controls remain available for analyst correction, while `Auto Align` gives a practical first-pass registration estimate without slowing down normal browsing. The automatic affine search uses downsampled/quantized GxH maps, sweeps a small set of rotations and scales, estimates translation with phase correlation, and writes the result back into the same alignment controls before recalculating candidate change maps. Projective tilt correction remains future work because roof/parallax mismatch needs more explicit direction, region-of-interest selection, and visual diagnostics before it is reliable enough for the demo.

The shared GxH overlap layer is stored in the background during satellite change processing. It is also saved as `snapshots/shared_gxh_overlap.png` when you save a satellite experiment, so the report can show which pixels were included/excluded from candidate change-map calculations.

Satellite runs add report-oriented metrics to the table and CSV:

- `change_mean_delta`
- `change_p95_delta`
- `change_active_percent`
- `water_pre_percent`
- `water_post_percent`
- `water_change_percent`
- `green_pre_mean`
- `green_post_mean`
- `green_change_mean`
- alignment settings and operator target
- `shared_gxh_overlap_percent`
- `excluded_area_percent`
- `valid_post_area_percent`
- `valid_pre_area_percent`
- `structural_similarity_score`
- `change_median_delta`
- `change_p90_delta`
- `change_max_delta`
- `change_component_count`
- `filtered_change_region_count`
- `filtered_change_region_area_percent`
- `largest_change_region_area_px`
- `largest_change_component_percent`
- `mean_change_region_area_px`
- `median_change_region_area_px`
- `change_region_density_per_megapixel`
- `change_severity_index`

Generate the standalone IDRT report locally with:

```powershell
python idrt_satellite_report.py --dataset-dir "path\to\IDRT application dataset"
```

Generated reports, figures, vectors, and metrics are written under `reports/`. This folder is intentionally ignored by Git because the outputs may contain source or derived dataset imagery. Share those deliverables separately only when the dataset terms allow it.

Use `Save Experiment` after opening an IDRT pair to save snapshots, metrics, manifest data, and `satellite_report_draft.md`. Treat this as foundational preview-image analysis: useful for portfolio demonstration and rapid visual screening, but not a substitute for georeferenced imagery, image registration, cloud masking, sensor calibration, or GIS-grade damage assessment.

For heavier tools, increase `Process every N frames`. The original camera frame still updates every frame, while the processed view and processed metrics update on the selected cadence.

The metrics table reports:

- source, algorithm, strength, threshold, and process cadence
- instant and rolling FPS
- failed frames
- original brightness/contrast/clipping/focus values
- processed mean, processed standard deviation, and active-pixel percentage
- radar peak spectral energy and spectral entropy when `Radar Range-Doppler Map` is active
- LiDAR-style relief standard deviation and edge density when `LiDAR Depth Relief` is active
- ultrasonic echo density and echo mean when `Ultrasonic B-Scan Envelope` is active
- Convolutional Image Search best x/y, correlation score, difference score, best rotation, template size, and stride

Use `Pin Note` or `Polygon Note` to attach analyst observations to the currently displayed image artifact. Confirmed annotations store image-space coordinates, source labels, rich-text notes, plain-text notes, current processing settings, and current metrics in `experiments/annotations.jsonl`.

Use `Save Experiment` to write a run folder containing `manifest.json`, `metrics.csv`, current frame snapshots, and relevant `annotations.jsonl` records when annotations exist. Use `Save Row` to export only the current table row, or `Start CSV Log` / `Stop Log` to record metrics continuously to a CSV file.

Experiment folders use this shape:

```text
experiments/
  run_YYYY-MM-DD_HHMMSS/
    manifest.json
    metrics.csv
    annotations.jsonl
    snapshots/
      original.png
      processed.png
```

Open `Open Experiment` from the home screen to browse saved runs and view summary metrics without opening the camera.

The metrics table uses a wider value column and sits beside the controls so timestamps, algorithm names, and longer values are readable.

You can also open a file directly at startup:

```powershell
python gaia.py --open path\to\image_or_video
```

For HDF5 files opened from the UI, the app detects paired microscopy/imaging dataset groups such as `noisy`, `noisy_1`, `noisy_2`, and `clean`. The HDF5 selector lets you choose the sample, input group, reference group, and z-slice when a sample is a stack. After loading, the analysis screen shows compact HDF5 controls so you can step through samples and z-slices without reopening the file.

When a reference group such as `clean` is available, denoising and sharpening runs report reference metrics:

- MSE
- PSNR
- SSIM
- residual mean
- residual standard deviation

These metrics are saved in experiment CSV rows alongside the existing brightness, contrast, focus, FPS, and processed-output metrics.

## Image Space View

For paired HDF5 samples, use `Image Space View` to switch from the normal two-pane display into a graph-only research layout:

- processed-result x/y/value 3D surface
- clean/reference x/y/value 3D surface

The processed-result surface updates when you change the selected algorithm, strength, threshold, sample, or z-slice. The 3D surfaces treat each pixel as a height sample: x and y are image coordinates in pixels, and z is grayscale intensity from 0 to 255. The app uses a GPU-backed VisPy surface renderer when available and falls back to Matplotlib if needed. VisPy surfaces include colored x/y/v axis guides. Use `Surface downsample` to render every Nth pixel; the default of 8 renders about 64 x 64 points from a 512 x 512 slice for better responsiveness. The z-slice and downsample controls also have explicit `-` / `+` buttons for reliable stepping. Use `Reset View` to reset the 3D camera angle.

When a z-stack HDF5 sample is loaded, the standard view also shows a variance heatmap beside the processed and reference images. This is a 2D plot of full-stack per-pixel variance for the selected HDF5 input sample. Brighter/hotter regions indicate pixels whose grayscale values vary more across z-slices.

For z-stack samples, the app also computes full-stack variance metrics for the selected input group:

- `z_variance_mean`
- `z_variance_max`
- `z_variance_p95`
- `z_high_variance_percent`

These metrics are intended to support speckle-noise exploration across slices while keeping the first advanced renderer CPU-only and easy to replace with a GPU path later.

## Future Work

- Improve `.bat` launcher reliability across machines and Python environments.
- Add multi-scale template matching for objects that appear at different sizes.
- Add FFT-accelerated correlation and optional GPU acceleration for larger search spaces.
- Add feature-assisted template search for perspective, scale, and partial-occlusion mismatch.
- Add projective tilt/homography diagnostics for satellite viewing-angle and roof-parallax mismatch.

## Project Layout

- `gaia.py`: stable public entrypoint.
- `run_gaia.bat`: Windows launcher for the desktop app.
- `gaia/main.py`: application startup, CLI arguments, and optional file opening.
- `gaia/config.py`: app constants, algorithms, pane sources, metric names, and data paths.
- `gaia/core/`: image processing, satellite comparison, template matching, metrics, media loading, and HDF5 helpers.
- `gaia/ui/`: Qt main window, image panes, evidence panel behavior, theme, dialogs, and 3D surface view.
- `gaia/annotations/`: JSON/JSONL persistence for analyst annotations.
- `gaia/experiments/`: experiment naming and export/report helpers.
- `gaia/workers/`: threaded camera and device workers.
- `tests/`: smoke tests for core behavior and refactor-regression coverage.

## Capture QA Metrics

- FPS: instant and rolling average.
- Dropped/failed frames.
- Brightness: mean, median, min/max, black clipping %, white clipping %.
- Contrast: standard deviation and dynamic range.
- Focus: variance of Laplacian on luminance.
- Motion/stability: frame-to-frame mean absolute difference.
- Channel balance: RGB means and normalized RGB ratios.

The GAIA Evidence panel keeps the highest-signal live metrics visible during analysis while preserving expanded metrics in CSV and experiment exports.

## Layout

- `Original Only`: one live original view.
- `Gradient x Hessian`: processed grayscale view on the left, original on the right.
