from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaia.annotations.store import append_jsonl, json_safe, read_jsonl
from gaia.core.metrics import image_metrics, processed_metrics, template_match_metrics
from gaia.core.template_matching import template_match_search


class TemplateMatchingTests(unittest.TestCase):
    def test_exact_template_match_finds_known_center_within_stride(self) -> None:
        rng = np.random.default_rng(7)
        base = rng.integers(0, 80, size=(72, 88, 3), dtype=np.uint8)
        template = rng.integers(120, 255, size=(13, 15, 3), dtype=np.uint8)
        top_left_x = 42
        top_left_y = 31
        base[top_left_y : top_left_y + template.shape[0], top_left_x : top_left_x + template.shape[1]] = template

        result = template_match_search(
            base,
            template,
            stride_px=1,
            angle_min=0,
            angle_max=0,
            angle_step=1,
        )

        self.assertEqual(result["best_x"], top_left_x + template.shape[1] // 2)
        self.assertEqual(result["best_y"], top_left_y + template.shape[0] // 2)
        self.assertGreater(float(result["corr_score"]), 0.99)
        self.assertLess(float(result["difference_score"]), 1.0)
        self.assertEqual(result["best_vector"], [[result["best_x"]], [result["best_y"]], [result["corr_score"]]])

    def test_template_larger_than_base_is_rejected(self) -> None:
        base = np.zeros((10, 10, 3), dtype=np.uint8)
        template = np.zeros((12, 12, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "larger than the selected base image"):
            template_match_search(base, template, stride_px=1, angle_min=0, angle_max=0, angle_step=1)

    def test_template_metrics_flatten_result_for_table(self) -> None:
        metrics = template_match_metrics(
            {
                "best_x": 12,
                "best_y": 34,
                "corr_score": 0.875,
                "difference_score": 14.5,
                "rotation_deg": 90.0,
                "template_width": 9,
                "template_height": 11,
                "stride_px": 2,
            }
        )

        self.assertEqual(metrics["template_match_x"], 12)
        self.assertEqual(metrics["template_match_y"], 34)
        self.assertEqual(metrics["template_match_rotation_deg"], 90.0)
        self.assertEqual(metrics["template_stride_px"], 2)


class MetricsTests(unittest.TestCase):
    def test_image_and_processed_metrics_are_numeric(self) -> None:
        gray = np.array([[0, 10, 20], [30, 255, 40], [50, 60, 70]], dtype=np.uint8)

        raw = image_metrics(gray)
        processed = processed_metrics(gray)

        self.assertEqual(raw["min"], 0.0)
        self.assertEqual(raw["max"], 255.0)
        self.assertGreater(raw["range"], 0.0)
        self.assertGreater(processed["processed_active_percent"], 0.0)


class AnnotationStoreTests(unittest.TestCase):
    def test_jsonl_annotations_round_trip_numpy_safe_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            annotation = {
                "annotation_id": "ann-test",
                "points": np.array([[3, 4], [5, 6]], dtype=np.int32),
                "center_x": np.int32(4),
                "center_y": np.int32(5),
                "label": "test note",
            }

            append_jsonl(path, json_safe(annotation))
            rows = read_jsonl(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["points"], [[3, 4], [5, 6]])
        self.assertEqual(rows[0]["center_x"], 4)
        self.assertEqual(rows[0]["label"], "test note")


class ImportTests(unittest.TestCase):
    def test_public_entrypoint_imports(self) -> None:
        from gaia.main import main

        self.assertTrue(callable(main))

    def test_idrt_pair_window_path_imports_resize_helper(self) -> None:
        from gaia.ui.main_window import resize_like

        self.assertTrue(callable(resize_like))

    def test_missing_idrt_dataset_shows_help_without_crashing(self) -> None:
        from qtpy.QtWidgets import QApplication, QMessageBox
        import gaia.ui.main_window as main_window
        from gaia.ui.main_window import FastQtPlatform

        app = QApplication.instance() or QApplication([])
        window = FastQtPlatform(0, "Default", "Auto", 1280, 720, 30.0)
        original_dataset_dir = main_window.DEFAULT_IDRT_DATASET_DIR
        original_information = QMessageBox.information
        captured: dict[str, object] = {}

        def capture_information(parent: object, title: str, text: str) -> None:
            captured["title"] = title
            captured["text"] = text

        try:
            main_window.DEFAULT_IDRT_DATASET_DIR = Path("__missing_idrt_dataset_for_test__")
            QMessageBox.information = capture_information  # type: ignore[method-assign]
            window.open_idrt_pair()
        finally:
            main_window.DEFAULT_IDRT_DATASET_DIR = original_dataset_dir
            QMessageBox.information = original_information  # type: ignore[method-assign]
            window.close()
            app.processEvents()

        self.assertEqual(captured["title"], "IDRT dataset unavailable")
        self.assertIn("intentionally not included", str(captured["text"]))
        self.assertIn("IDRT dataset unavailable", window.status_label.text())


if __name__ == "__main__":
    unittest.main()
