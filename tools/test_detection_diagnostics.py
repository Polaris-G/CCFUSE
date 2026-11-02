"""Behavioral tests for offline detection diagnostics; no images or model needed.

Place this file in the repository's tools/ directory to run it directly.
"""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "diagnose_detections.py"
SPEC = importlib.util.spec_from_file_location("diagnose_detections_under_test", MODULE_PATH)
DIAG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DIAG
SPEC.loader.exec_module(DIAG)


def gt(*rows):
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def pred(*rows):
    return np.asarray(rows, dtype=np.float64).reshape(-1, 6)


class BoxIoUTests(unittest.TestCase):
    def test_xyxy_iou_intersection_union_and_disjoint(self):
        a = np.array([[0, 0, 2, 2], [4, 4, 6, 6]], dtype=float)
        b = np.array([[0, 0, 2, 2], [1, 0, 3, 2]], dtype=float)
        np.testing.assert_allclose(DIAG.box_iou(a, b), [[1.0, 1.0 / 3.0], [0.0, 0.0]])

    def test_empty_inputs_keep_matrix_shape(self):
        empty = np.empty((0, 4), dtype=float)
        one = np.array([[0, 0, 1, 1]], dtype=float)
        self.assertEqual(DIAG.box_iou(empty, one).shape, (0, 1))
        self.assertEqual(DIAG.box_iou(one, empty).shape, (1, 0))
        self.assertEqual(DIAG.box_iou(empty, empty).shape, (0, 0))

    def test_zero_area_boxes_have_finite_zero_iou(self):
        zero = np.array([[.5, .5, .5, .5]], dtype=float)
        normal = np.array([[.25, .25, .75, .75]], dtype=float)
        np.testing.assert_array_equal(DIAG.box_iou(zero, normal), [[0.0]])
        np.testing.assert_array_equal(DIAG.box_iou(zero, zero), [[0.0]])


class AnalyzeImageTests(unittest.TestCase):
    def analyze(self, labels, predictions, conf=0.25, iou=0.5, nc=3):
        return DIAG.analyze_image(labels, predictions, nc, conf, iou, localization_iou=0.1)

    def assert_confusion(self, result, expected):
        matrix = np.asarray(result["confusion"])
        np.testing.assert_array_equal(matrix, np.asarray(expected))
        self.assertTrue(np.all(matrix >= 0))

    def test_exact_match_is_true_positive(self):
        result = self.analyze(gt([0, .5, .5, .25, .25]), pred([0, .5, .5, .25, .25, .8]))
        self.assertEqual(list(result["gt_status"]), ["tp"])
        self.assertEqual(list(result["pred_status"]), ["tp"])
        self.assertEqual(list(result["gt_pred_indices"]), [0])
        self.assert_confusion(result, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_wrong_class_records_predicted_row_true_column(self):
        result = self.analyze(gt([0, .5, .5, .25, .25]), pred([1, .5, .5, .25, .25, .8]))
        self.assertEqual(list(result["gt_status"]), ["classification"])
        self.assertEqual(list(result["pred_status"]), ["classification"])
        self.assertEqual(list(result["gt_pred_indices"]), [0])
        self.assert_confusion(result, [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_no_predictions_counts_every_ground_truth_as_unmatched(self):
        labels = gt([0, .25, .25, .125, .125], [2, .75, .75, .125, .125])
        result = self.analyze(labels, pred())
        self.assertEqual(list(result["gt_status"]), ["no_overlap_candidate", "no_overlap_candidate"])
        self.assertEqual(list(result["gt_pred_indices"]), [-1, -1])
        self.assertEqual(list(result["pred_status"]), [])
        self.assertEqual(result["retained_predictions"].shape, (0, 6))
        self.assert_confusion(result, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]])

    def test_no_ground_truth_counts_every_prediction_as_background(self):
        predictions = pred([0, .25, .25, .125, .125, .8], [2, .75, .75, .125, .125, .6])
        result = self.analyze(gt(), predictions)
        self.assertEqual(list(result["gt_status"]), [])
        self.assertEqual(list(result["gt_pred_indices"]), [])
        self.assertEqual(list(result["pred_status"]), ["background", "background"])
        self.assert_confusion(result, [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

    def test_both_empty(self):
        result = self.analyze(gt(), pred())
        self.assertEqual(result["retained_predictions"].shape, (0, 6))
        self.assertEqual(len(result["pred_source_indices"]), 0)
        self.assertEqual(list(result["gt_status"]), [])
        self.assertEqual(list(result["pred_status"]), [])
        np.testing.assert_array_equal(result["confusion"], np.zeros((4, 4)))

    def test_completely_disjoint_boxes_count_both_fp_and_fn(self):
        result = self.analyze(gt([0, .125, .125, .125, .125]), pred([1, .875, .875, .125, .125, .8]))
        self.assertEqual(list(result["gt_status"]), ["no_overlap_candidate"])
        self.assertEqual(list(result["gt_pred_indices"]), [-1])
        self.assertEqual(list(result["pred_status"]), ["background"])
        self.assert_confusion(result, [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0]])

    def test_zero_area_exported_prediction_is_preserved_as_false_positive(self):
        # Native-coordinate clipping can collapse a box originating in padding.
        result = self.analyze(gt([0, .5, .5, .25, .25]), pred([0, .5, .5, 0, .25, .8]))
        self.assertEqual(len(result["retained_predictions"]), 1)
        self.assertEqual(list(result["gt_status"]), ["no_overlap_candidate"])
        self.assertEqual(list(result["pred_status"]), ["background"])
        self.assertEqual(list(result["gt_pred_indices"]), [-1])
        self.assert_confusion(result, [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])

    def test_confidence_and_iou_boundaries_are_inclusive(self):
        # Binary-exact dimensions yield IoU == 0.5, avoiding roundoff ambiguity.
        result = self.analyze(gt([0, .5, .5, .25, .25]), pred([0, .5, .5, .5, .25, .25]))
        self.assertEqual(list(result["gt_status"]), ["tp"])
        self.assertEqual(list(result["pred_status"]), ["tp"])
        self.assertEqual(len(result["retained_predictions"]), 1)

    def test_confidence_filter_and_source_to_retained_index_mapping(self):
        labels = gt([0, .25, .25, .125, .125], [1, .75, .75, .125, .125])
        predictions = pred(
            [2, .5, .5, .125, .125, .1],
            [0, .25, .25, .125, .125, .9],
            [1, .75, .75, .125, .125, .249999],
            [1, .75, .75, .125, .125, .25],
        )
        result = self.analyze(labels, predictions)
        np.testing.assert_array_equal(result["pred_source_indices"], [1, 3])
        np.testing.assert_array_equal(result["retained_predictions"], predictions[[1, 3]])
        self.assertEqual(list(result["gt_pred_indices"]), [0, 1])
        self.assertEqual(list(result["gt_status"]), ["tp", "tp"])

    def test_all_predictions_filtered_out(self):
        result = self.analyze(gt([1, .5, .5, .25, .25]), pred([1, .5, .5, .25, .25, .249999]))
        self.assertEqual(result["retained_predictions"].shape, (0, 6))
        self.assertEqual(list(result["gt_status"]), ["no_overlap_candidate"])
        self.assertEqual(list(result["gt_pred_indices"]), [-1])
        self.assertEqual(result["confusion"][3, 1], 1)

    def test_duplicate_detection_has_one_tp_and_one_background_column_fp(self):
        result = self.analyze(gt([0, .5, .5, .25, .25]), pred(
            [0, .5, .5, .25, .25, .9], [0, .5, .5, .25, .25, .8]))
        self.assertCountEqual(list(result["pred_status"]), ["tp", "duplicate"])
        self.assertEqual(list(result["gt_status"]), ["tp"])
        matched = result["gt_pred_indices"][0]
        self.assertEqual(result["pred_status"][matched], "tp")
        self.assert_confusion(result, [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_same_class_tp_takes_priority_over_higher_iou_wrong_class(self):
        result = self.analyze(gt([0, .5, .5, .5, .5]), pred(
            [0, .5, .5, .25, .5, .5], [1, .5, .5, .5, .5, .99]))
        self.assertEqual(list(result["gt_status"]), ["tp"])
        self.assertEqual(list(result["gt_pred_indices"]), [0])
        self.assertEqual(list(result["pred_status"]), ["tp", "other_overlap"])
        self.assert_confusion(result, [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_global_iou_matching_and_assignment_conflict(self):
        # Smaller GT comes first, but the second GT has the globally higher IoU.
        labels = gt([0, .5, .5, .25, .5], [0, .5, .5, .5, .5])
        result = self.analyze(labels, pred([0, .5, .5, .5, .5, .9]))
        self.assertEqual(list(result["gt_status"]), ["assignment_conflict", "tp"])
        self.assertEqual(list(result["gt_pred_indices"]), [0, 0])
        self.assertEqual(list(result["pred_status"]), ["tp"])
        self.assert_confusion(result, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])

    def test_same_class_low_iou_is_localization_candidate(self):
        result = self.analyze(gt([0, .5, .5, .5, .5]), pred([0, .5, .5, .125, .5, .8]))
        self.assertEqual(list(result["gt_status"]), ["localization_candidate"])
        self.assertEqual(list(result["gt_pred_indices"]), [0])
        self.assertEqual(list(result["pred_status"]), ["other_overlap"])
        self.assertEqual(result["confusion"][3, 0], 1)
        self.assertEqual(result["confusion"][0, 3], 1)

    def test_other_class_low_iou_is_class_localization_candidate(self):
        result = self.analyze(gt([0, .5, .5, .5, .5]), pred([2, .5, .5, .125, .5, .8]))
        self.assertEqual(list(result["gt_status"]), ["class_localization_candidate"])
        self.assertEqual(list(result["gt_pred_indices"]), [0])
        self.assertEqual(list(result["pred_status"]), ["other_overlap"])
        self.assertEqual(result["confusion"][3, 0], 1)
        self.assertEqual(result["confusion"][2, 3], 1)

    def test_inputs_are_not_mutated(self):
        labels = gt([0, .5, .5, .25, .25])
        predictions = pred([0, .5, .5, .25, .25, .8])
        before_labels, before_preds = labels.copy(), predictions.copy()
        self.analyze(labels, predictions)
        np.testing.assert_array_equal(labels, before_labels)
        np.testing.assert_array_equal(predictions, before_preds)

    def test_classification_example_uses_actual_pair_not_highest_iou_gt(self):
        labels = gt([0, .5, .5, .5, .5], [1, .55, .5, .5, .5])
        predictions = pred([0, .5, .5, .5, .5, .9], [2, .5, .5, .5, .5, .8])
        result = self.analyze(labels, predictions)
        self.assertEqual(result['pred_status'][1], 'classification')
        self.assertEqual(DIAG.associated_gt(result, labels, 1), 1)

    def test_duplicate_example_prefers_matching_class(self):
        labels = gt([0, .5, .5, .5, .5], [1, .55, .5, .5, .5])
        predictions = pred([0, .5, .5, .5, .5, .9], [1, .55, .5, .5, .5, .9],
                           [0, .55, .5, .5, .5, .8])
        result = self.analyze(labels, predictions)
        self.assertEqual(result['pred_status'][2], 'duplicate')
        self.assertEqual(DIAG.associated_gt(result, labels, 2), 0)

    def test_background_example_does_not_assign_unrelated_gt(self):
        labels = gt([0, .125, .125, .125, .125])
        result = self.analyze(labels, pred([1, .875, .875, .125, .125, .8]))
        self.assertEqual(DIAG.associated_gt(result, labels, 0), -1)


class ReadLabelsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "labels.txt"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")
        return self.path

    def read(self, text, columns=5, nc=3):
        return DIAG.read_labels(self.write(text), columns, nc)

    def test_valid_gt_and_prediction_formats(self):
        np.testing.assert_allclose(self.read("0 0.5 0.5 0.25 0.25\n"), [[0, .5, .5, .25, .25]])
        np.testing.assert_allclose(self.read("2 0.5 0.5 0.25 0.25 0.75\n", columns=6),
                                   [[2, .5, .5, .25, .25, .75]])

    def test_empty_file_keeps_declared_column_count(self):
        for columns in (5, 6):
            with self.subTest(columns=columns):
                self.assertEqual(self.read("\n \t\n", columns=columns).shape, (0, columns))

    def test_missing_file_requires_explicit_opt_in(self):
        with self.assertRaises(FileNotFoundError):
            DIAG.read_labels(self.path, 6, 3)
        self.assertEqual(DIAG.read_labels(self.path, 6, 3, missing_ok=True).shape, (0, 6))

    def test_column_mismatch_and_nonnumeric_fields_fail(self):
        for text, columns in (("0 .5 .5 .25", 5), ("0 .5 .5 .25 .25 .9", 5),
                              ("0 .5 .5 .25 .25", 6), ("van .5 .5 .25 .25", 5)):
            with self.subTest(text=text, columns=columns), self.assertRaises(ValueError):
                self.read(text, columns=columns)

    def test_nonfinite_values_fail(self):
        for text, columns in (("0 nan .5 .25 .25", 5), ("0 .5 inf .25 .25", 5),
                              ("0 .5 .5 .25 .25 nan", 6), ("inf .5 .5 .25 .25", 5)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text, columns=columns)

    def test_class_must_be_integer_within_range(self):
        for cls in ("-1", "3", "0.5"):
            with self.subTest(cls=cls), self.assertRaises(ValueError):
                self.read(f"{cls} .5 .5 .25 .25")

    def test_normalized_coordinates_and_positive_size(self):
        invalid = ["0 -0.01 .5 .1 .1", "0 1.01 .5 .1 .1", "0 .5 1.01 .1 .1",
                   "0 .5 .5 0 .1", "0 .5 .5 .1 0", "0 .5 .5 -.1 .1", "0 .5 .5 .1 -.1",
                   "0 .5 .5 1.1 .1", "0 .5 .5 .1 1.1"]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)

    def test_zero_area_predictions_are_valid_but_negative_sizes_are_not(self):
        for width, height in ((0, .25), (.25, 0), (0, 0)):
            with self.subTest(width=width, height=height):
                values = self.read(f"0 .5 .5 {width} {height} .8", columns=6)
                self.assertEqual(values.shape, (1, 6))
        for width, height in ((-.01, .25), (.25, -.01)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                self.read(f"0 .5 .5 {width} {height} .8", columns=6)

    def test_bbox_corners_outside_image_fail(self):
        for text in ("0 .05 .5 .2 .2", "0 .95 .5 .2 .2", "0 .5 .05 .2 .2", "0 .5 .95 .2 .2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)

    def test_bbox_corner_rounding_tolerance(self):
        self.assertEqual(self.read("0 .1 .5 .200018 .2").shape, (1, 5))  # min x = -9e-6
        with self.assertRaises(ValueError):
            self.read("0 .1 .5 .200022 .2")  # min x = -1.1e-5

    def test_duplicate_numeric_rows_fail(self):
        for text, columns in (("0 .5 .5 .25 .25\n0 .5 .5 .25 .25\n", 5),
                              ("0 .5 .5 .25 .25 .8\n0 .5 .5 .25 .25 .8\n", 6)):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                self.read(text, columns=columns)

    def test_same_box_with_different_scores_is_not_a_duplicate_row(self):
        result = self.read("0 .5 .5 .25 .25 .8\n0 .5 .5 .25 .25 .7\n", columns=6)
        self.assertEqual(result.shape, (2, 6))


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "rgb/images/val"
        self.labels = self.root / "rgb/labels/val"
        self.predictions = self.root / "predictions"
        for path in (self.images, self.labels, self.predictions):
            path.mkdir(parents=True)
        for name in "abcd":
            (self.images / f"{name}.jpg").write_bytes(b"fixture-only-no-decoding-needed")
        (self.labels / "a.txt").write_text("0 .25 .25 .2 .2\n", encoding="utf-8")
        (self.labels / "b.txt").write_text("1 .7 .7 .2 .2\n", encoding="utf-8")
        (self.labels / "c.txt").write_text("1 .5 .5 .2 .2\n", encoding="utf-8")
        (self.labels / "d.txt").write_text("", encoding="utf-8")
        (self.predictions / "a.txt").write_text("0 .25 .25 .2 .2 .9\n", encoding="utf-8")
        (self.predictions / "b.txt").write_text("0 .7 .7 .2 .2 .8\n", encoding="utf-8")
        (self.predictions / "d.txt").write_text("0 .5 .5 .2 .2 .8\n", encoding="utf-8")
        self.data = self.root / "data.yaml"
        # JSON is a valid YAML subset and preserves Windows paths without shell escaping.
        self.data.write_text(json.dumps({"nc": 2, "names": ["bus", "van"],
                                         "val_rgb": str(self.images)}), encoding="utf-8")
        self.output = self.root / "report"

    def run_cli(self, *extra):
        return subprocess.run([sys.executable, "-B", str(MODULE_PATH), "--data", str(self.data),
                               "--pred-dir", str(self.predictions), "--output", str(self.output),
                               "--expected-images", "4", "--expected-labels", "3", "--conf", "0.25",
                               "--match-iou", "0.5", *extra], capture_output=True, text=True)

    def fingerprint_inputs(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for folder in
                (self.images, self.labels, self.predictions) for p in folder.iterdir()}

    def test_complete_report_counts_every_image_and_preserves_inputs(self):
        before = self.fingerprint_inputs()
        result = self.run_cli("--allow-missing-predictions")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, self.fingerprint_inputs())
        report = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report['tool_sha256'], hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest())
        self.assertIn('--allow-missing-predictions', report['command'])
        self.assertEqual(report["image_count"], 4)
        self.assertEqual(report["gt_count"], 3)
        self.assertEqual(report["missing_prediction_files"], ["c.jpg"])
        scenario = report["scenarios"][0]
        self.assertEqual(scenario["confusion_predicted_rows_true_columns"], [[1, 1, 1], [0, 0, 0], [0, 1, 0]])
        bus, van = scenario["classes"]
        self.assertEqual([bus[k] for k in ("tp", "fp", "fn")], [1, 2, 0])
        self.assertEqual([van[k] for k in ("tp", "fp", "fn")], [0, 0, 2])
        self.assertIsNone(van["precision"])
        self.assertEqual(van["gt_categories"], {"classification": 1, "no_overlap_candidate": 1})
        for name in ("summary.txt", "threshold_summary.csv", "examples.csv", "confusion_conf0.25_iou0.5.csv"):
            self.assertTrue((self.output / name).is_file(), name)
        original = (self.output / "summary.json").read_bytes()
        rerun = self.run_cli("--allow-missing-predictions")
        self.assertNotEqual(rerun.returncode, 0)
        self.assertIn("Output already exists", rerun.stderr)
        self.assertEqual((self.output / "summary.json").read_bytes(), original)

    def test_incomplete_exports_require_explicit_zero_prediction_assumption(self):
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing prediction TXT", result.stderr)
        self.assertFalse(self.output.exists())

    def test_wrong_split_counts_fail_before_creating_report(self):
        result = self.run_cli("--allow-missing-predictions", "--expected-labels", "999")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GT count mismatch", result.stderr)
        self.assertFalse(self.output.exists())

    def test_predictions_from_another_split_are_rejected(self):
        (self.predictions / "not_in_val.txt").write_text("0 .5 .5 .2 .2 .8", encoding="utf-8")
        result = self.run_cli("--allow-missing-predictions")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Predictions outside this split", result.stderr)
        self.assertFalse(self.output.exists())

    def test_close_thresholds_produce_distinct_matrix_files(self):
        result = self.run_cli('--allow-missing-predictions', '--conf', '0.2500001', '0.2500002')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        files = sorted(p.name for p in self.output.glob('confusion_*.csv'))
        self.assertEqual(files, ['confusion_conf0.2500001_iou0.5.csv', 'confusion_conf0.2500002_iou0.5.csv'])


if __name__ == "__main__":
    unittest.main()
