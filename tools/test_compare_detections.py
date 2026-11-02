"""Black-box tests for pairing completed detection diagnostics by GT identity.

Run with the project's Python environment. Fixtures are real tiny PNGs plus
YOLO labels; the original diagnose_detections CLI produces every input report.
No GPU, model weights, or production data are needed.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib


HERE = Path(__file__).resolve().parent
COMPARE = Path(os.environ.get("CCFUSE_COMPARE_SCRIPT", str(HERE / "compare_detections.py")))
_DIAGNOSTIC_CANDIDATES = [HERE / "diagnose_detections.py",
                          HERE.parent / "ccfuse_diagnostics" / "diagnose_detections.py"]
DIAGNOSE = Path(os.environ["CCFUSE_DIAGNOSE_SCRIPT"]) if "CCFUSE_DIAGNOSE_SCRIPT" in os.environ else next(
    (path for path in _DIAGNOSTIC_CANDIDATES if path.is_file()), _DIAGNOSTIC_CANDIDATES[0])


def tiny_png():
    """Create a valid RGB PNG without depending on an image library."""
    def chunk(name, data):
        payload = name + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xffffffff)
    pixels = b"".join(b"\x00" + b"\x80\xa0\xc0" * 32 for _ in range(32))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fingerprint_tree(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


class PairedDiagnosticsCliTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.assertTrue(COMPARE.is_file(), f"Comparison script missing: {COMPARE}")
        self.assertTrue(DIAGNOSE.is_file(), f"Diagnostic script missing: {DIAGNOSE}")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / "inputs"
        self.images = self.inputs / "rgb/images/val"
        self.ir_images = self.inputs / "ir/images/val"
        self.labels = self.inputs / "rgb/labels/val"
        for directory in (self.images, self.ir_images, self.labels):
            directory.mkdir(parents=True)
        for stem in "abcde":
            for directory in (self.images, self.ir_images):
                (directory / f"{stem}.png").write_bytes(tiny_png())
            # Four van targets plus one car target in the evaluated split.
            (self.labels / f"{stem}.txt").write_text(
                f"{0 if stem == 'e' else 1} .5 .5 .25 .25\n", encoding="utf-8")
        self.data = self.inputs / "data.yaml"
        self.data.write_text(json.dumps({"nc": 2, "names": ["car", "van"],
                                         "val_rgb": str(self.images), "val_ir": str(self.ir_images)}),
                             encoding="utf-8")
        self.predictions = {}
        self.reports = []
        for role in ("add", "cmfc", "full"):
            directory = self.inputs / role / "predictions"
            directory.mkdir(parents=True)
            self.predictions[role] = directory
            # Add and the other models have IDENTICAL aggregate diagnostics.
            # Their a/b GT identities switch TP <-> car error, exposing why
            # equal counts do not establish a shared set of mistakes.
            for stem in "ab":
                cls = int((stem == "b") if role == "add" else (stem == "a"))
                (directory / f"{stem}.txt").write_text(f"{cls} .5 .5 .25 .25 .8\n", encoding="utf-8")
            # At .05 the low-score correct class takes TP priority; at .25/.5
            # it is removed and the retained high-score car is a class error.
            (directory / "c.txt").write_text(
                "0 .5 .5 .25 .25 .9\n1 .5 .5 .25 .25 .15\n", encoding="utf-8")
            # d.txt is deliberately absent and explicitly recorded as such.
            (directory / "e.txt").write_text("0 .5 .5 .25 .25 .8\n", encoding="utf-8")
            report_dir = self.inputs / role / "diagnostic"
            result = subprocess.run(
                [sys.executable, "-B", str(DIAGNOSE), "--data", str(self.data),
                 "--pred-dir", str(directory), "--output", str(report_dir),
                 "--expected-images", "5", "--expected-labels", "5", "--focus", "van",
                 "--max-examples", "1", "--allow-missing-predictions"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.reports.append(report_dir / "summary.json")
        self.output = self.root / "paired"

    def run_compare(self, *extra, output=None, no_images=True):
        return subprocess.run(
            [sys.executable, "-B", str(COMPARE), "--reports", *map(str, self.reports),
             "--output", str(self.output if output is None else output), "--focus", "van",
             "--confusing-class", "car", "--expected-focus", "4", "--samples-per-group", "2",
             "--sample-seed", "42", *(["--no-images"] if no_images else []), *extra],
            capture_output=True, text=True)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_rejected(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.output.exists(), "Validation failure must precede output creation")

    def scenario(self, confidence=0.25, matching_iou=0.5):
        report = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        return next(row for row in report["scenarios"]
                    if row["confidence"] == confidence and row["matching_iou"] == matching_iou)

    def test_full_pairing_distinguishes_equal_totals_and_preserves_inputs(self):
        # The bounded examples CSV is not a source of full-population counts.
        for report in self.reports:
            (report.parent / "examples.csv").unlink()
        before = fingerprint_tree(self.inputs)
        self.assert_success(self.run_compare())
        self.assertEqual(before, fingerprint_tree(self.inputs))
        for name in ("summary.json", "summary.txt", "paired_targets.csv", "transitions.csv", "selection.csv"):
            self.assertTrue((self.output / name).is_file(), name)
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary["scenarios"]), 6)
        self.assertEqual({(row["confidence"], row["matching_iou"]) for row in summary["scenarios"]},
                         {(conf, iou) for conf in (.05, .25, .5) for iou in (.5, .75)})
        primary = self.scenario()
        self.assertEqual(primary["focus_gt"], 4)
        for role in ("add", "cmfc", "full"):
            self.assertEqual(primary["models"][role]["tp"], 1)
            self.assertEqual(primary["models"][role]["gt_categories"]["classification"], 2)
        paired = primary["pairwise"]["add_to_full"]
        self.assertEqual(paired["gained_tp"], 1)
        self.assertEqual(paired["lost_tp"], 1)
        self.assertEqual(paired["both_tp"], 0)
        self.assertEqual(paired["both_miss"], 2)
        self.assertEqual(paired["both_to_confusing_class"], 1)
        self.assertEqual(primary["groups"]["all_three_to_confusing_class"], 1)
        self.assertEqual(primary["groups"]["full_tp_add_miss"], 1)
        self.assertEqual(primary["groups"]["add_tp_full_miss"], 1)
        rows = read_csv(self.output / "paired_targets.csv")
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({(row["image"], row["gt_index_zero_based"],
                              row["confidence"], row["matching_iou"]) for row in rows}), 24)
        low = next(row for row in rows if Path(row["image"]).stem == "c"
                   and float(row["confidence"]) == .05 and float(row["matching_iou"]) == .5)
        high = next(row for row in rows if Path(row["image"]).stem == "c"
                    and float(row["confidence"]) == .25 and float(row["matching_iou"]) == .5)
        self.assertEqual(low["add_state"], "tp")
        self.assertEqual(high["add_state"], "as:car")
        self.assertEqual(high["add_status"], "classification")
        self.assertEqual(high["add_associated_class"], "car")
        self.assertAlmostEqual(float(high["add_associated_score"]), .9)
        self.assertEqual(self.scenario(.05)["groups"]["all_three_tp"], 1)

    def test_selection_is_deterministic_with_fixed_seed(self):
        self.assert_success(self.run_compare())
        second = self.root / "paired_second"
        self.assert_success(self.run_compare(output=second))
        self.assertEqual(read_csv(self.output / "selection.csv"), read_csv(second / "selection.csv"))

    def test_existing_output_is_rejected_without_overwrite(self):
        self.assert_success(self.run_compare())
        before = fingerprint_tree(self.output)
        result = self.run_compare()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, fingerprint_tree(self.output))

    def test_new_missing_prediction_is_not_silently_treated_as_zero(self):
        (self.predictions["full"] / "a.txt").unlink()
        self.assert_rejected(self.run_compare())

    def test_originally_missing_file_becoming_present_changes_coverage(self):
        (self.predictions["add"] / "d.txt").write_text("", encoding="utf-8")
        self.assert_rejected(self.run_compare())

    def test_source_prediction_change_that_alters_threshold_metrics_is_rejected(self):
        (self.predictions["cmfc"] / "a.txt").write_text("1 .5 .5 .25 .25 .1\n", encoding="utf-8")
        self.assert_rejected(self.run_compare())

    def test_report_matrix_change_is_rejected(self):
        path = self.reports[0]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["scenarios"][-1]["confusion_predicted_rows_true_columns"][0][1] += 1
        path.write_text(json.dumps(report), encoding="utf-8")
        self.assert_rejected(self.run_compare())

    def test_class_counts_are_checked_even_when_matrix_is_unchanged(self):
        path = self.reports[1]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["scenarios"][-1]["classes"][1]["gt_categories"]["classification"] += 1
        path.write_text(json.dumps(report), encoding="utf-8")
        self.assert_rejected(self.run_compare())

    def test_zero_ground_truth_metadata_mismatch_is_rejected(self):
        path = self.reports[0]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["zero_gt_images"] += 1
        path.write_text(json.dumps(report), encoding="utf-8")
        self.assert_rejected(self.run_compare())

    def test_mismatched_scenario_sets_are_rejected(self):
        path = self.reports[2]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["scenarios"].pop()
        path.write_text(json.dumps(report), encoding="utf-8")
        self.assert_rejected(self.run_compare())

    def test_duplicate_image_stems_are_rejected(self):
        nested = self.images / "nested"
        nested.mkdir()
        (nested / "a.png").write_bytes(tiny_png())
        self.assert_rejected(self.run_compare())

    def test_expected_focus_count_is_enforced(self):
        self.assert_rejected(self.run_compare("--expected-focus", "999"))

    def test_changed_gt_class_is_rejected(self):
        (self.labels / "a.txt").write_text("0 .5 .5 .25 .25\n", encoding="utf-8")
        self.assert_rejected(self.run_compare())

    def test_candidate_scores_keep_below_threshold_saved_boxes_and_absence_is_not_zero(self):
        self.assert_success(self.run_compare())
        rows = read_csv(self.output / "paired_targets.csv")
        target = next(row for row in rows if Path(row["image"]).stem == "c"
                      and float(row["confidence"]) == .25 and float(row["matching_iou"]) == .5)
        for role in ("add", "cmfc", "full"):
            self.assertEqual(target[f"{role}_state"], "as:car")
            self.assertAlmostEqual(float(target[f"{role}_best_focus_score"]), .15)
            self.assertAlmostEqual(float(target[f"{role}_best_confusing_score"]), .9)
            self.assertEqual(int(target[f"{role}_best_focus_pred_index"]), 1)
            self.assertAlmostEqual(float(target[f"{role}_best_focus_iou"]), 1.)
        absent = next(row for row in rows if Path(row["image"]).stem == "d"
                      and float(row["confidence"]) == .25 and float(row["matching_iou"]) == .5)
        for role in ("add", "cmfc", "full"):
            self.assertEqual(absent[f"{role}_best_focus_score"], "")
            self.assertEqual(absent[f"{role}_best_confusing_score"], "")
            self.assertEqual(absent[f"{role}_associated_score"], "")

    def test_missing_metadata_identifies_affected_ground_truth(self):
        self.assert_success(self.run_compare())
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        for role in ("add", "cmfc", "full"):
            self.assertEqual(summary["missing_predictions"][role],
                             [{"image": "d.png", "gt_count": 1, "gt_classes": {"van": 1}}])
            self.assertEqual(summary["source_reports"][role]["missing_prediction_files"], ["d.png"])
        self.assertEqual(set(summary["input_sha256"]), {"gt", "add", "cmfc", "full"})
        self.assertTrue(all(len(digest) == 64 for digest in summary["input_sha256"].values()))

    def test_output_inside_source_directories_is_rejected_without_writes(self):
        before = fingerprint_tree(self.inputs)
        for protected in (self.images, self.ir_images, self.labels,
                          self.predictions["add"], self.reports[0].parent):
            with self.subTest(protected=protected):
                forbidden = protected / "comparison_output"
                result = self.run_compare(output=forbidden)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(forbidden.exists())
        self.assertEqual(before, fingerprint_tree(self.inputs))

    def test_paired_images_render_end_to_end_and_deduplicate_selected_targets(self):
        from PIL import Image

        before = fingerprint_tree(self.inputs)
        self.assert_success(self.run_compare(no_images=False))
        self.assertEqual(before, fingerprint_tree(self.inputs))
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["render_errors"], [])
        selected = read_csv(self.output / "selection.csv")
        identities = {(row["image"], row["gt_index_zero_based"]) for row in selected}
        self.assertEqual(len(identities), 3)
        self.assertEqual(summary["rendered_panels"], len(identities))
        panels = {row["panel_file"] for row in selected}
        self.assertEqual(len(panels), len(identities))
        self.assertEqual(len(list((self.output / "images").glob("*.png"))), len(identities))
        for relative in panels:
            path = (self.output / relative).resolve()
            self.assertIn(self.output.resolve(), path.parents)
            self.assertTrue(path.is_file())
            with Image.open(path) as panel:
                self.assertGreater(panel.width, 32)
                self.assertGreater(panel.height, 32)
                panel.verify()


if __name__ == "__main__":
    unittest.main()
