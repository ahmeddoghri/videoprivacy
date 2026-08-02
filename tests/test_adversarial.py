import unittest

from videoprivacy.adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from videoprivacy.core import DEMO, analyze
from videoprivacy.eval_v2 import check_no_detection_dropped, coverage_gain


class AdversarialTest(unittest.TestCase):
    def test_holdout_disjoint_from_tuning(self):
        self.assertTrue(set(TUNING_SEEDS).isdisjoint(HOLDOUT_SEEDS))

    def test_original_benchmark_still_reproduces_exactly(self):
        result = analyze(DEMO)
        self.assertEqual(result["unique_tracks"], 2)
        self.assertEqual(result["gap_fills"], 1)
        self.assertEqual(result["redacted_regions"], 10)

    def test_no_real_detection_is_ever_dropped_on_tuning_seeds(self):
        """Every input detection must always result in exactly one
        non-inferred box in that frame's redaction report -- the tool's
        core safety invariant for a privacy-critical redaction pipeline."""
        result = check_no_detection_dropped(TUNING_SEEDS)
        self.assertFalse(result["detections_ever_dropped"])
        self.assertEqual(result["violations"], 0)
        self.assertGreater(result["frames_checked"], 1000)

    def test_no_real_detection_is_ever_dropped_on_frozen_holdout_seeds(self):
        result = check_no_detection_dropped(HOLDOUT_SEEDS)
        self.assertFalse(result["detections_ever_dropped"])
        self.assertEqual(result["violations"], 0)

    def test_gap_filling_coverage_gain_generalizes_on_tuning_seeds(self):
        result = coverage_gain(TUNING_SEEDS)
        self.assertGreater(result["mean_gain"], 0.15)

    def test_gap_filling_coverage_gain_generalizes_on_frozen_holdout_seeds(self):
        result = coverage_gain(HOLDOUT_SEEDS)
        self.assertGreater(result["mean_gain"], 0.15)

    def test_report_is_reproducible(self):
        a = coverage_gain(TUNING_SEEDS[:5])
        b = coverage_gain(TUNING_SEEDS[:5])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
