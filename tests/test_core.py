import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from videoprivacy.core import DEMO, analyze, redact_video


class VideoPrivacyTest(unittest.TestCase):
    def test_tracker_fills_one_missed_detection(self):
        result = analyze(DEMO)
        self.assertEqual(result["unique_tracks"], 2)
        self.assertEqual(result["gap_fills"], 1)
        self.assertEqual(result["redacted_regions"], 10)

    def test_redacts_a_real_synthetic_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "input.mp4", Path(directory) / "output.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
            for _ in range(3):
                writer.write(np.full((64, 64, 3), 255, dtype=np.uint8))
            writer.release()
            result = redact_video(str(source), str(target), detector=lambda _: [(16, 16, 32, 32)])
            self.assertEqual(result["frames"], 3)
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
