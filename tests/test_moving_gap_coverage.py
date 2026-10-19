"""A filled gap has to cover where the face *is*, not where it was."""
import unittest

from videoprivacy.core import track_frames


def _straight_line(speed: int, missing: set[int], frames: int = 12, size: int = 48):
    return [
        {"frame": i, "detections": [] if i in missing else [[20 + i * speed, 90, size, size]]}
        for i in range(frames)
    ]


def _covers(box, target) -> bool:
    bx, by, bw, bh = box
    tx, ty, tw, th = target
    return bx <= tx and by <= ty and bx + bw >= tx + tw and by + bh >= ty + th


class MovingGapCoverageTest(unittest.TestCase):
    def test_a_moving_face_stays_covered_through_a_gap(self):
        speed, missing = 8, {7, 8}
        report = track_frames(_straight_line(speed, missing), 320, 240, max_gap=2)
        for index in sorted(missing):
            boxes = [item["box"] for item in report[index]["redactions"]]
            self.assertTrue(boxes, f"frame {index} had no redaction at all")
            true_box = (20 + index * speed, 90, 48, 48)
            self.assertTrue(
                any(_covers(box, true_box) for box in boxes),
                f"frame {index}: {boxes} does not cover {true_box}",
            )

    def test_a_still_face_does_not_get_a_needlessly_inflated_box(self):
        report = track_frames(_straight_line(0, {7}), 320, 240, max_gap=2)
        filled = report[7]["redactions"][0]["box"]
        seen = report[6]["redactions"][0]["box"]
        self.assertEqual(filled, seen)

    def test_beyond_max_gap_the_track_is_dropped_rather_than_guessed_forever(self):
        report = track_frames(_straight_line(8, {7, 8, 9}), 320, 240, max_gap=2)
        self.assertEqual(report[9]["redactions"], [])


if __name__ == "__main__":
    unittest.main()
