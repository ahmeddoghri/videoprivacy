"""Property-based verification of videoprivacy's core tracking claims
across randomized multi-person scenarios.

This is not a bug fix -- videoprivacy's core.py mechanism is correct and
the published demo numbers reproduce exactly. This checks two things the
one-fixture demo can't: (1) is a real detection ever silently dropped
from the redaction report, and (2) does gap-filling actually improve
coverage over naive per-frame detection, generalized past one scenario.
"""
from __future__ import annotations

import json
import statistics as st

from .adversarial import HOLDOUT_SEEDS, TUNING_SEEDS, random_scenario
from .core import track_frames


def check_no_detection_dropped(seeds: list[int]) -> dict:
    violations = 0
    total_frames = 0
    for seed in seeds:
        frames, _ = random_scenario(seed)
        report = track_frames(frames, 1280, 720, max_gap=2)
        for inp, out in zip(frames, report):
            total_frames += 1
            real_boxes = sum(1 for item in out["redactions"] if not item["inferred"])
            if real_boxes != len(inp["detections"]):
                violations += 1
    return {"frames_checked": total_frames, "detections_ever_dropped": violations > 0, "violations": violations}


def coverage_gain(seeds: list[int]) -> dict:
    naive_ratios = []
    tracked_ratios = []
    for seed in seeds:
        frames, true_total = random_scenario(seed)
        report = track_frames(frames, 1280, 720, max_gap=2)
        naive_covered = sum(len(f["detections"]) for f in frames)
        tracked_covered = sum(len(f["redactions"]) for f in report)
        naive_ratios.append(naive_covered / true_total)
        tracked_ratios.append(tracked_covered / true_total)
    return {
        "n": len(seeds),
        "naive_mean_coverage": round(st.mean(naive_ratios), 3),
        "tracked_mean_coverage": round(st.mean(tracked_ratios), 3),
        "mean_gain": round(st.mean(tracked_ratios) - st.mean(naive_ratios), 3),
    }


def main() -> None:
    print("videoprivacy eval_v2: property-based verification across randomized multi-person scenarios")
    for label, seeds in (("tuning", TUNING_SEEDS), ("holdout", HOLDOUT_SEEDS)):
        print(f"\n{label} ({len(seeds)} seeds):")
        print("no-detection-dropped check:", json.dumps(check_no_detection_dropped(seeds)))
        print("coverage gain:", json.dumps(coverage_gain(seeds)))


if __name__ == "__main__":
    main()
