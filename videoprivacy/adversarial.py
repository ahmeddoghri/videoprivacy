"""Adversarial scenarios for the tracker's core coverage claim.

videoprivacy.py's mechanism is correct and the published demo numbers
reproduce exactly. This checks whether the claim generalizes past the
one five-frame, two-person fixture the demo ships with: does gap-filling
actually improve coverage over naive per-frame detection across many
randomized multi-person tracking scenarios, and is the "every detected
face gets redacted" invariant ever violated?

TUNING_SEEDS: used to characterize the tracker's behavior.
HOLDOUT_SEEDS: disjoint, evaluated exactly once after characterization.
"""

TUNING_SEEDS = list(range(1, 41))
HOLDOUT_SEEDS = list(range(1000, 1030))


def random_scenario(seed: int, n_frames: int = 30, n_people: int = 3, width: int = 1280, height: int = 720, miss_rate: float = 0.2):
    """Simulate n_people walking with continuous motion and a per-frame
    chance of a missed detection (a detector "blink"), the exact failure
    mode this tool exists to bridge."""
    import random as _random

    rng = _random.Random(seed)
    people = []
    for _ in range(n_people):
        x, y = rng.uniform(0, width - 150), rng.uniform(0, height - 150)
        vx, vy = rng.uniform(-15, 15), rng.uniform(-15, 15)
        people.append([x, y, vx, vy])
    frames = []
    for f in range(n_frames):
        detections = []
        for person in people:
            person[0] += person[2]
            person[1] += person[3]
            if rng.random() >= miss_rate:
                w, h = rng.uniform(80, 130), rng.uniform(80, 130)
                detections.append([int(person[0]), int(person[1]), int(w), int(h)])
        frames.append({"frame": f, "detections": detections})
    return frames, n_people * n_frames
