from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


Box = tuple[int, int, int, int]

DEMO = {
    "frame_width": 1280,
    "frame_height": 720,
    "max_gap": 2,
    "frames": [
        {"frame": 0, "detections": [[120, 180, 110, 110], [810, 170, 105, 105]]},
        {"frame": 1, "detections": [[132, 181, 110, 110], [798, 172, 105, 105]]},
        {"frame": 2, "detections": [[145, 182, 110, 110]]},
        {"frame": 3, "detections": [[157, 184, 110, 110], [775, 175, 105, 105]]},
        {"frame": 4, "detections": [[170, 185, 110, 110], [762, 177, 105, 105]]},
    ],
}


def _iou(left: Box, right: Box) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1, x2, y2 = max(lx, rx), max(ly, ry), min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def _clip(box: Box, width: int, height: int, padding: float = 0.12) -> Box:
    x, y, w, h = box
    px, py = round(w * padding), round(h * padding)
    left, top = max(0, x - px), max(0, y - py)
    right, bottom = min(width, x + w + px), min(height, y + h + py)
    return left, top, right - left, bottom - top


def track_frames(frames: list[dict[str, Any]], width: int, height: int, max_gap: int = 2) -> list[dict[str, Any]]:
    active: dict[int, dict[str, Any]] = {}
    next_id = 1
    report = []
    for frame in frames:
        index = int(frame["frame"])
        detections = [_clip(tuple(map(int, box)), width, height) for box in frame.get("detections", [])]
        assigned: set[int] = set()
        for detection in detections:
            matches = [(track_id, _iou(detection, item["box"])) for track_id, item in active.items() if track_id not in assigned]
            track_id, overlap = max(matches, key=lambda item: item[1], default=(0, 0.0))
            if overlap < 0.2:
                track_id, next_id = next_id, next_id + 1
            active[track_id] = {"box": detection, "last_seen": index}
            assigned.add(track_id)
        active = {track_id: item for track_id, item in active.items() if index - item["last_seen"] <= max_gap}
        boxes = [{"track_id": track_id, "box": list(item["box"]), "inferred": item["last_seen"] != index} for track_id, item in sorted(active.items())]
        report.append({"frame": index, "redactions": boxes})
    return report


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    frames = payload.get("frames", [])
    width, height = int(payload.get("frame_width", 0)), int(payload.get("frame_height", 0))
    if not frames or width <= 0 or height <= 0:
        raise ValueError("frames and positive frame dimensions are required")
    report = track_frames(frames, width, height, int(payload.get("max_gap", 2)))
    track_ids = {item["track_id"] for frame in report for item in frame["redactions"]}
    inferred = sum(item["inferred"] for frame in report for item in frame["redactions"])
    return {
        "frames": len(report),
        "unique_tracks": len(track_ids),
        "redacted_regions": sum(len(frame["redactions"]) for frame in report),
        "gap_fills": inferred,
        "report": report,
        "scope": "Detector output is tracked and blurred; detection recall must be evaluated for each camera domain.",
    }


def redact_video(input_path: str, output_path: str, detector: Callable[[Any], list[Box]] | None = None) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if detector is None:
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        detector = lambda image: [tuple(map(int, box)) for box in cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(30, 30))]
    active_frames = []
    raw_frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        raw_frames.append(image)
        active_frames.append({"frame": len(active_frames), "detections": [list(box) for box in detector(image)]})
    tracked = track_frames(active_frames, width, height)
    for image, frame in zip(raw_frames, tracked):
        for item in frame["redactions"]:
            x, y, w, h = item["box"]
            region = image[y:y + h, x:x + w]
            if region.size:
                kernel = max(9, (min(w, h) // 4) | 1)
                image[y:y + h, x:x + w] = cv2.GaussianBlur(region, (kernel, kernel), 0)
        writer.write(image)
    capture.release()
    writer.release()
    return {"input": input_path, "output": output_path, "frames": len(raw_frames), "report": analyze({"frames": active_frames, "frame_width": width, "frame_height": height})}
