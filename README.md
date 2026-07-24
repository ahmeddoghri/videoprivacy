# videoprivacy

**Tracked face redaction pipeline for video.**

![videoprivacy cover](demo/cover.png)

Detect, track, merge, and blur faces across video frames with an inspectable redaction report.

![videoprivacy workbench](demo/dashboard.png)

## What ships

- A deterministic domain analysis engine with explicit scope
- JSON API and responsive local browser workbench
- CLI demo and file-driven analysis
- Docker image, unit tests, and GitHub Actions matrix
- No API keys and no uploaded user data

## Run it end to end

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
videoprivacy demo
videoprivacy redact input.mp4 redacted.mp4
videoprivacy serve
```

Open <http://127.0.0.1:8090>. Analyze your own JSON input with `videoprivacy analyze input.json`.

## API

- `GET /api/demo` returns the committed fixture and result.
- `POST /api/analyze` runs the same engine on a JSON body.

## Current basis

- [OpenCV DNN face detection](https://docs.opencv.org/master/d0/dd4/tutorial_dnn_face.html)

## Demo result

Two identities cross five frames. One face detector miss is bridged by track memory, so all ten face regions remain covered. The integration test writes and redacts a real three-frame MP4 with OpenCV.

## Scope

The default local pipeline uses OpenCV's bundled frontal-face cascade for zero-setup operation. Production deployments should replace it with a domain-tested DNN detector and measure recall across pose, lighting, age, skin tone, motion blur, and camera geometry.

## Test

```bash
python -m unittest discover -s tests -v
```

MIT licensed.
