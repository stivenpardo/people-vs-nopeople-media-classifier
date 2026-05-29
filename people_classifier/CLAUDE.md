# CLAUDE.md — People Classifier

## What this project does

Sorts a folder of phone photos and videos into two output buckets:

- `people/` — media that contains at least one **full human body** (all required keypoints visible, person large enough in frame, correct aspect ratio)
- `nopeople/` — everything else: no person, partial bodies, small/distant figures, shadows

Entry point: `python interfaces/cli.py --config config.yaml`

---

## Architecture — Clean/Hexagonal layers

```
domain/          Pure data + abstract interfaces. No external dependencies.
application/     Use-case logic. Depends only on domain abstractions.
infrastructure/  Concrete adapters (YOLO, MediaPipe, OpenCV, filesystem, Pydantic config).
interfaces/      CLI wiring. Assembles the object graph and calls application layer.
```

Dependency direction: `interfaces → application → domain ← infrastructure`

---

## Key files

| File | Purpose |
|---|---|
| `config.yaml` | All tuneable parameters — paths, thresholds, device, workers |
| `domain/entities.py` | `MediaType`, `ClassificationLabel`, `MediaFile`, `BoundingBox`, `Keypoint`, `HumanBodyStats`, `ClassificationResult` |
| `domain/interfaces.py` | `IPersonDetector`, `IPoseEstimator`, `IVideoProcessor`, `IFileSystem` |
| `infrastructure/config_loader.py` | `AppConfig` — Pydantic v2 model; validates config.yaml at startup |
| `application/classifier_service.py` | Core decision logic: detect → area check → aspect-ratio check → pose → keypoint check |
| `application/file_organizer.py` | Copies or moves files into `people/` / `nopeople/` subdirs (collision-safe) |
| `infrastructure/yolo_detector.py` | `YOLOPersonDetector` — wraps YOLOv8n, class 0 (person) only |
| `infrastructure/mediapipe_pose.py` | `MediaPipePoseEstimator` — crops to bbox + 15 % padding before running pose |
| `infrastructure/video_processor.py` | `OpenCVVideoProcessor` — samples frames at `sample_fps` |
| `infrastructure/file_system.py` | `LocalFileSystem` — shutil copy/move, recursive media discovery |
| `interfaces/cli.py` | `main()` — loads `AppConfig`, builds graph, runs parallel classify + organize |
| `evaluate.py` | Compares predicted vs ground-truth folder layout; prints sklearn report + confusion matrix |

---

## Classification logic (ClassifierService)

1. **YOLO** detects persons; filters by `confidence_threshold` (default 0.6).
2. No persons → `NO_PEOPLE`.
3. Per person — **area check**: bbox must cover ≥ `min_human_bbox_area_ratio` (5 %) of frame.
4. Per person — **aspect-ratio check**: bbox height ÷ width must be ≥ `min_bbox_aspect_ratio` (1.2). Filters out very wide/crouching boxes that are unlikely to be full-body.
5. **MediaPipe** pose runs on the **cropped region** (bbox + 15 % padding) — one crop per detected person for correct multi-person handling.
6. **Keypoint check**: all keypoints in `min_body_parts` must have MediaPipe visibility ≥ `full_body_keypoint_threshold` (0.5).
7. First person passing all four checks → `PEOPLE`. If none pass → `NO_PEOPLE`.
8. **Videos**: sampled at `video_sample_fps` (1 fps); first frame yielding `PEOPLE` → whole video goes to `people/`.

---

## Config reference (`config.yaml`)

```yaml
input_folder: "C:\\...\\Photos-2021"
output_folder: "C:\\...\\CleanDataFromv3"  # people/ and nopeople/ created inside
operation: "move"                          # "copy" or "move"
confidence_threshold: 0.6                 # YOLO min confidence
min_human_bbox_area_ratio: 0.05           # person must be ≥5% of image
min_bbox_aspect_ratio: 1.2               # height/width — filters non-standing poses
full_body_keypoint_threshold: 0.5        # MediaPipe visibility per keypoint
min_body_parts: [left_ankle, right_ankle, left_shoulder, right_shoulder, left_hip, right_hip]
video_sample_fps: 1.0
device: "cpu"                             # or "cuda"
num_workers: 4                            # ThreadPoolExecutor workers
log_level: "INFO"
```

---

## Running

```bash
pip install -r requirements.txt

# Classify
python interfaces/cli.py --config config.yaml

# Evaluate (ground-truth folder needs people/ and nopeople/ sub-dirs)
python evaluate.py --ground_truth ./gt --predicted ./output
```

---

## Known gotchas

- `AppConfig.from_yaml()` validates all fields at startup via Pydantic v2 — a bad config value raises a `ValidationError` with a clear message before any inference runs.
- `ThreadPoolExecutor` with `num_workers > 1` and `device: cuda` can cause GPU OOM — reduce `num_workers` to 1 or 2 when using CUDA.
- MediaPipe `static_image_mode=True` is required for per-image classification (vs video tracking mode).
- YOLOv8n weights download on first run (~6 MB). Ensure internet access or pre-place `yolov8n.pt` in the working directory.
