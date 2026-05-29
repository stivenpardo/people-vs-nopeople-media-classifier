# People vs No-People Media Classifier

Binary classifier that sorts phone photos and videos into:

- **`people/`** — media containing at least one **complete, full human body** (head to feet, all major limbs visible)
- **`nopeople/`** — everything else: no person, partial bodies, shadows, distant/tiny figures, hands-only, animals, landscapes

## Requirements

- Python 3.10+
- CPU or NVIDIA GPU (CUDA optional)

## Installation

```bash
pip install -r requirements.txt
```

On first run, YOLOv8 nano weights (`yolov8n.pt`) are downloaded automatically (~6 MB).

## Step-by-step usage guide

### Step 1 — Verify Python version

```bash
python --version
```

Requires Python 3.10 or newer. Install from [python.org](https://python.org) if needed.

---

### Step 2 — Navigate to the project folder

```bash
cd C:\Users\michael.pardob\projects\image-database-cleanup-v3\people_classifier
```

---

### Step 3 — Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Your prompt will change to `(venv)` confirming the environment is active.

---

### Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

Takes 1–3 minutes. YOLOv8 nano weights (`yolov8n.pt`, ~6 MB) download automatically on the **first run**, not during install.

---

### Step 5 — Prepare your input folder

Place your photos and videos in any folder. The default config points to:

```
C:\Users\michael.pardob\Downloads\Photos-2021
```

Sub-folders are scanned recursively. Supported formats:

| Type | Extensions |
|---|---|
| Images | `.jpg` `.jpeg` `.png` `.bmp` `.tiff` |
| Videos | `.mp4` `.mov` `.avi` `.mkv` `.3gp` |

---

### Step 6 — Review `config.yaml`

Open `config.yaml` and verify at minimum:

```yaml
input_folder: "C:\\Users\\michael.pardob\\Downloads\\Photos-2021"
output_folder: "C:\\Users\\michael.pardob\\Downloads\\CleanDataFromv3"
operation: "copy"   # use "copy" on first run to keep originals safe
```

> **Tip:** Keep `operation: "copy"` until you are happy with the results. Switch to `"move"` once you have verified the output.

---

### Step 7 — Run the classifier

Always run from inside the `people_classifier` folder. Two equivalent commands:

```powershell
# Recommended — project root entry point
python main.py

# Alternative — run the CLI module directly
python .\interfaces\cli.py
```

To use a custom config file:

```powershell
python main.py --config path\to\my_config.yaml
```

You will see a progress bar followed by a summary:

```
Found 1842 media files in C:\...\Photos-2021
Classifying: 100%|████████████| 1842/1842 [08:34<00:00]

Summary  →  people: 612  |  nopeople: 1230  |  total: 1842
```

> **Important:** Always run from the `people_classifier` folder, not from a sub-folder.
> Running `python interfaces\cli.py` from inside `interfaces\` will cause a `ModuleNotFoundError`.

---

### Step 8 — Check the output

Open your `output_folder`. You will find:

```
CleanDataFromv3\
├── people\       ← full-body human photos/videos
└── nopeople\     ← everything else
```

Browse a sample from each folder before doing anything permanent with the files.

---

### Step 9 (optional) — Tune thresholds

**Too many false negatives** (real full-body photos in `nopeople/`) → loosen the filters:

```yaml
confidence_threshold: 0.5         # was 0.6
min_human_bbox_area_ratio: 0.03   # was 0.05
min_bbox_aspect_ratio: 1.0        # was 1.2
full_body_keypoint_threshold: 0.4 # was 0.5
```

**Too many false positives** (partial bodies in `people/`) → tighten them in the opposite direction.

Re-run after each change. Use `"copy"` mode so previous output is not overwritten.

---

### Step 10 (optional) — Evaluate accuracy

Hand-label a sample of 50–100 files, then run:

**1. Create a ground-truth folder:**

```
gt\
├── people\     ← manually confirmed full-body photos
└── nopeople\   ← manually confirmed non-full-body photos
```

**2. Run the evaluation script:**

```bash
python evaluate.py \
  --ground_truth .\gt \
  --predicted "C:\Users\michael.pardob\Downloads\CleanDataFromv3" \
  --output confusion_matrix.png
```

Prints precision / recall / F1 per class and saves a confusion-matrix PNG.

---

### Quick reference

| Goal | Command |
|---|---|
| Run classifier (default config) | `python main.py` |
| Run with a custom config | `python main.py --config path\to\my_config.yaml` |
| Run via CLI module directly | `python .\interfaces\cli.py` |
| Evaluate results | `python evaluate.py --ground_truth .\gt --predicted .\output` |

## Configuration (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `input_folder` | `./input` | Source folder (scanned recursively) |
| `output_folder` | — | Destination; `people/` and `nopeople/` created inside |
| `operation` | `move` | `copy` or `move` source files |
| `confidence_threshold` | `0.6` | YOLO minimum detection confidence |
| `min_human_bbox_area_ratio` | `0.05` | Person bounding box must cover ≥ 5 % of the frame |
| `min_bbox_aspect_ratio` | `1.2` | Height ÷ width of bounding box (filters out very wide/crouching detections) |
| `full_body_keypoint_threshold` | `0.5` | MediaPipe visibility score each required keypoint must exceed |
| `min_body_parts` | ankles, shoulders, hips | Keypoints that must all be visible |
| `video_sample_fps` | `1.0` | Frames per second to sample from videos |
| `device` | `cpu` | `cpu` or `cuda` |
| `num_workers` | `4` | Parallel classification threads |
| `log_level` | `INFO` | Python logging level |

## Classification pipeline

```
Image / Video frame
      │
      ▼
 YOLOv8-nano ──► no person detected  ──► nopeople/
      │
      ▼ (per detected person)
 Area check (≥5% of frame) ──► too small  ──► nopeople/
      │
 Aspect-ratio check (H/W ≥ 1.2) ──► wrong shape  ──► nopeople/
      │
 MediaPipe Pose (cropped to bbox + 15% padding)
      │
 Keypoint visibility check (6 keypoints ≥ 0.5) ──► missing  ──► nopeople/
      │
      ▼
   people/
```

For videos, frames are sampled at `video_sample_fps`. The **first frame** that contains a full human body classifies the entire video as `people/`.

## Evaluation

```bash
python evaluate.py \
  --ground_truth ./labelled_dataset \
  --predicted    ./output \
  --output       confusion_matrix.png
```

Both folders must contain `people/` and `nopeople/` sub-directories with the same filenames. Prints precision / recall / F1 per class and saves a confusion-matrix PNG.

## Architecture

```
domain/          Pure entities + abstract interfaces (no external deps)
application/     Use-case logic — classifier_service, file_organizer
infrastructure/  Adapters — YOLOv8, MediaPipe, OpenCV, LocalFileSystem, Pydantic config
interfaces/      CLI entry point (interfaces/cli.py)
```

## Supported file types

| Type | Extensions |
|---|---|
| Images | `.jpg` `.jpeg` `.png` `.bmp` `.tiff` |
| Videos | `.mp4` `.mov` `.avi` `.mkv` `.3gp` |
