# ByteTrack++ Person Tracker

A real-time person detection and tracking system powered by YOLOv8 (ONNX) and a custom ByteTrack++ multi-object tracker. Features Kalman filtering, appearance-based re-identification, motion estimation, and an optional robot-follow mode.

---

## Features

- **YOLOv8 ONNX inference** — fast CPU-based person detection
- **ByteTrack++ multi-object tracker** — two-stage association using high- and low-confidence detections
- **Kalman filter** — constant-velocity model for smooth bounding box prediction
- **Appearance re-ID** — HSV color histograms to recover lost track IDs after occlusion
- **Motion estimation** — classifies each tracked person as `APPROACHING`, `RECEDING`, `LEFT`, `RIGHT`, `UP`, `DOWN`, or `STABLE` using bounding box height as a depth proxy
- **Click-to-lock target** — click on any person in the frame to designate them as the primary target
- **Robot follow mode** — generates steering commands (`TURN_LEFT`, `TURN_RIGHT`, `ADVANCE`, `STOP`, `HOLD`, `SEARCH`) based on the target's position and depth
- **Trail visualization** — fading motion trail drawn for each tracked person

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- ONNXRuntime

Install dependencies:

```bash
pip install opencv-python numpy onnxruntime
```

---

## Project Structure

```
.
├── main.py               # Main tracking script
└── weights/
    └── v8_n_fp32.onnx    # YOLOv8-nano ONNX model (you must supply this)
```

---

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/bytetrack-plus-plus.git
   cd bytetrack-plus-plus
   ```

2. **Add the model weights**

   Place your YOLOv8-nano ONNX model at `./weights/v8_n_fp32.onnx`.
   You can export it from Ultralytics:
   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=onnx imgsz=320
   ```

3. **Run**
   ```bash
   python main.py
   ```

   The script opens your default webcam (`/dev/video0` / camera index `0`).

---

## Controls

| Key / Action | Effect |
|---|---|
| `Q` | Quit |
| `R` | Toggle robot follow mode on/off |
| Left click on a person | Lock that person as the tracking target |

---

## Configuration

All tunable parameters are at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `./weights/v8_n_fp32.onnx` | Path to ONNX model |
| `IMG_SIZE` | `320` | Inference resolution (square) |
| `CONF_THRES` | `0.5` | High-confidence detection threshold |
| `NMS_THRES` | `0.45` | Non-maximum suppression IoU threshold |
| `ROBOT_DEADZONE_X` | `0.15` | Horizontal deadzone fraction before steering |
| `ROBOT_DEADZONE_DEPTH` | `0.10` | Depth deadzone before advancing/stopping |

---

## How It Works

### Detection
Each frame is letterbox-resized to `IMG_SIZE × IMG_SIZE` and run through the YOLOv8 ONNX model. Outputs are filtered to the `person` class (class ID `0`) and NMS is applied.

### Tracking (ByteTrack++)
1. Detections are split into **high-confidence** (≥ 0.5) and **low-confidence** (0.1–0.5) sets.
2. High-confidence detections are matched to existing tracks via IoU against Kalman-predicted boxes.
3. Unmatched tracks are then offered to low-confidence detections.
4. Still-unmatched detections are compared against **lost tracks** using HSV histogram similarity for re-ID before spawning new track IDs.

### Robot Follow Mode
When enabled, the system computes steering commands based on:
- **Horizontal offset** of the target's center → `TURN_LEFT` / `TURN_RIGHT`
- **Bounding box height ratio** as a depth proxy → `ADVANCE` / `STOP`

Commands are displayed on the HUD and printed to stdout.

---

## License

MIT
