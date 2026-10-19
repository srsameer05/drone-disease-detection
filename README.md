# Disease Drone

A two-drone precision agriculture system for automated crop disease detection and targeted treatment. A **Scout Drone** surveys fields and captures geotagged images; a **Treatment Drone** applies treatment only where disease is detected.

**Status:** Active Development (v0.3) — Full ML inference pipeline, real-time live video streaming, and MAVLink telemetry synchronization complete. Hardware integration pending.

## Project Structure

```
disease-drone/
├── ml/                         # Machine learning pipeline
│   ├── configs/                # Training & dataset YAML configs
│   ├── data/scripts/           # Data download & preprocessing
│   ├── training/               # Train & evaluate scripts
│   ├── inference/              # Detection & ONNX export
│   └── models/                 # Trained model weights (empty until training)
├── engine/                     # Core processing engine
│   ├── decision.py             # DBSCAN clustering, GPS conversion, spray zones
│   ├── planner.py              # Lawnmower scan & TSP spray path planning
│   ├── ingest.py               # EXIF GPS extraction, image processing pipeline
│   └── base_station.py         # Central orchestrator
├── drone/                      # Drone communication
│   └── comms.py                # MAVLink connection, telemetry, mission upload
├── dashboard/                  # Web dashboard (FastAPI + Leaflet.js)
│   ├── app.py                  # FastAPI backend & API routes
│   ├── database.py             # SQLite data layer
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS, JS, assets
├── architecture.md             # Full system architecture document
├── RESUME.md                   # Session resume point & progress tracker
└── requirements.txt
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv drn-env
source drn-env/bin/activate    # macOS/Linux
# drn-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## ML Pipeline

YOLOv8-nano (Ultralytics) trained on PlantVillage + PlantDoc datasets for 5-class crop disease detection.

**Classes:** `healthy`, `leaf_blight`, `leaf_spot`, `rust`, `powdery_mildew`

**Dataset:** ~57K images preprocessed into YOLO format (train: 39,810 / val: 11,372 / test: 5,693)

```bash
# Download training data (requires Kaggle API key)
python ml/data/scripts/download_data.py

# Preprocess into YOLO format
python ml/data/scripts/preprocess.py

# Train model (auto-detects MPS on Apple Silicon, CUDA on NVIDIA, or CPU fallback)
python ml/training/train.py

# Override device or hyperparameters
python ml/training/train.py --device mps --epochs 50 --batch 8

# Evaluate on test set
python ml/training/evaluate.py

# Run inference on a single image
python ml/inference/detect.py path/to/image.jpg
```

### Cross-Platform Device Support

Training automatically detects the best available hardware:

| Platform | Device | Config Value |
|----------|--------|-------------|
| macOS (Apple Silicon) | MPS GPU | `mps` |
| Windows/Linux (NVIDIA) | CUDA GPU | `0` |
| Any | CPU fallback | `cpu` |

Set `device: auto` in `ml/configs/train.yaml` (default) or override with `--device`.

## Decision Engine

Converts raw ML detections into actionable spray zones.

```python
from engine.decision import process_detections, Detection

detections = [
    Detection(lat=12.97, lon=77.59, class_name="leaf_blight", confidence=0.9),
    Detection(lat=12.97001, lon=77.59001, class_name="rust", confidence=0.85),
]

# Cluster detections and generate spray zones
zones = process_detections(detections, mission_id=1, db_module=db)
```

**Pipeline:** Filter low-confidence detections -> DBSCAN clustering (GPS coords) -> Convex hull + buffer -> Severity scoring -> Persist to DB

## Mission Planner

Generates flight paths for both scout and treatment drones.

```python
from engine.planner import generate_scan_waypoints, optimize_spray_path

# Lawnmower scan pattern
scan_area = [[12.970, 77.593], [12.970, 77.596], [12.974, 77.596], [12.974, 77.593]]
waypoints = generate_scan_waypoints(scan_area, altitude=4.0, overlap=0.3)

# Optimised spray path (nearest-neighbour TSP)
spray_wps = optimize_spray_path(spray_zones, home=(12.970, 77.593))

# Export for QGroundControl
from engine.planner import mission_to_qgc_plan
plan = mission_to_qgc_plan(waypoints)
```

## Image Ingestion

Processes drone images end-to-end: EXIF GPS extraction -> ML inference -> pixel-to-GPS conversion -> DB persistence.

```python
from engine.ingest import process_folder, watch_folder

# Batch process a folder of drone images
detections = process_folder("path/to/images/", mission_id=1, db_module=db)

# Real-time folder watcher (blocking)
watch_folder("path/to/incoming/", mission_id=1, db_module=db)
```

## Drone Communication

MAVLink interface via pymavlink for ArduPilot-based flight controllers.

```python
from drone.comms import DroneLink

drone = DroneLink("udp:127.0.0.1:14550")  # SITL simulator
# drone = DroneLink("/dev/ttyUSB0")        # Serial radio
# drone = DroneLink("tcp:192.168.1.10:5760")  # WiFi

drone.connect()
drone.upload_mission(mavlink_items)
drone.arm_and_start_mission()

telemetry = drone.get_telemetry()
print(f"Position: {telemetry.lat}, {telemetry.lon} | Battery: {telemetry.battery_remaining}%")
```

## Base Station Orchestrator

Ties all components together for the full end-to-end workflow.

```python
from engine.base_station import BaseStation

bs = BaseStation()

# Plan and export a scan mission (works without live drone)
mission_id, waypoints = bs.plan_scan_mission(scan_polygon)
bs.export_scan_plan(scan_polygon, "scan_mission.plan")

# Process scout images after flight
zones = bs.process_scout_images("path/to/images/", mission_id)

# Plan and export a spray mission from approved zones
spray_id, spray_wps = bs.plan_spray_mission(scan_mission_id=mission_id)
bs.export_spray_plan("spray_mission.plan", scan_mission_id=mission_id)
```

## Dashboard (v0.2)

Full-flow web control interface for the entire mission lifecycle: plan -> fly -> detect -> approve -> spray -> review.

```bash
uvicorn dashboard.app:app --reload --port 8000
```

Then open `http://localhost:8000`. Use **Load Demo Data** to populate, or **New Scan Mission** to start the wizard.

### Features

**Mission Management:**
- **Mission Wizard** — 4-step flow: draw area -> configure altitude/overlap/speed/FOV -> preview waypoints + stats -> launch (simulate or save)
- **Mission History panel** — browse all past missions with type/status filters, aggregated detection/zone/treatment counts
- **Mission Detail modal** — full breakdown with timeline, stats, detection breakdown by class, spray zones, QGroundControl `.plan` export
- **Real-time mission progress** — % complete shown in HUD, broadcast over WebSocket

**Live Drone Tracking:**
- **Telemetry HUD** overlay on the map — drone type, mode, altitude, speed, heading, battery (color-coded), mission progress bar
- **Animated drone marker** with heading-rotated icon
- **Breadcrumb trail** of past positions
- **Follow mode** — map auto-pans with the drone
- **WebSocket** push at 2 Hz from telemetry simulator or real drone

**Drone Connection:**
- Click the **Scout** or **Treatment** indicators in the header to open the connect modal
- Supports MAVLink over UDP (e.g. Mission Planner SITL forwarding on `127.0.0.1:14551`), TCP (WiFi), or Serial
- Real drone telemetry automatically overrides the simulator and pushes live GPS to the map and HUD

**Map Layers:**
- Toggle detections, spray zones, health heatmap, flight path, drone trail independently
- Click disease in legend to filter detections by class
- Filter detections by text search + disease dropdown

**Visualization:**
- Dark theme (CARTO dark tiles) with Leaflet.js
- Detection markers color-coded by disease, sized by confidence
- Spray zones rendered as colored polygons with status badges
- Health heatmap (green -> red gradient)
- Disease distribution chart in sidebar
- Animated stat cards with counter rollups
- Animated dashed flight path polyline

**Machine Learning & Inference Integrations:**
- **Upload & Detect:** Upload images or videos (`.mp4`) to run batch YOLO inference with dynamic confidence thresholding and area-based noise filtering.
- **Live Stream Detection:** Connect to a live MJPG feed or local webcam (URL `0`) to process and display real-time bounding boxes via WebSockets, with configurable FPS limiters and thresholds.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/stats` | Summary statistics + disease distribution |
| **Missions** | | |
| POST | `/api/missions` | Create a mission (basic) |
| GET | `/api/missions` | List missions |
| GET | `/api/missions/summary` | History list with aggregated counts (filterable) |
| GET | `/api/missions/{id}` | Get mission detail |
| GET | `/api/missions/{id}/full` | Mission with detections, zones, treatments |
| PUT | `/api/missions/{id}` | Update mission status |
| POST | `/api/missions/{id}/simulate` | Run simulated flight |
| POST | `/api/missions/{id}/upload` | Upload mission to real drone |
| POST | `/api/missions/{id}/launch` | Arm + start mission on real drone |
| **Planning** | | |
| POST | `/api/plan/scan` | Generate lawnmower waypoints + stats from polygon |
| POST | `/api/plan/spray` | Generate TSP-optimised spray path from approved zones |
| **Drone Control** | | |
| POST | `/api/drone/connect` | Connect to real drone via MAVLink |
| POST | `/api/drone/disconnect/{type}` | Disconnect scout/treatment |
| GET | `/api/drone/status` | Current connection status |
| GET | `/api/telemetry` | Latest telemetry snapshot (real or simulated) |
| WS | `/ws/telemetry` | Live telemetry + event broadcast |
| POST | `/api/simulator/stop` | Stop the telemetry simulator |
| **Live Video & Image Processing** | | |
| POST | `/detect` | Upload image for detection |
| POST | `/api/detect/video` | Upload video for detection (frame sampling + aggregation) |
| POST | `/api/stream/start` | Start live MJPG stream / webcam YOLO detection |
| POST | `/api/stream/stop` | Stop live stream |
| GET | `/api/stream/frame` | Pull latest annotated JPEG frame |
| WS | `/ws/stream-detect` | Live broadcast of stream detection results & stats |
| **Data** | | |
| GET / POST | `/api/detections` | List / add detections |
| GET / POST / PUT | `/api/spray-zones` | List / add / approve-reject zones |
| GET / POST | `/api/treatments` | List / log treatments |
| GET / POST | `/api/health` | Health heatmap data |
| **Demo** | | |
| POST | `/api/demo/seed` | Seed full demo dataset |
| POST | `/api/demo/clear` | Clear all data |
| POST | `/api/demo/full-flow` | Create planned mission ready to simulate |

## Testing Without a Real Drone

The dashboard ships with a built-in **telemetry simulator** that walks waypoint paths with realistic position interpolation, heading, battery drain, and progress tracking. No drone or SITL setup needed:

1. Start the dashboard: `uvicorn dashboard.app:app --reload --port 8000`
2. Click **New Scan Mission** → draw an area → preview → **Simulate Flight**
3. Watch the drone marker fly along the path with live HUD updates

For real ArduPilot SITL testing:

```bash
# In a separate terminal, launch ArduCopter SITL
sim_vehicle.py -v ArduCopter --console

# Then in the dashboard, click the Scout drone indicator in the header
# and connect to: udp:127.0.0.1:14550
```

## Architecture

See `architecture.md` for the full system design, hardware recommendations, data flow, and budget breakdown.

### End-to-End Flow

```
1. Operator draws scan area on dashboard
2. Mission planner generates lawnmower waypoints
3. Scout drone flies mission, captures geotagged images
4. Images transferred to base station
5. ML model runs inference on each image
6. Detections clustered into spray zones (DBSCAN + convex hull)
7. Spray zones displayed on dashboard for operator approval
8. Operator approves -> treatment drone receives optimised spray path
9. Treatment drone flies to zones, applies treatment
10. All data logged to SQLite, health map updated
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Training & Inference | PyTorch, Ultralytics YOLOv8, ONNX Runtime |
| Decision Engine | scikit-learn (DBSCAN), Shapely (geometry) |
| Mission Planning | Custom lawnmower + nearest-neighbour TSP |
| Drone Communication | pymavlink (MAVLink protocol) |
| Image Processing | OpenCV, Pillow, ExifRead |
| Backend | FastAPI, SQLite |
| Frontend | Jinja2, Leaflet.js, OpenStreetMap |

## License

This project is for educational and research purposes.
