"""
Disease Drone Dashboard — FastAPI Backend

Endpoints:
    GET  /                              → Dashboard UI
    GET  /api/stats                     → Dashboard summary stats

    POST /api/missions                  → Create a mission (basic)
    GET  /api/missions                  → List missions
    GET  /api/missions/summary          → Mission history with aggregated counts
    GET  /api/missions/{id}             → Get mission detail (basic)
    GET  /api/missions/{id}/full        → Get mission with detections, zones, treatments
    PUT  /api/missions/{id}             → Update mission status

    POST /api/plan/scan                 → Plan scan waypoints from polygon
    POST /api/plan/spray                → Plan spray waypoints from approved zones
    POST /api/missions/{id}/simulate    → Simulate mission execution (no drone needed)

    GET  /api/detections                → List detections
    POST /api/detections                → Add detection(s)
    GET  /api/spray-zones               → List spray zones
    POST /api/spray-zones               → Add spray zone
    PUT  /api/spray-zones/{id}          → Approve/reject zone
    GET  /api/treatments                → List treatments
    POST /api/treatments                → Log treatment
    GET  /api/health                    → Health heatmap data
    POST /api/health                    → Add health point

    GET  /api/telemetry                 → Snapshot of current drone telemetry
    WS   /ws/telemetry                  → Live telemetry stream

    POST /api/stream/start              → Start live MJPG stream detection
    POST /api/stream/stop               → Stop live stream
    GET  /api/stream/status             → Stream status
    GET  /api/stream/frame              → Latest annotated JPEG frame
    GET  /api/stream/snapshot           → Save current frame as detection
    WS   /ws/stream-detect              → Live detection results stream

    POST /api/demo/seed                 → Seed demo data
    POST /api/demo/clear                → Clear all data
    POST /api/demo/full-flow            → Run full end-to-end demo (scan→detect→spray)

Usage:
    uvicorn dashboard.app:app --reload --port 8000
"""

import asyncio
import base64
import io
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

# Make engine importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from . import database as db
from engine.decision import (
    Detection,
    process_detections,
    cluster_detections,
    generate_spray_zones,
    DISEASE_SEVERITY_WEIGHTS,
)
from engine.planner import (
    generate_scan_waypoints,
    optimize_spray_path,
    to_mavlink_mission,
    mission_to_qgc_plan,
    mission_stats,
)
from ml.inference.detect import detect_diseases, detect_batch

from .services.telemetry import telemetry_service
from .api.telemetry import router as telemetry_router

# Resolve model weights path (prefer ONNX for 3x faster inference, fallback to best.pt)
ONNX_WEIGHTS = PROJECT_ROOT / "ml" / "models" / "best.onnx"
MODEL_WEIGHTS = PROJECT_ROOT / "ml" / "models" / "best.pt"
if not MODEL_WEIGHTS.exists():
    MODEL_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
MODEL_PATH = str(ONNX_WEIGHTS) if ONNX_WEIGHTS.exists() else str(MODEL_WEIGHTS)

# Test images directory for demo inference
TEST_IMAGES_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "test" / "images"

# Upload directory for user-submitted images
UPLOAD_DIR = PROJECT_ROOT / "dashboard" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Snapshot directory for stream captures
SNAPSHOT_DIR = PROJECT_ROOT / "dashboard" / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)


# ── Telemetry Simulator (no drone needed for demos) ──

class TelemetrySimulator:
    """Simulates drone flying along a waypoint path.

    Used when no real drone is connected — allows the dashboard's live
    tracking features to work for demos and development.

    Supports pause, resume, stop, abort and a speed multiplier so the
    operator can fast-forward long flights.
    """

    BASE_SPEED = 2.0   # m/s, multiplied by speed_multiplier

    def __init__(self):
        self.active = False
        self.paused = False
        self.aborted = False
        self.mission_id = None
        self.drone_type = "scout"
        self.waypoints = []
        self.segment_distances = []     # cached metres per segment
        self.total_distance = 0.0
        self.current_idx = 0
        self.position = None
        self.alt = 0.0
        self.heading = 0.0
        self.battery = 100
        self.speed = 0.0
        self.speed_multiplier = 1.0
        self.mode = "STANDBY"
        self.armed = False
        self.progress = 0.0
        self.elapsed_s = 0.0             # sim seconds since start
        self.distance_traveled = 0.0
        self._segment_progress = 0.0

    def start(self, mission_id: int, waypoints: list, drone_type: str = "scout"):
        """Begin simulating a mission."""
        # Reset everything
        self.reset()
        self.active = True
        self.paused = False
        self.aborted = False
        self.mission_id = mission_id
        self.drone_type = drone_type
        self.waypoints = [
            (wp["x"], wp["y"], wp["z"]) for wp in waypoints
            if wp.get("command") in (16, 22)  # waypoint or takeoff
        ]
        self.current_idx = 0
        self._segment_progress = 0.0
        self.battery = 100
        self.speed = self.BASE_SPEED * self.speed_multiplier
        self.mode = "AUTO"
        self.armed = True
        self.progress = 0.0
        self.elapsed_s = 0.0
        self.distance_traveled = 0.0
        if self.waypoints:
            self.position = (self.waypoints[0][0], self.waypoints[0][1])
            self.alt = self.waypoints[0][2]

        # Pre-compute segment distances for ETA
        self._compute_segment_distances()

    def _compute_segment_distances(self):
        self.segment_distances = []
        self.total_distance = 0.0
        for i in range(len(self.waypoints) - 1):
            a, b = self.waypoints[i], self.waypoints[i + 1]
            mlat = 111_320
            mlon = 111_320 * math.cos(math.radians(a[0]))
            d = math.sqrt(((b[0] - a[0]) * mlat) ** 2 + ((b[1] - a[1]) * mlon) ** 2)
            self.segment_distances.append(d)
            self.total_distance += d

    def reset(self):
        """Reset all flight state — used after stop/abort/clear."""
        self.active = False
        self.paused = False
        self.aborted = False
        self.mission_id = None
        self.waypoints = []
        self.segment_distances = []
        self.total_distance = 0.0
        self.current_idx = 0
        self.position = None
        self.alt = 0.0
        self.heading = 0.0
        self.battery = 100
        self.speed = 0.0
        self.mode = "STANDBY"
        self.armed = False
        self.progress = 0.0
        self.elapsed_s = 0.0
        self.distance_traveled = 0.0
        self._segment_progress = 0.0

    def stop(self):
        """Graceful stop (mission complete)."""
        self.active = False
        self.mode = "STANDBY"
        self.armed = False
        self.speed = 0.0

    def abort(self):
        """Emergency abort — mission marked as aborted."""
        self.active = False
        self.paused = False
        self.aborted = True
        self.mode = "ABORTED"
        self.armed = False
        self.speed = 0.0

    def pause(self):
        if self.active:
            self.paused = True
            self.mode = "PAUSED"
            self.speed = 0.0

    def resume(self):
        if self.active and self.paused:
            self.paused = False
            self.mode = "AUTO"
            self.speed = self.BASE_SPEED * self.speed_multiplier

    def set_speed_multiplier(self, mult: float):
        mult = max(0.25, min(mult, 16.0))
        self.speed_multiplier = mult
        if self.active and not self.paused:
            self.speed = self.BASE_SPEED * mult

    def step(self, dt: float = 0.5):
        """Advance simulation by `dt` seconds.

        Consumes as many segments as fit into the dt budget. This is important
        for lawnmower patterns where the row-turn segments are very short
        (often <5m) and a single 0.5s tick at 8x speed can span 2-3 segments.
        Without multi-segment consumption, the drone would stutter at every
        row turn.
        """
        if not self.active or self.paused or not self.waypoints:
            return
        if self.current_idx >= len(self.waypoints) - 1:
            self.stop()
            self.progress = 1.0
            return

        self.elapsed_s += dt

        # Total distance to cover this tick
        budget = self.speed * dt

        # Consume segments until budget runs out or we hit the end
        while budget > 0 and self.current_idx < len(self.waypoints) - 1:
            a = self.waypoints[self.current_idx]
            b = self.waypoints[self.current_idx + 1]

            seg_dist = (
                self.segment_distances[self.current_idx]
                if self.current_idx < len(self.segment_distances) else 0.1
            )
            seg_dist = max(seg_dist, 0.1)

            seg_remaining_m = seg_dist * (1.0 - self._segment_progress)

            if budget >= seg_remaining_m:
                # Finish this segment, advance to next
                self.distance_traveled += seg_remaining_m
                budget -= seg_remaining_m
                self.current_idx += 1
                self._segment_progress = 0.0
                self.position = (b[0], b[1])
                self.alt = b[2]
                self._update_heading(a, b)

                # If we just consumed the final segment, stop immediately
                # so the loop broadcasts the completion on this same tick.
                if self.current_idx >= len(self.waypoints) - 1:
                    self.distance_traveled = self.total_distance
                    self.progress = 1.0
                    self.stop()
                    return
            else:
                # Partial advance within this segment
                t_advance = budget / seg_dist
                self._segment_progress = min(1.0, self._segment_progress + t_advance)
                t = self._segment_progress
                self.position = (
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                )
                self.alt = a[2] + (b[2] - a[2]) * t
                self._update_heading(a, b)
                self.distance_traveled += budget
                budget = 0

        # Battery drain (scales with speed multiplier)
        self.battery = max(15, self.battery - 0.05 * self.speed_multiplier)

        # Overall progress
        if self.total_distance > 0:
            self.progress = min(1.0, self.distance_traveled / self.total_distance)
        elif len(self.waypoints) > 1:
            self.progress = (self.current_idx + self._segment_progress) / (len(self.waypoints) - 1)

    def _update_heading(self, a, b):
        """Update heading based on the segment direction."""
        mlat = 111_320
        mlon = 111_320 * math.cos(math.radians(a[0]))
        dx = (b[1] - a[1]) * mlon
        dy = (b[0] - a[0]) * mlat
        if abs(dx) > 0.01 or abs(dy) > 0.01:    # avoid NaN on zero-length segments
            self.heading = (math.degrees(math.atan2(dx, dy)) + 360) % 360

    def distance_to_next_wp(self) -> float:
        if not self.position or self.current_idx >= len(self.waypoints) - 1:
            return 0.0
        b = self.waypoints[self.current_idx + 1]
        mlat = 111_320
        mlon = 111_320 * math.cos(math.radians(self.position[0]))
        return math.sqrt(
            ((b[0] - self.position[0]) * mlat) ** 2 +
            ((b[1] - self.position[1]) * mlon) ** 2
        )

    def eta_s(self) -> float:
        """Estimated time remaining in seconds (real time, accounting for speed mult)."""
        remaining_m = self.total_distance - self.distance_traveled
        if remaining_m <= 0 or self.speed <= 0:
            return 0.0
        return remaining_m / self.speed

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "paused": self.paused,
            "aborted": self.aborted,
            "mission_id": self.mission_id,
            "drone_type": self.drone_type,
            "lat": round(self.position[0], 7) if self.position else None,
            "lon": round(self.position[1], 7) if self.position else None,
            "alt": round(self.alt, 2),
            "heading": round(self.heading, 1),
            "battery": round(self.battery, 1),
            "groundspeed": round(self.speed, 2),
            "speed_multiplier": self.speed_multiplier,
            "mode": self.mode,
            "armed": self.armed,
            "progress": round(self.progress, 4),
            "waypoint_index": self.current_idx,
            "waypoint_count": len(self.waypoints),
            "elapsed_s": round(self.elapsed_s, 1),
            "eta_s": round(self.eta_s(), 1),
            "distance_traveled_m": round(self.distance_traveled, 1),
            "total_distance_m": round(self.total_distance, 1),
            "distance_to_next_m": round(self.distance_to_next_wp(), 1),
        }


telemetry_sim = TelemetrySimulator()


# ── Drone Controller (real drone or simulator) ──

class DroneController:
    """Unified abstraction over the real DroneLink and the simulator.

    The dashboard talks only to this object — the underlying source can be
    switched at runtime via /api/drone/connect (real) or /api/simulator/stop
    (sim). When no real drone is connected, the simulator is used so the
    dashboard remains fully functional for development and demos.
    """

    def __init__(self):
        self.scout_link = None       # DroneLink for scout
        self.treatment_link = None   # DroneLink for treatment
        self.scout_conn_str = None
        self.treatment_conn_str = None

    def is_real_connected(self) -> bool:
        return self.scout_link is not None or self.treatment_link is not None

    def connect_scout(self, connection_string: str):
        from drone.comms import DroneLink
        if self.scout_link:
            try:
                self.scout_link.close()
            except Exception:
                pass
        link = DroneLink(connection_string)
        link.connect(timeout=15)
        link.request_data_stream(rate_hz=4)
        self.scout_link = link
        self.scout_conn_str = connection_string

    def connect_treatment(self, connection_string: str):
        from drone.comms import DroneLink
        if self.treatment_link:
            try:
                self.treatment_link.close()
            except Exception:
                pass
        link = DroneLink(connection_string)
        link.connect(timeout=15)
        link.request_data_stream(rate_hz=4)
        self.treatment_link = link
        self.treatment_conn_str = connection_string

    def disconnect_scout(self):
        if self.scout_link:
            try:
                self.scout_link.close()
            except Exception:
                pass
        self.scout_link = None
        self.scout_conn_str = None

    def disconnect_treatment(self):
        if self.treatment_link:
            try:
                self.treatment_link.close()
            except Exception:
                pass
        self.treatment_link = None
        self.treatment_conn_str = None

    def upload_mission_to(self, drone_type: str, mission_items: list):
        link = self.scout_link if drone_type == "scout" else self.treatment_link
        if link is None:
            raise RuntimeError(f"{drone_type} drone not connected")
        link.upload_mission(mission_items)

    def start_mission_on(self, drone_type: str):
        link = self.scout_link if drone_type == "scout" else self.treatment_link
        if link is None:
            raise RuntimeError(f"{drone_type} drone not connected")
        link.arm_and_start_mission()

    def get_real_telemetry(self, drone_type: str = "scout") -> dict | None:
        """Read telemetry from a real drone if connected."""
        link = self.scout_link if drone_type == "scout" else self.treatment_link
        if link is None:
            return None
        try:
            t = link.get_telemetry()
            return {
                "active": t.armed,
                "drone_type": drone_type,
                "lat": t.lat,
                "lon": t.lon,
                "alt": t.alt,
                "heading": t.heading,
                "groundspeed": t.groundspeed,
                "battery": t.battery_remaining if t.battery_remaining >= 0 else 100,
                "mode": t.mode,
                "armed": t.armed,
                "progress": 0.0,         # not tracked from real drone
                "waypoint_index": 0,
                "waypoint_count": 0,
                "source": "real",
            }
        except Exception as e:
            print(f"DroneController.get_real_telemetry error: {e}")
            return None

    def status(self) -> dict:
        return {
            "scout_connected": self.scout_link is not None,
            "scout_connection": self.scout_conn_str,
            "treatment_connected": self.treatment_link is not None,
            "treatment_connection": self.treatment_conn_str,
            "simulator_active": telemetry_sim.active,
        }


drone_ctrl = DroneController()


# ── WebSocket Connection Manager ──

class WSConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSConnectionManager()


# ── Live Stream Detection Manager ──

class StreamDetectManager:
    """Dedicated WS manager for the stream-detect WebSocket clients."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


stream_ws_manager = StreamDetectManager()


class StreamDetector:
    """Captures an MJPG stream in a background thread and runs YOLO inference.

    Architecture:
    - A daemon thread reads frames from the MJPG stream via OpenCV.
    - Every `frame_skip`-th frame is passed through the YOLO model.
    - Detection results are queued for the async broadcast loop.
    - The latest annotated frame (YOLO .plot()) is cached as JPEG bytes
      for serving via GET /api/stream/frame.
    - Handles stream disconnects gracefully with auto-retry.

    The async broadcast loop runs as an asyncio task and pushes detection
    results to all connected WebSocket clients.
    """

    def __init__(self):
        self.active = False
        self.stream_url = ""
        self.frame_skip = 5
        self.conf_threshold = 0.6
        self.model = None
        self._thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._broadcast_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Thread-safe shared state
        self._lock = threading.Lock()
        self._raw_frame = None
        self._raw_frame_id = 0
        self._latest_frame_jpg: bytes | None = None      # raw camera frame
        self._latest_annotated_jpg: bytes | None = None   # YOLO-annotated frame
        self._latest_detections: list[dict] = []
        self._detection_queue: list[dict] = []             # pending broadcasts
        self._frame_count = 0
        self._processed_count = 0
        self._fps = 0.0
        self._last_error: str | None = None
        self._started_at: float = 0.0

    def start(self, stream_url: str, frame_skip: int = 5,
              conf_threshold: float = 0.6, loop=None):
        """Start capturing and processing the stream."""
        if self.active:
            self.stop()

        self.stream_url = stream_url
        self.frame_skip = max(1, frame_skip)
        self.conf_threshold = conf_threshold
        self._stop_event.clear()
        self._frame_count = 0
        self._processed_count = 0
        self._fps = 0.0
        self._last_error = None
        self._started_at = time.time()
        self._loop = loop or asyncio.get_event_loop()

        # Load model once (expensive)
        if self.model is None:
            print(f"[StreamDetector] Loading YOLO model: {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)

        self.active = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        self._inference_thread.start()

        # Start async broadcast loop
        self._broadcast_task = asyncio.ensure_future(
            self._broadcast_loop(), loop=self._loop
        )
        print(f"[StreamDetector] Started on {stream_url} (skip={frame_skip}, conf={conf_threshold})")

    def stop(self):
        """Stop capturing."""
        self.active = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=3)
        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()
        self._thread = None
        self._inference_thread = None
        self._broadcast_task = None
        with self._lock:
            self._raw_frame = None
            self._latest_frame_jpg = None
            self._latest_annotated_jpg = None
            self._latest_detections = []
            self._detection_queue = []
        print("[StreamDetector] Stopped")

    def update_settings(self, frame_skip: int | None = None,
                        conf_threshold: float | None = None):
        if frame_skip is not None:
            self.frame_skip = max(1, frame_skip)
        if conf_threshold is not None:
            self.conf_threshold = max(0.1, min(0.99, conf_threshold))

    def _capture_loop(self):
        """Background thread: strictly read frames from stream to avoid buffering backlog."""
        retry_delay = 1.0
        max_retry_delay = 10.0

        while not self._stop_event.is_set():
            url_str = str(self.stream_url).strip()
            if url_str.isdigit():
                source = int(url_str)
            elif url_str.lower() in ("webcam", "camera"):
                source = 0
            else:
                source = self.stream_url

            if isinstance(source, int) and sys.platform.startswith("win"):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Disable OpenCV buffering
            
            if not cap.isOpened():
                with self._lock:
                    self._last_error = f"Cannot open stream: {self.stream_url}"
                print(f"[StreamDetector] {self._last_error}, retrying in {retry_delay}s")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
                continue

            retry_delay = 1.0  # reset on successful connect
            print(f"[StreamDetector] Connected to {self.stream_url}")

            while not self._stop_event.is_set():
                cap.grab()  # Always extract latest frame to clear internal driver buffers
                ret, frame = cap.read()
                if not ret:
                    with self._lock:
                        self._last_error = "Stream read failed — reconnecting"
                    print(f"[StreamDetector] {self._last_error}")
                    break  # reconnect

                with self._lock:
                    self._raw_frame = frame
                    self._raw_frame_id += 1
                    self._last_error = None

            cap.release()
            if not self._stop_event.is_set():
                print("[StreamDetector] Stream disconnected, retrying...")
                time.sleep(retry_delay)

        self.active = False

    def _inference_loop(self):
        """Background thread: grabs latest frame, runs YOLO, handles frame skipping."""
        fps_timer = time.time()
        fps_frame_count = 0
        last_processed_id = -1
        
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._raw_frame
                frame_id = self._raw_frame_id
            
            if frame is None or frame_id == last_processed_id:
                time.sleep(0.01)
                continue
                
            last_processed_id = frame_id
            self._frame_count += 1
            fps_frame_count += 1
            
            # FPS calculation (over a rolling window)
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                self._fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # Encode raw frame as JPEG for serving
            _, raw_jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._lock:
                self._latest_frame_jpg = raw_jpg.tobytes()

            # Frame skipping: only run inference on every Nth frame
            if self._frame_count % self.frame_skip != 0:
                continue

            # ─── YOLO Inference ───
            try:
                # Reduce inference load by using imgsz=320
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=0.5,
                    imgsz=320,
                    verbose=False,
                )

                detections = []
                for result in results:
                    boxes = result.boxes
                    
                    # Avoid Heavy Rendering: Do NOT call .plot() on every frame
                    if len(boxes) > 0:
                        annotated = result.plot()
                    else:
                        annotated = frame

                    _, ann_jpg = cv2.imencode(
                        '.jpg', annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 75]
                    )
                    with self._lock:
                        self._latest_annotated_jpg = ann_jpg.tobytes()

                    # Extract detection data
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i].item())
                        if conf < self.conf_threshold:
                            continue
                            
                        bbox = boxes.xyxy[i].tolist()
                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)
                        if area < 500:
                            continue
                            
                        cls_id = int(boxes.cls[i].item())
                        cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        
                        detections.append({
                            "class_name": cls_name,
                            "class_id": cls_id,
                            "confidence": round(conf, 4),
                            "bbox": [round(v, 2) for v in bbox],
                            "lat": telemetry_service.lat,
                            "lon": telemetry_service.lon,
                            "disease": cls_name,
                        })

                self._processed_count += 1
                with self._lock:
                    self._latest_detections = detections
                    self._detection_queue.append({
                        "type": "stream_detection",
                        "status": "no_leaf" if len(detections) == 0 else "success",
                        "frame_id": self._frame_count,
                        "timestamp": time.time(),
                        "detections": detections,
                        "detection_count": len(detections),
                    })

            except Exception as e:
                print(f"[StreamDetector] Inference error: {e}")

        self.active = False

    async def _broadcast_loop(self):
        """Async loop: drain the detection queue and broadcast to WS clients."""
        while self.active:
            try:
                pending = []
                with self._lock:
                    if self._detection_queue:
                        pending = self._detection_queue[:]
                        self._detection_queue.clear()

                for msg in pending:
                    await stream_ws_manager.broadcast(msg)

                await asyncio.sleep(0.1)  # 10 Hz max broadcast rate
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[StreamDetector] Broadcast error: {e}")
                await asyncio.sleep(0.5)

    def get_frame_jpg(self, annotated: bool = True) -> bytes | None:
        with self._lock:
            if annotated and self._latest_annotated_jpg:
                return self._latest_annotated_jpg
            return self._latest_frame_jpg

    def get_latest_detections(self) -> list[dict]:
        with self._lock:
            return list(self._latest_detections)

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "stream_url": self.stream_url,
            "frame_skip": self.frame_skip,
            "conf_threshold": self.conf_threshold,
            "frame_count": self._frame_count,
            "processed_count": self._processed_count,
            "fps": round(self._fps, 1),
            "detection_count": len(self._latest_detections),
            "last_error": self._last_error,
            "uptime_s": round(time.time() - self._started_at, 1) if self.active else 0,
            "ws_clients": len(stream_ws_manager.active),
        }


stream_detector = StreamDetector()


# ── Background Telemetry Loop ──

async def telemetry_loop():
    """Tick the simulator and broadcast telemetry from sim or real drones."""
    while True:
        try:
            # Real-drone telemetry takes precedence if connected
            if telemetry_service.connected:
                s_snap = telemetry_service.get_snapshot()
                ws_snap = {
                    "active": True,
                    "paused": False,
                    "drone_type": "scout",
                    "lat": s_snap["lat"] if s_snap["lat"] != 0.0 else None,
                    "lon": s_snap["lon"] if s_snap["lon"] != 0.0 else None,
                    "alt": s_snap["alt"],
                    "heading": s_snap["heading"],
                    "groundspeed": s_snap["speed"],
                    "battery": s_snap["battery"],
                    "mode": "MAVLINK",
                    "progress": 0,
                    "source": "real"
                }
                await ws_manager.broadcast({"type": "telemetry", "data": ws_snap})

            # Simulator: tick if active+not paused; broadcast if active OR paused
            was_active = telemetry_sim.active
            was_aborted = telemetry_sim.aborted
            telemetry_sim.step(dt=0.5)

            if telemetry_sim.active or telemetry_sim.paused:
                snap = telemetry_sim.snapshot()
                snap["source"] = "simulator"
                await ws_manager.broadcast({"type": "telemetry", "data": snap})
            elif was_active and not telemetry_sim.active:
                # Transition to inactive — emit a final inactive snapshot so the
                # frontend reliably tears down the marker even if it missed it.
                snap = telemetry_sim.snapshot()
                snap["source"] = "simulator"
                await ws_manager.broadcast({"type": "telemetry", "data": snap})

                if telemetry_sim.mission_id and not was_aborted:
                    db.update_mission_status(telemetry_sim.mission_id, "completed")
                    await ws_manager.broadcast({
                        "type": "mission_complete",
                        "mission_id": telemetry_sim.mission_id,
                    })
                elif telemetry_sim.mission_id and was_aborted:
                    db.update_mission_status(telemetry_sim.mission_id, "aborted")
                    await ws_manager.broadcast({
                        "type": "mission_aborted",
                        "mission_id": telemetry_sim.mission_id,
                    })
        except Exception as e:
            print(f"telemetry_loop error: {e}")
        await asyncio.sleep(0.5)


# ── App Setup ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    task = asyncio.create_task(telemetry_loop())
    telemetry_service.start()
    yield
    task.cancel()
    telemetry_service.stop()
    # Clean up stream detector on shutdown
    if stream_detector.active:
        stream_detector.stop()


app = FastAPI(title="Disease Drone Dashboard", version="0.2.0", lifespan=lifespan)
app.include_router(telemetry_router)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Pages ──

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ── API: Stats ──

@app.get("/api/stats")
async def get_stats():
    return db.get_dashboard_stats()


# ── API: Classes ──

_YOLO_CLASSES = None

@app.get("/api/classes")
async def get_classes():
    global _YOLO_CLASSES
    if _YOLO_CLASSES is None:
        _YOLO_CLASSES = YOLO(MODEL_PATH).names
    return _YOLO_CLASSES



# ── API: Missions ──

class MissionCreate(BaseModel):
    type: str = "scan"
    scan_area: list | None = None
    notes: str | None = None

class MissionUpdate(BaseModel):
    status: str

@app.post("/api/missions")
async def create_mission(body: MissionCreate):
    mid = db.create_mission(body.type, body.scan_area, body.notes)
    return {"id": mid, "status": "created"}

@app.get("/api/missions")
async def list_missions():
    return db.get_missions()

@app.get("/api/missions/summary")
async def mission_history(
    limit: int = 100,
    mission_type: str | None = None,
    status: str | None = None,
):
    """Mission history list with aggregated counts."""
    return db.get_missions_summary(limit=limit, mission_type=mission_type, status=status)

@app.get("/api/missions/{mission_id}")
async def get_mission(mission_id: int):
    m = db.get_mission(mission_id)
    if not m:
        raise HTTPException(404, "Mission not found")
    return m

@app.get("/api/missions/{mission_id}/full")
async def get_mission_full(mission_id: int):
    """Full mission detail: waypoints, detections, spray zones, treatments."""
    m = db.get_mission_full(mission_id)
    if not m:
        raise HTTPException(404, "Mission not found")
    return m

@app.put("/api/missions/{mission_id}")
async def update_mission(mission_id: int, body: MissionUpdate):
    db.update_mission_status(mission_id, body.status)
    return {"id": mission_id, "status": body.status}


# ── API: Mission Planning ──

class ScanPlanRequest(BaseModel):
    polygon: list                    # [[lat, lon], ...]
    altitude: float = 4.0
    overlap: float = 0.3
    camera_hfov_deg: float = 62.2
    camera_vfov_deg: float = 48.8
    flight_speed: float = 2.0
    save: bool = False               # if True, persist as a mission
    notes: str | None = None


@app.post("/api/plan/scan")
async def plan_scan(body: ScanPlanRequest):
    """Generate lawnmower waypoints for a scan polygon.

    Returns waypoints + estimated mission stats. If `save=True`, also
    creates a mission record in the database.
    """
    if not body.polygon or len(body.polygon) < 3:
        raise HTTPException(400, "Polygon must have at least 3 vertices")

    waypoints = generate_scan_waypoints(
        scan_polygon=body.polygon,
        altitude=body.altitude,
        overlap=body.overlap,
        camera_hfov_deg=body.camera_hfov_deg,
        camera_vfov_deg=body.camera_vfov_deg,
        flight_speed=body.flight_speed,
    )
    mavlink_items = to_mavlink_mission(waypoints)

    stats = mission_stats(
        waypoints,
        flight_speed=body.flight_speed,
        altitude=body.altitude,
        camera_hfov_deg=body.camera_hfov_deg,
        camera_vfov_deg=body.camera_vfov_deg,
        overlap=body.overlap,
    )
    stats["overlap"] = body.overlap

    response = {
        "waypoints": mavlink_items,
        "waypoint_count": len(mavlink_items),
        "stats": stats,
    }

    if body.save:
        mission_id = db.create_mission(
            mission_type="scan",
            scan_area=body.polygon,
            notes=body.notes or f"Scan @ {body.altitude}m, {int(body.overlap*100)}% overlap",
        )
        db.update_mission_waypoints(mission_id, mavlink_items)
        response["mission_id"] = mission_id

    return response


class SprayPlanRequest(BaseModel):
    zone_ids: list[int] | None = None
    scan_mission_id: int | None = None
    home_lat: float | None = None
    home_lon: float | None = None
    altitude: float = 3.0
    hover_time: float = 5.0
    save: bool = False


@app.post("/api/plan/spray")
async def plan_spray(body: SprayPlanRequest):
    """Generate optimised spray path through approved zones."""
    if body.zone_ids:
        all_zones = db.get_spray_zones()
        zones = [z for z in all_zones if z["id"] in body.zone_ids]
    elif body.scan_mission_id:
        zones = db.get_spray_zones(mission_id=body.scan_mission_id, status="approved")
    else:
        zones = db.get_spray_zones(status="approved")

    if not zones:
        raise HTTPException(400, "No approved zones to spray")

    home = None
    if body.home_lat is not None and body.home_lon is not None:
        home = (body.home_lat, body.home_lon)

    waypoints = optimize_spray_path(
        spray_zones=zones,
        home=home,
        altitude=body.altitude,
        hover_time=body.hover_time,
    )
    mavlink_items = to_mavlink_mission(waypoints)

    stats = mission_stats(
        waypoints,
        flight_speed=3.0,
        altitude=body.altitude,
        hover_time=body.hover_time,
    )
    stats["zone_count"] = len(zones)

    response = {
        "waypoints": mavlink_items,
        "waypoint_count": len(mavlink_items),
        "stats": stats,
    }

    if body.save:
        mission_id = db.create_mission(
            mission_type="spray",
            notes=f"Spray mission: {len(zones)} zones",
        )
        db.update_mission_waypoints(mission_id, mavlink_items)
        response["mission_id"] = mission_id

    return response


# ── API: Mission Execution (simulated) ──

@app.post("/api/missions/{mission_id}/simulate")
async def simulate_mission(mission_id: int):
    """Simulate a mission's flight using the telemetry simulator."""
    mission = db.get_mission(mission_id)
    if not mission:
        raise HTTPException(404, "Mission not found")
    if not mission.get("waypoints"):
        raise HTTPException(400, "Mission has no waypoints — plan it first")

    waypoints = json.loads(mission["waypoints"])
    drone_type = "scout" if mission["type"] == "scan" else "treatment"

    telemetry_sim.start(mission_id, waypoints, drone_type=drone_type)
    db.update_mission_status(mission_id, "in_progress")

    return {
        "status": "simulating",
        "mission_id": mission_id,
        "waypoint_count": len(waypoints),
    }


@app.post("/api/simulator/stop")
async def stop_simulator():
    telemetry_sim.stop()
    # Send a final inactive snapshot so the frontend tears down the HUD/marker
    snap = telemetry_sim.snapshot()
    snap["source"] = "simulator"
    await ws_manager.broadcast({"type": "telemetry", "data": snap})
    return {"status": "stopped"}


@app.post("/api/simulator/abort")
async def abort_simulator():
    """Emergency abort — stops the sim and marks the mission as aborted."""
    mid = telemetry_sim.mission_id
    telemetry_sim.abort()
    if mid:
        db.update_mission_status(mid, "aborted")
    snap = telemetry_sim.snapshot()
    snap["source"] = "simulator"
    await ws_manager.broadcast({"type": "telemetry", "data": snap})
    await ws_manager.broadcast({"type": "mission_aborted", "mission_id": mid})
    return {"status": "aborted", "mission_id": mid}


@app.post("/api/simulator/pause")
async def pause_simulator():
    telemetry_sim.pause()
    return {"status": "paused"}


@app.post("/api/simulator/resume")
async def resume_simulator():
    telemetry_sim.resume()
    return {"status": "resumed"}


class SpeedRequest(BaseModel):
    multiplier: float


@app.post("/api/simulator/speed")
async def set_simulator_speed(body: SpeedRequest):
    telemetry_sim.set_speed_multiplier(body.multiplier)
    return {"status": "ok", "multiplier": telemetry_sim.speed_multiplier}


# ── API: Detections ──

class DetectionCreate(BaseModel):
    mission_id: int
    class_name: str
    confidence: float
    lat: float
    lon: float
    bbox: list | None = None
    image_path: str | None = None


@app.get("/api/detections")
async def list_detections(mission_id: int | None = None):
    return db.get_detections(mission_id)


@app.post("/api/detections")
async def add_detection(body: DetectionCreate):
    did = db.add_detection(
        body.mission_id, body.class_name, body.confidence,
        body.lat, body.lon, body.bbox, body.image_path,
    )
    return {"id": did}


# ── API: Spray Zones ──

class SprayZoneCreate(BaseModel):
    mission_id: int
    geometry: list
    center_lat: float
    center_lon: float
    severity: float = 0.0
    disease_type: str | None = None


class SprayZoneUpdate(BaseModel):
    status: str


@app.get("/api/spray-zones")
async def list_spray_zones(mission_id: int | None = None, status: str | None = None):
    return db.get_spray_zones(mission_id, status)


@app.post("/api/spray-zones")
async def add_spray_zone(body: SprayZoneCreate):
    zid = db.add_spray_zone(
        body.mission_id, body.geometry, body.center_lat, body.center_lon,
        body.severity, body.disease_type,
    )
    return {"id": zid}


@app.put("/api/spray-zones/{zone_id}")
async def update_spray_zone(zone_id: int, body: SprayZoneUpdate):
    db.update_spray_zone_status(zone_id, body.status)
    return {"id": zone_id, "status": body.status}


# ── API: Treatments ──

class TreatmentCreate(BaseModel):
    spray_zone_id: int
    mission_id: int
    lat: float
    lon: float
    spray_duration: float | None = None
    chemical: str | None = None
    notes: str | None = None


@app.get("/api/treatments")
async def list_treatments(mission_id: int | None = None):
    return db.get_treatments(mission_id)


@app.post("/api/treatments")
async def add_treatment(body: TreatmentCreate):
    tid = db.add_treatment(
        body.spray_zone_id, body.mission_id,
        body.lat, body.lon, body.spray_duration, body.chemical, body.notes,
    )
    return {"id": tid}


# ── API: Health ──

class HealthPoint(BaseModel):
    lat: float
    lon: float
    health_score: float
    disease_type: str | None = None


@app.get("/api/health")
async def get_health():
    return db.get_health_data()


@app.post("/api/health")
async def add_health(body: HealthPoint):
    db.add_health_point(body.lat, body.lon, body.health_score, body.disease_type)
    return {"status": "ok"}


# ── API: Telemetry ──
# Handled by telemetry_router in dashboard/api/telemetry.py


# ── API: Drone Connection ──

class DroneConnectRequest(BaseModel):
    drone_type: str             # 'scout' or 'treatment'
    connection: str             # MAVLink URI: udp:..., tcp:..., /dev/ttyUSB0


@app.post("/api/drone/connect")
async def drone_connect(body: DroneConnectRequest):
    """Start the background telemetry service."""
    if body.drone_type not in ("scout", "treatment"):
        raise HTTPException(400, "drone_type must be 'scout' or 'treatment'")
    telemetry_service.start()
    return {"status": "connecting"}


@app.post("/api/drone/disconnect/{drone_type}")
async def drone_disconnect(drone_type: str):
    telemetry_service.stop()
    return {"status": "disconnected"}


@app.get("/api/drone/status")
async def drone_status():
    return {
        "scout_connected": telemetry_service.connected,
        "scout_connection": telemetry_service.connection_string,
        "treatment_connected": False,
        "treatment_connection": None,
        "simulator_active": telemetry_sim.active,
    }


@app.post("/api/missions/{mission_id}/upload")
async def upload_mission_to_drone(mission_id: int):
    """Upload a mission's waypoints to the connected real drone.

    The drone is selected by mission type (scout for 'scan', treatment for 'spray').
    """
    mission = db.get_mission(mission_id)
    if not mission:
        raise HTTPException(404, "Mission not found")
    if not mission.get("waypoints"):
        raise HTTPException(400, "Mission has no waypoints")

    drone_type = "scout" if mission["type"] == "scan" else "treatment"
    waypoints = json.loads(mission["waypoints"])

    try:
        drone_ctrl.upload_mission_to(drone_type, waypoints)
        return {"status": "uploaded", "drone_type": drone_type, "items": len(waypoints)}
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")


@app.post("/api/missions/{mission_id}/launch")
async def launch_mission(mission_id: int):
    """Launch a mission on the real drone (arm + AUTO mode)."""
    mission = db.get_mission(mission_id)
    if not mission:
        raise HTTPException(404, "Mission not found")

    drone_type = "scout" if mission["type"] == "scan" else "treatment"
    try:
        drone_ctrl.start_mission_on(drone_type)
        db.update_mission_status(mission_id, "in_progress")
        await ws_manager.broadcast({
            "type": "mission_launched",
            "mission_id": mission_id,
            "drone_type": drone_type,
        })
        return {"status": "launched", "mission_id": mission_id}
    except Exception as e:
        raise HTTPException(500, f"Launch failed: {e}")


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    """Live telemetry & event stream."""
    await ws_manager.connect(websocket)
    # Send current snapshot on connect
    await websocket.send_text(json.dumps({
        "type": "telemetry",
        "data": telemetry_sim.snapshot(),
    }))
    try:
        while True:
            # We don't expect inbound messages, but keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ── Detection API ──

@app.post("/detect")
async def detect_image(
    image: UploadFile = File(...),
    conf: float = Query(0.4, ge=0.1, le=1.0),
    lat: float | None = Query(None),
    lon: float | None = Query(None),
    mission_id: int | None = Query(None),
):
    """Run disease detection on an uploaded image.

    Returns a list of detections with class_name, confidence, and bbox.
    Optionally stores results in the database if mission_id is provided.
    """
    # Save upload to a temp file
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    file_id = uuid.uuid4().hex[:8]
    save_path = UPLOAD_DIR / f"{file_id}{suffix}"
    try:
        contents = await image.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        # Run inference
        results = detect_diseases(
            str(save_path),
            model_path=MODEL_PATH,
            conf_threshold=conf,
        )

        detections = results["detections"]
        message = results["message"]

        # Optionally persist to DB
        if mission_id is not None:
            for det in detections:
                db.add_detection(
                    mission_id=mission_id,
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    lat=lat or 0.0,
                    lon=lon or 0.0,
                    bbox=det["bbox"],
                    image_path=str(save_path),
                )

        return {
            "status": "no_leaf" if len(detections) == 0 else "success",
            "message": message,
            "detections": detections,
            "count": len(detections),
            "image": str(save_path.name),
        }
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {e}")


@app.post("/api/detect/video")
async def api_detect_video(
    video: UploadFile = File(...),
    conf: float = Query(0.5),
    frame_skip: int = Query(5)
):
    """Run disease detection on an uploaded video file.
    
    Processes every Nth frame (default 5) to optimize performance.
    Returns aggregated detection results across the video.
    """
    suffix = Path(video.filename).suffix if video.filename else ".mp4"
    file_id = uuid.uuid4().hex[:8]
    save_path = UPLOAD_DIR / f"{file_id}{suffix}"
    
    try:
        contents = await video.read()
        with open(save_path, "wb") as f:
            f.write(contents)
            
        cap = cv2.VideoCapture(str(save_path))
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")
            
        model = YOLO(MODEL_PATH)
        all_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Resize frame to speed up inference
            frame_resized = cv2.resize(frame, (640, 480))
            
            results = model.predict(
                source=frame_resized,
                conf=conf,
                iou=0.5,
                imgsz=320,
                verbose=False
            )
            
            frame_detections = []
            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:
                    continue
                    
                for i in range(len(boxes)):
                    box_conf = float(boxes.conf[i].item())
                    if box_conf < conf:
                        continue
                        
                    bbox = boxes.xyxy[i].tolist()
                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    if area < 500:
                        continue
                        
                    cls_id = int(boxes.cls[i].item())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    
                    frame_detections.append({
                        "frame": frame_count,
                        "class_name": cls_name,
                        "confidence": round(box_conf, 4),
                        "bbox": [round(v, 2) for v in bbox]
                    })
                    
            if frame_detections:
                all_detections.extend(frame_detections)
                
        cap.release()
        
        # Optionally clean up the video file
        # save_path.unlink(missing_ok=True)
        
        return {
            "status": "no_leaf" if len(all_detections) == 0 else "success",
            "message": "No leaf detected" if len(all_detections) == 0 else None,
            "detections": all_detections,
            "frames_processed": frame_count // frame_skip,
            "total_frames": frame_count
        }
        
    except Exception as e:
        raise HTTPException(500, f"Video detection failed: {e}")



@app.post("/api/demo/detect")
async def demo_detect():
    """Run real model inference on a sample of test images.

    Picks ~25 random images from the test set, runs YOLOv8 inference,
    assigns mock GPS coordinates scattered around the demo field center,
    stores detections in the DB with a new scan mission, then uses the
    decision engine to cluster detections into spray zones.

    This replaces the purely random demo data with real model predictions.
    """
    CENTER_LAT, CENTER_LON = 12.9716, 77.5946
    SPREAD = 0.008

    # Collect test images
    if not TEST_IMAGES_DIR.exists():
        raise HTTPException(400, f"Test images not found at {TEST_IMAGES_DIR}")

    all_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not all_images:
        all_images = sorted(TEST_IMAGES_DIR.glob("*.png"))
    if not all_images:
        raise HTTPException(400, "No test images found")

    # Sample a manageable subset
    sample_size = min(25, len(all_images))
    sample_images = random.sample(all_images, sample_size)

    # Create a scan mission
    scan_area = [
        [CENTER_LAT - SPREAD, CENTER_LON - SPREAD],
        [CENTER_LAT - SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON - SPREAD],
    ]
    scan_id = db.create_mission("scan", scan_area, "ML inference demo — real model predictions")

    # Plan and store waypoints for the demo scan
    wps = generate_scan_waypoints(scan_area, altitude=4.0, overlap=0.3)
    db.update_mission_waypoints(scan_id, to_mavlink_mission(wps))

    # Run batch inference
    image_paths = [str(p) for p in sample_images]
    try:
        batch_results = detect_batch(
            image_paths,
            model_path=MODEL_PATH,
            conf_threshold=0.3,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    # Store detections with mock GPS
    all_detections = []  # for spray zone clustering
    total_count = 0

    for img_path, result_dict in batch_results.items():
        dets = result_dict["detections"]
        # Assign a mock GPS position for this "image capture point"
        img_lat = CENTER_LAT + random.uniform(-SPREAD, SPREAD)
        img_lon = CENTER_LON + random.uniform(-SPREAD, SPREAD)

        for det in dets:
            # Scatter each detection slightly around the image point
            det_lat = img_lat + random.uniform(-0.0003, 0.0003)
            det_lon = img_lon + random.uniform(-0.0003, 0.0003)

            db.add_detection(
                mission_id=scan_id,
                class_name=det["class_name"],
                confidence=det["confidence"],
                lat=det_lat,
                lon=det_lon,
                bbox=det["bbox"],
                image_path=str(Path(img_path).name),
            )
            total_count += 1

            # Collect for spray zone clustering (only diseased)
            all_detections.append(Detection(
                lat=det_lat,
                lon=det_lon,
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox=det["bbox"],
                image_path=str(Path(img_path).name),
            ))

    # Generate spray zones from real detections using the decision engine
    zones = process_detections(
        all_detections,
        mission_id=scan_id,
        eps_metres=50.0,       # wider clustering radius for field scale
        min_samples=2,
        buffer_metres=20.0,
        confidence_threshold=0.5,
        db_module=db,
    )

    # Generate health grid based on detection proximity
    steps = 15
    for i in range(steps):
        for j in range(steps):
            lat = CENTER_LAT - SPREAD + (2 * SPREAD * i / steps)
            lon = CENTER_LON - SPREAD + (2 * SPREAD * j / steps)
            # Base health degrades near detections (diseased ones)
            base_health = 0.90
            for det in all_detections:
                if det.class_name == "healthy":
                    continue
                dist = math.sqrt((lat - det.lat) ** 2 + (lon - det.lon) ** 2)
                if dist < 0.004:
                    base_health -= det.confidence * (1 - dist / 0.004) * 0.15
            health = max(0.1, min(1.0, base_health + random.uniform(-0.05, 0.05)))
            db.add_health_point(
                lat, lon, round(health, 3),
                random.choice(["leaf_blight", "leaf_spot", "rust", "powdery_mildew"]) if health < 0.6 else None,
            )

    db.update_mission_status(scan_id, "completed")

    return {
        "status": "completed",
        "scan_mission": scan_id,
        "images_processed": sample_size,
        "total_detections": total_count,
        "spray_zones_created": len(zones),
    }


# ── Demo Data ──

@app.post("/api/demo/clear")
async def clear_demo():
    """Wipe all data from the database and fully reset the simulator."""
    conn = db.get_db()
    for table in ["treatments", "spray_zones", "detections", "field_health", "missions"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()

    # Fully reset the simulator (not just stop) so position/state is gone
    telemetry_sim.reset()

    # Broadcast a "cleared" event so the frontend tears down all visuals
    # AND a final inactive telemetry snapshot so any in-flight WS message
    # can't recreate the marker after the user clicked clear.
    await ws_manager.broadcast({"type": "data_cleared"})
    snap = telemetry_sim.snapshot()
    snap["source"] = "simulator"
    await ws_manager.broadcast({"type": "telemetry", "data": snap})

    return {"status": "cleared"}


@app.post("/api/demo/seed")
async def seed_demo():
    """Populate the database with realistic demo data for UI development."""
    CENTER_LAT, CENTER_LON = 12.9716, 77.5946
    SPREAD = 0.008

    diseases = ["leaf_blight", "leaf_spot", "rust", "powdery_mildew", "healthy"]
    chemicals = ["Copper fungicide", "Neem oil", "Mancozeb", "Sulfur dust"]

    scan_area = [
        [CENTER_LAT - SPREAD, CENTER_LON - SPREAD],
        [CENTER_LAT - SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON - SPREAD],
    ]
    scan_id = db.create_mission("scan", scan_area, "Demo scan mission")
    db.update_mission_status(scan_id, "completed")

    # Plan and store waypoints for the demo scan
    wps = generate_scan_waypoints(scan_area, altitude=4.0, overlap=0.3)
    db.update_mission_waypoints(scan_id, to_mavlink_mission(wps))

    # Detections across the field
    for _ in range(80):
        lat = CENTER_LAT + random.uniform(-SPREAD, SPREAD)
        lon = CENTER_LON + random.uniform(-SPREAD, SPREAD)
        disease = random.choices(diseases, weights=[25, 30, 15, 10, 20])[0]
        conf = random.uniform(0.55, 0.98) if disease != "healthy" else random.uniform(0.80, 0.99)
        db.add_detection(scan_id, disease, round(conf, 3), lat, lon)

    # Spray zones
    zone_centers = [
        (CENTER_LAT + 0.003, CENTER_LON - 0.002, "leaf_blight", 0.82),
        (CENTER_LAT - 0.004, CENTER_LON + 0.003, "leaf_spot", 0.65),
        (CENTER_LAT + 0.001, CENTER_LON + 0.005, "rust", 0.74),
        (CENTER_LAT - 0.002, CENTER_LON - 0.004, "powdery_mildew", 0.58),
    ]
    zone_ids = []
    for lat, lon, disease, severity in zone_centers:
        r = 0.001
        geometry = [
            [lat - r, lon - r], [lat - r, lon + r],
            [lat + r, lon + r], [lat + r, lon - r],
        ]
        zid = db.add_spray_zone(scan_id, geometry, lat, lon, severity, disease)
        zone_ids.append(zid)

    db.update_spray_zone_status(zone_ids[0], "approved")
    db.update_spray_zone_status(zone_ids[1], "approved")
    db.update_spray_zone_status(zone_ids[2], "treated")

    spray_id = db.create_mission("spray", notes="Demo spray mission")
    db.update_mission_status(spray_id, "completed")
    spray_wps = optimize_spray_path(
        [{"center_lat": z[0], "center_lon": z[1]} for z in zone_centers[:3]],
        home=(CENTER_LAT, CENTER_LON),
    )
    db.update_mission_waypoints(spray_id, to_mavlink_mission(spray_wps))

    for zid, (lat, lon, _, _) in zip(zone_ids[:3], zone_centers[:3]):
        db.add_treatment(
            zid, spray_id, lat, lon,
            spray_duration=random.uniform(3, 12),
            chemical=random.choice(chemicals),
        )
        db.update_spray_zone_status(zid, "treated")

    # Health grid
    steps = 15
    for i in range(steps):
        for j in range(steps):
            lat = CENTER_LAT - SPREAD + (2 * SPREAD * i / steps)
            lon = CENTER_LON - SPREAD + (2 * SPREAD * j / steps)
            base_health = 0.85
            for zlat, zlon, _, sev in zone_centers:
                dist = math.sqrt((lat - zlat) ** 2 + (lon - zlon) ** 2)
                if dist < 0.004:
                    base_health -= sev * (1 - dist / 0.004) * 0.4
            health = max(0.1, min(1.0, base_health + random.uniform(-0.08, 0.08)))
            db.add_health_point(
                lat, lon, round(health, 3),
                random.choice(diseases[:4]) if health < 0.6 else None,
            )

    return {"status": "seeded", "scan_mission": scan_id, "spray_mission": spray_id}


@app.post("/api/demo/full-flow")
async def demo_full_flow():
    """Create a planned scan mission ready to be simulated.

    Unlike `/api/demo/seed` (which creates a fully completed mission with
    detections), this creates a fresh scan mission with waypoints so the
    user can hit "Simulate" and watch the drone fly.
    """
    CENTER_LAT, CENTER_LON = 12.9716, 77.5946
    SPREAD = 0.005

    scan_area = [
        [CENTER_LAT - SPREAD, CENTER_LON - SPREAD],
        [CENTER_LAT - SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON + SPREAD],
        [CENTER_LAT + SPREAD, CENTER_LON - SPREAD],
    ]

    waypoints = generate_scan_waypoints(scan_area, altitude=4.0, overlap=0.3)
    mavlink_items = to_mavlink_mission(waypoints)

    mission_id = db.create_mission(
        mission_type="scan",
        scan_area=scan_area,
        notes="Demo scan — ready to simulate",
    )
    db.update_mission_waypoints(mission_id, mavlink_items)

    return {
        "mission_id": mission_id,
        "waypoint_count": len(mavlink_items),
        "status": "ready",
    }


# ── Live Stream Detection API ──

class StreamStartRequest(BaseModel):
    stream_url: str = "http://192.168.1.100:8080/?action=stream"
    frame_skip: int = 5
    conf_threshold: float = 0.6


class StreamSettingsRequest(BaseModel):
    frame_skip: int | None = None
    conf_threshold: float | None = None


@app.post("/api/stream/start")
async def start_stream(body: StreamStartRequest):
    """Start capturing and processing a live MJPG stream.

    The stream is read in a background thread. Every `frame_skip`-th frame
    is passed through YOLO for disease detection. Results are broadcast to
    all connected `/ws/stream-detect` WebSocket clients.
    """
    loop = asyncio.get_event_loop()
    stream_detector.start(
        stream_url=body.stream_url,
        frame_skip=body.frame_skip,
        conf_threshold=body.conf_threshold,
        loop=loop,
    )
    return {"status": "started", **stream_detector.snapshot()}


@app.post("/api/stream/stop")
async def stop_stream():
    """Stop the live stream detection."""
    stream_detector.stop()
    await stream_ws_manager.broadcast({"type": "stream_stopped"})
    return {"status": "stopped"}


@app.get("/api/stream/status")
async def stream_status():
    """Get current stream status, stats, and detection counts."""
    return stream_detector.snapshot()


@app.post("/api/stream/settings")
async def update_stream_settings(body: StreamSettingsRequest):
    """Update frame_skip and/or conf_threshold while stream is running."""
    stream_detector.update_settings(
        frame_skip=body.frame_skip,
        conf_threshold=body.conf_threshold,
    )
    return {"status": "updated", **stream_detector.snapshot()}


@app.get("/api/stream/detections")
async def stream_detections():
    """Get the latest detections from the live stream."""
    return {
        "detections": stream_detector.get_latest_detections(),
        "active": stream_detector.active,
    }


@app.get("/api/stream/frame")
async def stream_frame(annotated: bool = True):
    """Get the latest frame as JPEG (raw or YOLO-annotated)."""
    jpg = stream_detector.get_frame_jpg(annotated=annotated)
    if jpg is None:
        # Prevent UI crash by rendering a fallback black frame instead of 404
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black_frame, "Stream initializing...", (180, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        _, black_jpg = cv2.imencode('.jpg', black_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return Response(content=black_jpg.tobytes(), media_type="image/jpeg")
    return Response(content=jpg, media_type="image/jpeg")


@app.post("/api/stream/snapshot")
async def stream_snapshot(
    mission_id: int | None = Query(None),
    lat: float | None = Query(None),
    lon: float | None = Query(None),
):
    """Save the current annotated frame and its detections.

    Optionally stores detections in the database under a mission.
    """
    jpg = stream_detector.get_frame_jpg(annotated=True)
    dets = stream_detector.get_latest_detections()

    if jpg is None:
        raise HTTPException(404, "No frame available")

    # Save frame to disk
    snap_id = uuid.uuid4().hex[:8]
    snap_path = SNAPSHOT_DIR / f"snap_{snap_id}.jpg"
    with open(snap_path, "wb") as f:
        f.write(jpg)

    # Optionally persist detections to DB
    if mission_id is not None:
        for det in dets:
            db.add_detection(
                mission_id=mission_id,
                class_name=det["class_name"],
                confidence=det["confidence"],
                lat=lat or 0.0,
                lon=lon or 0.0,
                bbox=det.get("bbox"),
                image_path=str(snap_path.name),
            )

    return {
        "snapshot": str(snap_path.name),
        "detections": dets,
        "count": len(dets),
    }


@app.websocket("/ws/stream-detect")
async def ws_stream_detect(websocket: WebSocket):
    """Live stream detection results.

    Clients receive messages of type 'stream_detection' with detection data
    every time inference completes on a new frame.
    """
    await stream_ws_manager.connect(websocket)
    # Send current status immediately
    await websocket.send_text(json.dumps({
        "type": "stream_status",
        "data": stream_detector.snapshot(),
    }))
    try:
        while True:
            # Accept inbound messages (settings changes from client)
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                if data.get("type") == "update_settings":
                    stream_detector.update_settings(
                        frame_skip=data.get("frame_skip"),
                        conf_threshold=data.get("conf_threshold"),
                    )
                    await websocket.send_text(json.dumps({
                        "type": "stream_status",
                        "data": stream_detector.snapshot(),
                    }))
            except Exception:
                pass
    except WebSocketDisconnect:
        stream_ws_manager.disconnect(websocket)
    except Exception:
        stream_ws_manager.disconnect(websocket)
