"""
Live USB / Drone inference wrapper for ML module.
"""
from live_cam import run_drone_detector

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Drone Crop Disease Detection")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (e.g. 0 for USB cam)")
    parser.add_argument("--engine", type=str, default="onnx", choices=["onnx", "pt"], help="Engine")
    parser.add_argument("--profile", type=int, default=1, choices=[1, 2, 3], help="1=Turbo, 2=Balanced, 3=Precision")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--headless", action="store_true", help="Headless onboard mode")
    args = parser.parse_args()

    run_drone_detector(
        camera_id=args.camera,
        engine=args.engine,
        profile=args.profile,
        conf=args.conf,
        width=args.width,
        height=args.height,
        headless=args.headless,
    )
