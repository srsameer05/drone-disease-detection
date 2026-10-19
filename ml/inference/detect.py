"""
Inference module for crop disease detection.

Provides:
    - detect_diseases(): Run detection on a single image
    - detect_batch(): Run detection on multiple images
    - export_onnx(): Export model to ONNX format
    - DetectionResult: Structured output dataclass

Usage:
    from ml.inference.detect import detect_diseases, export_onnx

    # Single image
    results = detect_diseases("path/to/image.jpg")
    for det in results:
        print(det.class_name, det.confidence, det.bbox)

    # Export to ONNX
    export_onnx("runs/detect/train/weights/best.pt")
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

ML_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ML_DIR / "models" / "best.pt"


@dataclass
class DetectionResult:
    """Single detection result."""
    bbox: list  # [x1, y1, x2, y2] in pixels
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self):
        return asdict(self)


def detect_diseases(
    image_path: str,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    imgsz: int = 640,
) -> dict:
    """
    Run disease detection on a single image.

    Args:
        image_path: Path to input image.
        model_path: Path to model weights. Defaults to best.pt.
        conf_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for NMS.
        imgsz: Input image size.

    Returns:
        Dict with 'detections' and 'message'.
    """
    def resolve_model_path(model_path: Optional[str] = None) -> str:
        default = DEFAULT_MODEL
        fallback = Path("runs/detect/train/weights/best.pt")

        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Provided model not found: {path}")
            return str(path)

        if default.exists():
            return str(default)

        if fallback.exists():
            return str(fallback)

        raise FileNotFoundError(
            "No model weights found.\n"
            f"Checked:\n- {default}\n- {fallback}\n"
            "Please provide --model explicitly."
        )

    model = YOLO(resolve_model_path(model_path))
    class_names = model.names

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            continue
            
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < conf_threshold:
                continue
                
            bbox = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            
            if area < 500:
                continue
                
            cls_id = int(boxes.cls[i].item())
            cls_name = class_names.get(cls_id, f"class_{cls_id}")

            detections.append(DetectionResult(
                bbox=[round(v, 2) for v in bbox],
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf, 4),
            ).to_dict())

    if not detections:
        return {
            "detections": [],
            "message": "No leaf detected"
        }
    return {
        "detections": detections,
        "message": None
    }


def detect_batch(
    image_paths: list[str],
    model_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    imgsz: int = 640,
) -> dict[str, dict]:
    """
    Run disease detection on multiple images.

    Returns:
        Dict mapping image path → dict with 'detections' and 'message'.
    """
    if model_path is None:
        model_path = str(DEFAULT_MODEL)

    model = YOLO(model_path)
    class_names = model.names

    results = model.predict(
        source=image_paths,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    output = {}
    for img_path, result in zip(image_paths, results):
        detections = []
        boxes = result.boxes
        if len(boxes) == 0:
            output[img_path] = {"detections": [], "message": "No leaf detected"}
            continue
            
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < conf_threshold:
                continue
                
            bbox = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            
            if area < 500:
                continue
                
            cls_id = int(boxes.cls[i].item())
            cls_name = class_names.get(cls_id, f"class_{cls_id}")

            detections.append(DetectionResult(
                bbox=[round(v, 2) for v in bbox],
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf, 4),
            ).to_dict())

        if not detections:
            output[img_path] = {"detections": [], "message": "No leaf detected"}
        else:
            output[img_path] = {"detections": detections, "message": None}

    return output


def export_onnx(
    model_path: Optional[str] = None,
    imgsz: int = 640,
    simplify: bool = True,
) -> str:
    """
    Export YOLO model to ONNX format.
    """

    if model_path is None:
        model_path = str(DEFAULT_MODEL)

    print(f"Exporting to ONNX: {model_path}")

    model = YOLO(model_path)

    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=simplify)

    print(f"Exported: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disease detection inference")
    parser.add_argument("image", nargs="?", help="Image path for detection")
    parser.add_argument("--model", type=str, default=None, help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--export-onnx", action="store_true", help="Export model to ONNX")
    args = parser.parse_args()

    if args.export_onnx:
        export_onnx(args.model)
    elif args.image:
        result = detect_diseases(args.image, model_path=args.model, conf_threshold=args.conf)
        if not result["detections"]:
            print(result["message"])
        for det in result["detections"]:
            print(f"  {det['class_name']} ({det['confidence']:.2%}) @ {det['bbox']}")
    else:
        print("Provide an image path or use --export-onnx")

