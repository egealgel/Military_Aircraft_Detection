"""
Utility functions for Military Aircraft Recognition.

This module provides helper functions for:
- Loading YOLO model
- Processing images and videos
- Drawing detection results
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "best.pt"

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.25
# NMS IoU threshold
IOU_THRESHOLD = 0.45

# ─── Model Loading ───────────────────────────────────────────────────────────

_model_instance: Optional[YOLO] = None


def get_model(model_path: Optional[str] = None) -> YOLO:
    """
    Load and cache the YOLO model.

    Class names are extracted dynamically from the trained model itself,
    so there is no need for a hardcoded list.

    Args:
        model_path: Path to the model weights file. If None, uses default path.

    Returns:
        Loaded YOLO model instance.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    global _model_instance

    if model_path is None:
        model_path = str(DEFAULT_MODEL_PATH)

    if _model_instance is not None:
        return _model_instance

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n\n"
            "Please download the trained model:\n"
            "1. Train the model using the Colab notebook (colab_notebooks/)\n"
            "2. Download best.pt from Google Drive\n"
            "3. Place it in: app/models/best.pt"
        )

    _model_instance = YOLO(model_path)
    return _model_instance


def get_class_names(model: Optional[YOLO] = None) -> list:
    """
    Get class names from the model.

    The class names are stored inside the trained model weights,
    so they are always up-to-date with the actual training data.

    Args:
        model: YOLO model instance. If None, loads the default model.

    Returns:
        List of class name strings as they appear in the dataset.
    """
    if model is None:
        model = get_model()
    return list(model.names.values())


# ─── Image Processing ────────────────────────────────────────────────────────


def process_image(
    input_image: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> np.ndarray:
    """
    Run object detection on an input image.

    Args:
        input_image: Input image as numpy array (H, W, 3) in RGB format.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Annotated image with bounding boxes as numpy array.
    """
    model = get_model()
    results = model(
        input_image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    return results[0].plot()


def process_image_pil(
    input_image: Image.Image,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> Image.Image:
    """
    Run object detection on a PIL image.

    Args:
        input_image: PIL Image in RGB format.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.

    Returns:
        PIL Image with bounding boxes drawn.
    """
    np_image = np.array(input_image)
    result_np = process_image(np_image, conf_threshold, iou_threshold)
    # result_np comes in BGR from ultralytics, convert back to RGB
    return Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB))


# ─── Video Processing ────────────────────────────────────────────────────────


def process_video(
    input_video_path: str,
    output_video_path: str,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    frame_skip: int = 0,
    progress_callback=None,
) -> str:
    """
    Run object detection on a video file.

    Args:
        input_video_path: Path to input video file.
        output_video_path: Path to save the output video (.mp4).
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.
        frame_skip: Process every (frame_skip + 1)-th frame.
        progress_callback: Optional callable that accepts progress (0.0 to 1.0).

    Returns:
        Path to the output video file.
    """
    import imageio.v3 as iio

    model = get_model()

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Collect annotated frames
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every (frame_skip + 1)-th frame
        if frame_count % (frame_skip + 1) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, conf=conf_threshold, iou=iou_threshold, verbose=False)
            annotated = results[0].plot()  # BGR
            frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_count += 1

        if progress_callback and frame_count % 30 == 0:
            progress_callback(min(frame_count / total_frames, 1.0))

    cap.release()

    # Write MP4 with imageio (bundled ffmpeg, works on macOS)
    iio.imwrite(output_video_path, frames, fps=fps, codec="libx264", quality=7)

    if progress_callback:
        progress_callback(1.0)

    return output_video_path


# ─── Helper ──────────────────────────────────────────────────────────────────


def get_detection_summary(
    input_image: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Get a summary of detections for an image (without annotation).

    Class names are read dynamically from the model,
    so all classes from the original dataset are supported.

    Args:
        input_image: Input image as numpy array (RGB).
        conf_threshold: Confidence threshold.

    Returns:
        Dictionary with detection counts per class.
    """
    model = get_model()
    results = model(input_image, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    if boxes is None:
        return {"total": 0, "detections": []}

    summary = {"total": len(boxes), "detections": []}
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        summary["detections"].append(
            {"class": class_name, "confidence": round(confidence, 3)}
        )

    return summary
