"""
Military Aircraft Recognition - Gradio Web App

This application provides a web interface for military aircraft detection
using a trained YOLOv8 model. Supports both image and video input.

Usage:
    python gradio_app.py

The app will be available at http://localhost:7860
"""

import os
import tempfile
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from utils import (
    process_image,
    process_video,
    get_detection_summary,
    DEFAULT_MODEL_PATH,
    MODEL_DIR,
)

# ─── Constants ───────────────────────────────────────────────────────────────

TITLE = "🛩️ Military Aircraft Recognition"
DESCRIPTION = """
Detect military aircraft in images and videos using YOLOv8.

**Dataset:** Trained on the [MilitaryAircraftDetectionDataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)
with **all 100+ aircraft types** dynamically extracted from annotations.

**Instructions:**
- **Image tab**: Upload an image (.jpg, .jpeg, .png) to detect aircraft
- **Video tab**: Upload a video (.mp4, .avi, .mov) for frame-by-frame detection
"""

MODEL_STATUS_PATH = MODEL_DIR / ".model_loaded"

# ─── Initialization ──────────────────────────────────────────────────────────


def check_model_exists() -> bool:
    """Check if the trained model exists, return status message."""
    if DEFAULT_MODEL_PATH.exists():
        return True
    return False


# ─── Image Processing Functions ──────────────────────────────────────────────


def detect_image(image: np.ndarray, conf_threshold: float) -> tuple:
    """
    Detect aircraft in an uploaded image.

    Args:
        image: Input image as numpy array (RGB).
        conf_threshold: Confidence threshold slider value (0.0 to 1.0).

    Returns:
        Tuple of (annotated_image, detection_summary).
    """
    if image is None:
        return None, "⚠️ Please upload an image first."

    try:
        result_img = process_image(image, conf_threshold=conf_threshold)
        summary = get_detection_summary(image, conf_threshold=conf_threshold)

        # Build summary text
        total = summary["total"]
        if total == 0:
            text = "❌ No aircraft detected."
        else:
            text = f"✅ **{total} aircraft detected:**\n"
            for det in summary["detections"]:
                text += f"- {det['class']} ({det['confidence']:.1%})\n"

        # result_img is in BGR from ultralytics, Gradio needs RGB
        return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), text

    except Exception as e:
        return None, f"❌ Error: {str(e)}\n```\n{traceback.format_exc()}\n```"


# ─── Video Processing Functions ──────────────────────────────────────────────


def detect_video(
    video_path: str,
    conf_threshold: float,
    frame_skip: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """
    Detect aircraft in an uploaded video.

    Args:
        video_path: Path to uploaded video file.
        conf_threshold: Confidence threshold (0.0 to 1.0).
        frame_skip: Process every (frame_skip + 1)-th frame.
        progress: Gradio progress bar.

    Returns:
        Tuple of (output_video_path, status_message).
    """
    if video_path is None:
        return None, "⚠️ Please upload a video first."

    if not os.path.exists(video_path):
        return None, "⚠️ Video file not found."

    # Create output path (persistent, not tempfile - Gradio needs it)
    output_dir = str(MODEL_DIR / "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.mp4")

    try:
        progress(0, desc="Starting video processing...")

        def report_progress(p: float):
            progress(p, desc=f"Processing: {p:.0%}")

        process_video(
            input_video_path=video_path,
            output_video_path=output_path,
            conf_threshold=conf_threshold,
            iou_threshold=0.45,
            frame_skip=frame_skip,
            progress_callback=report_progress,
        )

        progress(1.0, desc="✅ Complete!")
        return output_path, "✅ Video processing complete! You can preview or download the result below."

    except Exception as e:
        return None, f"❌ Error processing video: {str(e)}\n```\n{traceback.format_exc()}\n```"


# ─── UI Setup ────────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    """Build and return the Gradio interface."""
    model_available = check_model_exists()

    with gr.Blocks(
        title="Military Aircraft Recognition",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
        css="""
        .gradio-container { max-width: 1200px !important; margin: auto; }
        .summary-box { min-height: 100px; }
        footer { display: none !important; }
        """,
    ) as app:

        # Header
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        # Model status
        if not model_available:
            gr.Warning(
                "⚠️ **Model not found!** "
                "Please train the model first using `colab_notebooks/military_aircraft_yolov8_training.ipynb`, "
                "then download `best.pt` and place it in the `app/models/` folder."
            )
        else:
            gr.Info("✅ Model loaded and ready!")

        # Confidence threshold slider (shared)
        with gr.Row():
            conf_threshold = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="🎯 Confidence Threshold",
                info="Lower values detect more objects but may include false positives.",
            )

        # Tabs for Image / Video
        with gr.Tabs():
            # ── Image Tab ────────────────────────────────────────────────
            with gr.TabItem("📷 Image Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            height=400,
                        )
                        with gr.Row():
                            image_clear = gr.Button("🗑️ Clear", variant="secondary")
                            image_submit = gr.Button(
                                "🔍 Detect Aircraft", variant="primary", scale=2
                            )

                    with gr.Column(scale=1):
                        image_output = gr.Image(
                            label="Detection Result",
                            type="numpy",
                            height=400,
                        )
                        image_summary = gr.Markdown(
                            label="Detection Summary", elem_classes=["summary-box"]
                        )

                image_submit.click(
                    fn=detect_image,
                    inputs=[image_input, conf_threshold],
                    outputs=[image_output, image_summary],
                )
                image_clear.click(
                    fn=lambda: (None, ""),
                    outputs=[image_input, image_summary],
                )

            # ── Video Tab ────────────────────────────────────────────────
            with gr.TabItem("🎥 Video Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video",
                            height=400,
                        )
                        frame_skip = gr.Slider(
                            minimum=0,
                            maximum=5,
                            value=1,
                            step=1,
                            label="⏭️ Frame Skip",
                            info="0 = all frames (slower), 1 = every other frame, 5 = every 6th frame (faster)",
                        )
                        video_submit = gr.Button(
                            "🔍 Detect Aircraft in Video",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        video_output = gr.Video(
                            label="Processed Video",
                            height=400,
                            interactive=False,
                        )
                        video_status = gr.Markdown(
                            label="Status", elem_classes=["summary-box"]
                        )

                video_submit.click(
                    fn=detect_video,
                    inputs=[video_input, conf_threshold, frame_skip],
                    outputs=[video_output, video_status],
                )

        # Footer
        gr.Markdown(
            """
            ---
            Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 
            & [Gradio](https://gradio.app/) 🤗
            """
        )

    return app


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(exist_ok=True)

    # Launch the app
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
