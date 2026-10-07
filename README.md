#  Military Aircraft Recognition

A YOLOv8-based object detection system for identifying military aircraft from images and videos. Trained on the [MilitaryAircraftDetectionDataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset) from Kaggle.

##  Project Overview

| Component           | Technology                                |
| ------------------- | ----------------------------------------- |
| **Detection Model** | Ultralytics YOLOv8                        |
| **Training**        | Google Colab Pro (GPU: A100)           |
| **Web Interface**   | Gradio                                    |
| **Dataset**         | Kaggle - MilitaryAircraftDetectionDataset |



https://youtu.be/ALHd1C4-gx4





<img width="1200" height="800" alt="a10" src="https://github.com/user-attachments/assets/653ec244-47b4-41d1-9060-27c99603d979" />


##  Training Details

- **Base Model**: `yolov8m.pt` (medium)
- **Image Size**: 640×640
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 16
- **Hardware**: Google Colab Pro (NVIDIA A100 GPU)
- **Training Time**: ~5 hours

##  Dependencies

- Python 3.9+
- ultralytics>=8.3.0
- gradio>=5.0.0
- opencv-python-headless>=4.9.0
- Pillow>=10.0.0
- numpy>=1.24.0

