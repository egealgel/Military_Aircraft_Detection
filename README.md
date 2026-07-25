# 🛩️ Military Aircraft Recognition

A YOLOv8-based object detection system for identifying military aircraft from images and videos. Trained on the [MilitaryAircraftDetectionDataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset) from Kaggle.

## 📋 Project Overview

| Component           | Technology                                |
| ------------------- | ----------------------------------------- |
| **Detection Model** | Ultralytics YOLOv8                        |
| **Training**        | Google Colab Pro (GPU: T4/P100)           |
| **Web Interface**   | Gradio                                    |
| **Dataset**         | Kaggle - MilitaryAircraftDetectionDataset |

## 🚀 Quick Start

### 1. Train the Model (Colab Pro)

1. Open [`colab_notebooks/military_aircraft_yolov8_training.ipynb`](colab_notebooks/military_aircraft_yolov8_training.ipynb) in Google Colab
2. Follow the step-by-step instructions in the notebook
3. Download the trained `best.pt` from Google Drive

### 2. Setup the Web App

```bash
# Navigate to the app directory
cd app

# Install dependencies
pip install -r requirements.txt

# Place the trained model
# Download best.pt from Google Drive and save to:
# app/models/best.pt
```

### 3. Run the Web App

```bash
cd app
python gradio_app.py
```

Open your browser at **http://localhost:7860**

## 🎯 Features

- **Image Detection**: Upload images (.jpg, .jpeg, .png) for aircraft detection
- **Video Detection**: Upload videos (.mp4, .avi, .mov) for frame-by-frame analysis
- **Confidence Threshold**: Adjustable sensitivity via slider
- **Frame Skip**: Skip frames in video for faster processing
- **Detection Summary**: Real-time summary of detected aircraft types and confidence scores

## 🏗️ Project Structure

```
military_aircraft_recognition/
├── plans/
│   └── plan.md                           # Project plan
├── colab_notebooks/
│   └── military_aircraft_yolov8_training.ipynb  # Colab training notebook
├── app/
│   ├── requirements.txt                  # Python dependencies
│   ├── gradio_app.py                     # Gradio web application
│   ├── utils.py                          # Detection utilities
│   └── models/                           # Trained model weights
└── README.md
```

## 🧠 Supported Aircraft Types

The model is trained on the [MilitaryAircraftDetectionDataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)
which contains **100+ different military aircraft types**. All class names are
extracted **dynamically** from the dataset annotations during training — no
hardcoded list is needed. The model stores all class names internally, and the
Gradio app reads them automatically.

This means **every aircraft type in the dataset** will be detected and labeled
correctly, covering a wide range including:

- **Fighter Jets**: F-14 Tomcat, F-15 Eagle, F-16 Fighting Falcon, F-22 Raptor, F-35 Lightning, F-117 Nighthawk, A-10 Thunderbolt, and more
- **Bombers**: B-1 Lancer, B-2 Spirit, B-52 Stratofortress, etc.
- **Transport & Tanker**: C-130 Hercules, C-17 Globemaster, C-5 Galaxy, KC-10 Extender, KC-135 Stratotanker
- **Reconnaissance & Other**: E-2 Hawkeye, U-2 Dragon Lady, and many more

## 📊 Training Details

- **Base Model**: `yolov8m.pt` (medium)
- **Image Size**: 640×640
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 16
- **Hardware**: Google Colab Pro (NVIDIA T4/P100)
- **Training Time**: ~1-3 hours

## 🔧 Dependencies

- Python 3.9+
- ultralytics>=8.3.0
- gradio>=5.0.0
- opencv-python-headless>=4.9.0
- Pillow>=10.0.0
- numpy>=1.24.0

## 🤝 Deployment

The Gradio app can be deployed on:

- **Local**: Run `python gradio_app.py`
- **HuggingFace Spaces**: Upload to [HuggingFace Spaces](https://huggingface.co/spaces) for free hosting with GPU

## 📝 License

This project is for educational and research purposes only.

## 🙏 Acknowledgments

- Dataset: [MilitaryAircraftDetectionDataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset) on Kaggle
- Framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Web Interface: [Gradio](https://gradio.app/)
