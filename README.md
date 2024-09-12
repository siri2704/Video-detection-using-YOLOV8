 YOLOv8 Object Detection for Video Processing

## Overview

This project uses the YOLOv8 model for object detection in videos. Instead of displaying the results live, the detections are processed and saved as image frames. YOLOv8 is a state-of-the-art object detection model that provides high accuracy and speed. The system processes a video file frame-by-frame, performs object detection, and saves the frames with the detection results.

## How It Works

1. **YOLOv8 Model:**
   - The YOLOv8 model is used to detect objects in the frames from a video.
   
2. **Video Source:**
   - The video source can either be a local video file (`your_video.mp4`) or a live video stream.

3. **Saving Frames:**
   - Instead of showing the detections on the screen, the system saves the frames with detection results into image files.

## Setup Instructions

### Prerequisites

- Python 3.6+
- A CUDA-enabled GPU and CUDA drivers (optional, for faster processing with GPU).
- OpenCV (for video processing).
- PyTorch (for deep learning models).
- `ultralytics` package for YOLOv8 model.

### Required Packages

Install the necessary Python packages:

```bash
pip install torch torchvision torchaudio
pip install opencv-python opencv-python-headless
pip install ultralytics
```

### Clone the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/yolov8-video-detection.git
cd yolov8-video-detection
```

## Code Explanation

The provided code processes video input and performs object detection using YOLOv8.

```python
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8l.pt")

# Predict objects in the video without showing the result live
results = model.predict(source='your_video.mp4')  # Replace with your video file path

# Process each frame in the results
for result in results:
    # Save the frame with detection results
    cv2.imwrite("output_frame.jpg", result.orig_img)
```

## Key Features

- **Object Detection:** Uses YOLOv8 to detect objects in a video source.
- **Frame Processing:** Each frame is analyzed, and the results are saved as images.
- **Custom Video Source:** Replace `'your_video.mp4'` with the path to your video file or a stream source.

## How to Run

1. **Prepare Video File:**
   - Place your video file in the same directory or provide a valid path to the `source` parameter in the script.

2. **Run the Script:**
   - Run the script using Python:

     ```bash
     python main.py
     ```

## Output

The processed frames will be saved as `.jpg` files in the working directory. Each frame corresponds to one detection result from the video.

## Using CUDA

This project can leverage CUDA for faster detection if you have a compatible NVIDIA GPU and drivers installed. By default, the YOLO model will use GPU if available.

## Dependencies

- `torch`: Provides deep learning support for running the YOLOv8 model.
- `opencv-python`: Handles video capturing and frame processing.
- `ultralytics`: The library used for YOLOv8 model.

## License

This project is licensed under the MIT License.

## Acknowledgements

- **YOLOv8:** Developed by [Ultralytics](https://github.com/ultralytics)

Contributions are welcome! Feel free to open issues or pull requests.
```

Feel free to adjust any details, such as replacing `yourusername` with your actual GitHub username or updating any specific instructions as needed.
