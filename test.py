import cv2
from ultralytics import YOLO

model = YOLO("yolov8l.pt")

# Remove show=True and use an alternative method to handle the output
results = model.predict(source='your_video.mp4')  # or a different video source

# Process and save results instead of displaying them
for result in results:
    # Do something with the result, e.g., save the frame with detections
    cv2.imwrite("output_frame.jpg", result.orig_img)
