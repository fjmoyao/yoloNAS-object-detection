import cv2
from contextlib import contextmanager
from super_gradients.training import models
from super_gradients.common.object_names import Models

# Define a context manager to get the model object and ensure it is properly cleaned up after use
@contextmanager
def get_model():
    model = models.get("yolo_nas_s", pretrained_weights="coco")
    yield model
    del model

# Define a function to run real-time object detection using the webcam
def run_realtime_detection():
    # Get the model object using the get_model function
    with get_model() as model:
        # Use the model's predict_webcam method to perform real-time object detection using the computer's webcam
        output = model.predict_webcam()

# If the script is run directly as a subprocess, call the run_realtime_detection function to perform real-time object detection
if __name__ == "__main__":
    run_realtime_detection()

