from super_gradients.training import models
import torch

# Get the YOLO NAS small model with pretrained weights on COCO dataset
yolo_nas = models.get("yolo_nas_l", pretrained_weights="coco")

# Set the device to use GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else "cpu"

# Create a Detector class to encapsulate object detection functionality
class Detector:
    def __init__(self, model=yolo_nas):
        self.model = model

    def onImage(self, image_path: str, conf_threshold: float = 0.25, output_file: str = "im_detections"):
        """
        Perform object detection on a single image file.
        - image_path: The path to the image file.
        - conf_threshold: Confidence threshold for object detection.
        - output_file: The output file name for the annotated image.
        """

        model = self.model.to(device)
        detections = model.predict(image_path, conf=conf_threshold)
        detections.save(output_file)
        #self.model.to(device).predict(image_path, conf=conf_threshold).save(output_file)

    def onVideo(self, video_path: str, output_file: str = "detections.mp4"):
        """
        Perform object detection on a video file.
        - video_path: The path to the video file.
        - output_file: The output file name for the annotated video.
        """
        model = self.model.to(device)
        detections = model.predict(video_path)
        detections.save(output_file)

    def realTime(self):
        """
        Perform object detection in real-time using the webcam.
        """
        output = self.model.predict_webcam()