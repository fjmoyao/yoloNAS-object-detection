import cv2 
from super_gradients.training import models
from super_gradients.common.object_names import Models 
model = models.get("yolo_nas_s", pretrained_weights="coco")

output = model.predict_webcam()

