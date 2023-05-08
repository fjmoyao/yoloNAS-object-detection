from super_gradients.training import models
import torch

yolo_nas_l = models.get("yolo_nas_s", pretrained_weights="coco")
device = 'cuda' if torch.cuda.is_available() else "cpu"
class Detector:
    def __init__(self, model=yolo_nas_l):
        self.model = model


    def onImage(self, imagePath):
        #self.model.predict(imagePath, conf=0.25)#.save("./predicted.png")
        self.model.to(device).predict(imagePath, conf=0.25).save("im_detections")

    def onVideo(self, videoPath):
        self.model.to(device).predict(videoPath).save("detection.mp4")

    def realTime(self):
        output = self.model.predict_webcam()

