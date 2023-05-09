# Object detection with YOLO NAS

This repository contains a script for object detection using the YOLO NAS model with pretrained weights on the COCO dataset. The *detector.py* script provides functions to detect objects in images and videos, and the *yolonas_webcam.py* script is for real-time object detection using the webcam. The *app.py* script provides a Streamlit app for user-friendly detection.

### Requirements
- Python 3.9
- PyTorch 1.9 or higher
- OpenCV 4.5 or higher
- Streamlit 1.0 or higher

### Installation

Clone this repository:
```
git clone https://github.com/fjmoyao/yoloNAS-object-detection.git

```

Install the required packages:
```
pip install -r requirements.txt
```

### Usage 
To start the Streamlit app, run the following command:
```
streamlit run app.py
```
To start real-time object detection using the webcam, run the following command:
```
python yolonas_webcam.py
```

