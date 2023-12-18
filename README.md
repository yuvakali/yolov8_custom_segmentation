
### YOLOv8-Segmentation Training and Inference
This repository contains code for training and inference using YOLOv8 for segmentation tasks. YOLOv8, developed by Ultralytics, is a state-of-the-art object detection and segmentation model.

### Getting Started
Follow the steps below to get started with training and making predictions using YOLOv8 for segmentation.

### Training images are in roboflow or  drive link below
Roboflow
```bash
https://app.roboflow.com/yuvakali/cricket-ctxho/1
```
google drive
```bash
https://drive.google.com/file/d/1C4SIo4T0MCgl_T_V94H_dVly4iGIS9xW/view?usp=drive_link
```

### Prerequisites
Make sure you have the necessary dependencies installed. You can install them using the following command:

```bash
Copy code
pip install -U -r requirements.txt
Clone the YOLOv8 Repository
Clone the YOLOv8 repository from Ultralytics:
```

```bash
Copy code
git clone https://github.com/yuvakali/yolov8_custom_segmentation
Configuration
Specify the configuration file for the YOLOv8 model in the cfg variable. 
```

### You can train the YOLOv8 model from the command line using the following command:

```bash
Copy code
!yolo task=segment mode=train data=custom.yaml model=yolov8x-seg_custom.yaml epochs=10 imgsz=640
Training from Python Script
Load the YOLOv8 model and train it using the following Python script:
```
Python
Copy code
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-seg_custom.yaml")

# Train the model
results = model.train(data="custom_data.yaml", epochs=200, workers=1, batch=8, imgsz=640)
Resume Training
You can resume training using the following Python script:

python
Copy code
from ultralytics import YOLO

model = YOLO()
model.resume(task="segment")  # resume last detection training
 model.resume(model="runs\segment\train\weights\last.pt")   resume from a given model/run
Making Predictions
Make predictions using the trained model with the following commands:

```bash
Copy code
!yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source="test_data"
```
python
Copy code
from ultralytics import YOLO

model = YOLO("runs/segment/train/weights/best.pt")

model.predict(source="test_data")  # Display predictions. Accepts all YOLO predict arguments
Feel free to customize the paths, configuration, and other parameters according to your project requirements.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
Ultralytics for developing YOLOv8 and providing a powerful platform for object detection and segmentation.
The YOLO community for contributions and support.
Happy training and predicting!
