#####################################################
#         Jean-Luc.Charles@mailo.com
#            2026/05/20 - v1.1
#
# Example of Python programs and dev tree structure
# to train a YOLO model to detect objects in images
#####################################################
#
# ObjectDetection/     <- the root dir for object detection
# ├── Data
# │   ├── test
# │   ├── train
# │   └── valid
# └── Training
#     ├── Results
#     ├── YOLO-pretrained
#     └── YOLO-trained
#

from pathlib import Path
from ultralytics import YOLO

#
# !!!!! this program must be run from the DIADEM-DL/ObjectDetection directory !!!!!!
#
cur_dir   = Path.cwd()
model_dir  = Path('./Training/YOLO-pretrained')
yolo_model = model_dir / 'yolov8n.pt'  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

#
# parameters passed to the model.train()...) method:
#
SEED  = 1234
BATCH = 16
EPOCH = 30
IMG_SIZE  = 512
PATIENCE  = 10
DATA_PATH = './Data/data.yaml'
PROJECT_PATH = cur_dir / 'Training' / 'YOLO-trained' 
PROJECT_NAME = f'batch-{BATCH:02d}_epo-{EPOCH:03d}'

#
# Load a pretrained model:
#
model = YOLO(yolo_model)  

#
# Train the YOLO model:
#
model.train(data=DATA_PATH, 
            project=PROJECT_PATH, 
            name=PROJECT_NAME, 
            epochs=EPOCH, 
            imgsz=IMG_SIZE, 
            batch=BATCH, 
            patience=PATIENCE, 
            cache=False,
            workers=0,
            exist_ok=True, 
            pretrained=True,
            optimizer='auto', 
            seed=SEED,
            verbose=False)


