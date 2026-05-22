#####################################################
#         Jean-Luc.Charles@mailo.com
#            2026/05/20 - v1.1
#
# Example of Python programs and dev tree structure
# to train a YOLO model to detect objects in images
#####################################################
#
# ObjectDetection/
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
# !!!!! this program must be run from the ObjectDetection directory !!!!!!
#
curr_dir   = Path.cwd()

#
# parameters passed to the model.train()...) method:
#
SEED  = 1234
BATCH = 16
EPOCH = 30
IMG_SIZE  = 512
PATIENCE  = 10
DATA_PATH = './Data/data.yaml'
PROJECT_PATH = curr_dir / 'Training' / 'YOLO-trained' 
PROJECT_NAME = f'batch-{BATCH:02d}_epo-{EPOCH:03d}'

#
# Load a pretrained model:
#
model = YOLO(PROJECT_PATH / PROJECT_NAME / 'weights/best.pt' )  # load a pretrained model 

#
# Evaluate the model on the test set:
#
metrics = model.val(data=DATA_PATH, 
                    project=PROJECT_PATH, 
                    name=PROJECT_NAME, 
                    batch=1,
                    split='test',
                    plots=False,
                    verbose=False)


