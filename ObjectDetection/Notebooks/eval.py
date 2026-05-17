from pathlib import Path
from ultralytics import YOLO
from time import sleep

BATCH = 32
EPOCH = 20

model_dir = Path('./YOLO-pretrained')
data_path = '../Data/'

from pathlib import Path

project = Path.cwd() / './YOLO-trained/' 
name = f'batch-{BATCH:02d}_epo-{EPOCH:03d}'

model = YOLO(project / name / 'weights/best.pt' )  # load a pretrained model 
metrics = model.val(project=project, 
                    name=name,
                    batch=1,
                    split='test',)
print(metrics)
