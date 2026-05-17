from pathlib import Path
from ultralytics import YOLO
from time import sleep

BATCH = 32
EPOCH = 20
model_dir = Path('./YOLO-pretrained')
data_path = '../Data/'

from pathlib import Path
Path.cwd()

yolo = 'YOLOv8n'


project = Path.cwd() / './YOLO-trained/' 
name = f'batch-{BATCH:02d}_epo-{EPOCH:03d}'


model = YOLO(model_dir / f'{yolo.lower()}.pt')  # load a pretrained model 

model.train(data=data_path, 
            project=project, 
            name=name, 
            epochs=EPOCH, 
            imgsz=256, 
            batch=BATCH, 
            patience=8, 
            cache=False,
            workers=0,
            exist_ok=True, 
            pretrained=True,
            optimizer='auto', 
            seed=1234,
            verbose=False)

