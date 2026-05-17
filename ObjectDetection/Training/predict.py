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
from PIL import Image

#
# !!!!! this program must be run from the ObjectDetection directory !!!!!!
#
curr_dir   = Path.cwd()

#
# project parameters 
#
SEED  = 1234
BATCH = 16
EPOCH = 20

PROJECT_PATH = curr_dir / 'Training' / 'YOLO-trained' 
PROJECT_NAME = f'batch-{BATCH:02d}_epo-{EPOCH:03d}'

#
# Load a pretrained model:
#
model = YOLO(PROJECT_PATH / PROJECT_NAME / 'weights/best.pt' )  # load a pretrained model 

#
# Make model predictions on the test set:   
#
 
for img in (curr_dir / 'Data' / 'test' / 'images').glob('*.jpg'):
    print(f'Detecting objects on image: {img.name}')
    results = model.predict(str(img), verbose=False)  # results list

    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Show results to screen (in supported environments)
        #r.show()

        # Save results to disk
        r.save( curr_dir / 'Training' / 'YOLO-trained' / 'Results' / f'{img.stem}_pred.jpg')
        
print('See the predicted images in the "Training/YOLO-trained/Results" directory')


