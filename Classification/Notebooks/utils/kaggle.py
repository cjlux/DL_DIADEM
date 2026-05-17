import os, sys
from pathlib import Path

def config():
    '''
    Configure some path and install some modules if running within a KAGGLE session
    '''
    
    if 'kaggle' in Path.cwd().as_posix():
        KAGGLE = True
        print(f'Notebook running on KAGGLE')
    else:
        KAGGLE = False
        print(f'Notebook running locally')
    
    if KAGGLE:
        #
        # Add the path to the <utils> directory to the sys.path list:
        #
        paths = list(Path('/kaggle').rglob('utils'))
        if len(paths) == 0:
            print('\tNo <utils> directory found in your workspace, sorry.')
        else:
            util_path = paths[0].parent.as_posix()
            sys.path.append(util_path)
            print(f'\t<{util_path}> added to sys.path.')
            #
        # Create a link <img> to the image directory of the dataset éDL img"
        #
        img_dataset_path = Path('/kaggle/input/datasets/jlcharles/dl-img')
        link_to_img_dir  = Path.cwd() / 'img'
        if not link_to_img_dir.is_symlink(): 
            link_to_img_dir.symlink_to(img_dataset_path)
        print(f'\tlink <{link_to_img_dir}> to <{img_dataset_path}> OK')
        #
        # import the GPUtil module missing on KAGGLE
        #
        try:
            os.system('pip install GPUtil')
        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)    
