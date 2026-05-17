# version 2.6 - 2026/05/05

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import tensorflow as tf
import GPUtil

from matplotlib.lines import Line2D
from seaborn import heatmap
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from time import time
from stat import ST_CTIME
from cpuinfo import get_cpu_info

def cpu_gpu():
    device = f"CPU [{get_cpu_info()['brand_raw']}]"
    if tf.config.list_physical_devices('GPU'):
        device += f" - GPU {[g.name for g in GPUtil.getGPUs()]}"
    return device
    
def display_history_stat(hist:list):
    metrics = ('accuracy', 'loss', 'val_accuracy', 'val_loss')
    for m in metrics:
        values = np.array([h.history[m] for h in hist])
        print(f'{m}: mean:{values.mean():.3f} std:{values.std():.3f} ', end='')

def elapsed_time_since(t0):
    t = int(time()-t0)
    h = int(t//3600)
    m = int((t - h*3600)//60)
    s = int((t - h*3600 -m*60))
    return f"Elapsed time {t}s -> {h:02d}:{m:02d}:{s:02d}"

def scan_dir(path, tag=None, sortbydate=False):
    '''
    To list the files in a directory tree, with optionnaly a tag to filter the names
    and a flag to sort files by date.
    '''
    import os
    files = []
    data  = [item for item in os.walk(path)]
    
    for line in data:
        DIR, SUBDIRS, FILES = line[0], line[1], line[2]
        if FILES:
            FILES.sort()
            for F in FILES:
                if tag and tag in F:
                    files.append(f'{DIR}/{F}')
    if files:
        if sortbydate:
            files = [(os.stat(f)[ST_CTIME], f) for f in files]
            files.sort()
            files = [f for s, f in files]
        else:
            files.sort()

    return files
    

def display_image(image_array: np.ndarray, 
                  RC: tuple,
                  start: int=0,
                  figsize: tuple=None, 
                  label_array: np.ndarray=None, 
                  reverse: bool=False,
                  ret: bool=False):
    '''
    To display images from image_array on a R x C grid, starting at rank r.
    
    Parameters:
      image_array: the array of images
      RC           the tuple (rows, columns) of the image grid
      start        the starting rank in the array image_array (default: 0)
      figsize      the sise of the display (default: (C//2+1, R//2+1))
      label_array  an optional array of labels to give the imshow title
      reverse      wether to reverse video the image or not (default: False)
      ret          wether to return the fig or not (useful for marimo)
    '''
    R, C = RC
    if figsize is None: figsize=(C//2+1, R//2+1 )
    fig, axes = plt.subplots(R, C, figsize=figsize)
    plt.subplots_adjust(top=.95, bottom=.01, hspace=.05, wspace=.3)
    for i, ax in enumerate(axes.flatten()):
        im = image_array[start + i]
        if reverse: im = 255 - im
        if label_array is not None: ax.set_title(str(label_array[start + i]), fontsize=8)
        ax.imshow(im, cmap='gray')
        ax.axis('off')
    if ret: return fig

def plot_proportion_bar(prop: dict, 
                        class_rank: list, 
                        figsize: tuple=(6,4), 
                        title: str='Proportion of digits in dataset', 
                        ret: bool=False):
    '''
    To plot propotion of classes in different datasets.

    Parameters:
      prop         the dictionnary {<name of the dataset>: <[# of 1, # of 2 ... in the dataset]>}
      class_rank   the list of the class labels (displayed on the X axis)
      figsize      the sise of the display (default: (6,4)
      title        title to display (defaut is empty title 'Proportion of digits in dataset')
      ret          wether to return the fig or not (useful for marimo)
    '''
    width = 0.95/(len(prop.keys()))
    
    coeff = -1/len(prop.keys()) if len(prop.keys()) != 1 else -0.95/10
    x = np.arange(len(class_rank))  # the label ranks on x axis
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    for name, values in prop.items():
        offset = width * coeff
        rects = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(rects, label_type='edge', padding=-10, fontsize=8, color='w', weight='bold')
        coeff += 1
        
    ax.set_ylabel('number of digits')
    ax.set_xlabel('Classes')
    ax.set_title(title)
    ax.set_xticks(x, class_rank)
    ax.set_xlim(-0.5, len(class_rank))
    ax.legend()

def split_stratified_into_train_val_test(dataset: (np.ndarray, np.ndarray), 
                                         frac_train: float=0.7, 
                                         frac_val: float=0.15,
                                         frac_test: float=0.15, 
                                         seed: float=None): 
    '''
    Splits a numpy dataset (data, label) into three subsets: train, val, and test
    following fractional ratios provided, where each subset is stratified by the labels array. 
    Splitting into 3 datasets is achieved by running the scikit-learn train_test_split() twice.

    Parameters:
      dataset: the dataset to split, as a tuple (data array, label array)
      frac_train, frac_val, frac_test  : The ratios with which the dataset will be split into 
             train, val, and test data. The values should be expressed as float fractions 
             and should sum to 1.0.
      seed:    the seed to pass to train_test_split().

    Returns:
      (x_train, y_train), (x_val, y_val), (x_test, y_test) : the 3 sub-dataset.
    '''

    if round(frac_train + frac_val + frac_test, 5) != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))
    X, Y = dataset

    # note on train_test_split : Stratified train/test split is not implemented for shuffle=False 
    # Split the dataset into train and temp datasets:
    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, 
                                                        stratify=Y, 
                                                        test_size=1.0 - frac_train,
                                                        shuffle=True,
                                                        random_state=seed)
                                                        
    # Then split the temp datafset into val and test datasets:
    relative_frac_test = frac_test / (frac_val + frac_test)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                    stratify=y_temp,
                                                    test_size=relative_frac_test,
                                                    shuffle=True,
                                                    random_state=seed)
    
    assert len(X) == len(x_train) + len(x_val) + len(x_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)



def plot_loss_accuracy(hist: list, 
                       max_epoch: int = None,
                       min_acc: float=None, max_acc: float=None, 
                       min_loss: float=None, max_loss: float=None, 
                       plot_train: bool=True, 
                       plot_valid: bool=True,
                       single_legend: bool= True,
                       figsize: tuple=(15,5),
                       message: str='',
                       device_info: bool=True,
                       ret: bool=False):
    '''
    Plot training & validation loss & accuracy values, giving an argument
    'hist' of type 'tensorflow.python.keras.callbacks.History'. 
    '''
    
    custom_lines = [Line2D([0], [0], color='blue', lw=1, marker='o'),
                    Line2D([0], [0], color='orange', lw=1, marker='o')]
    train_color = ('blue', 'royalblue', 'cornflowerblue', 'dodgerblue', 'deepskyblue', 'lightskyblue', 'paleturquoise')
    val_color   = ('orange', 'gold', 'goldenrod', 'darkgoldenrod', 'lightcoral', 'firebrick', 'maroon')

    title_acc = {(True, True): "Model Accuracy",
                 (True, False): "Training Accuray",
                 (False, True): "Validation Accuracy"}
    title_val = {(True, True): "Model Loss",
                 (True, False): "Training Loss",
                 (False, True): "Validation Loss"}
    
    if not isinstance(hist, list): hist = [hist]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.1)
    
    # the title:
    device = cpu_gpu()
    if message: fig.suptitle(message, fontsize=12)
    if device_info: 
        fig.text(0.5, 0.9, device, horizontalalignment='center', fontsize=10, color='gray')

    #
    # Plot training & validation accuracy
    #
    for (i, h) in enumerate(hist):
        
        elaps_time  = h.params.get('elaps', None)
        epoch_array = np.array(h.epoch)+1 if max_epoch is None else np.arange(1, max_epoch+1)
        nb_epoch    = h.params.get('nb_epoch', len(epoch_array))

        if elaps_time: elaps_time = elaps_time.split()[2]
        
        if h.history.get('accuracy') and plot_train:
            label = f'train-#{i+1:1d}'
            if elaps_time: label += f' - {elaps_time}'
            color = train_color[i % len(train_color)]
            ax1.plot(epoch_array, h.history['accuracy'][:nb_epoch], 'o-', markersize=4, color=color, label=label)
            
        if h.history.get('val_accuracy') and plot_valid:
            label = f'valid-#{i+1:1d}'
            if elaps_time: label += f' - {elaps_time}'
            color = val_color[i % len(val_color)]
            ax1.plot(epoch_array, h.history['val_accuracy'][:nb_epoch], 'o-', markersize=4, color=color, label=label)
    
    ax1.set_title(title_acc[(plot_train, plot_valid)])
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.set_xlabel('Epoch', fontsize=10) 
    y_min, y_max = ax1.get_ylim()
    if min_acc is not None: y_min = min_acc
    if max_acc is not None: y_max = max_acc
    ax1.set_ylim((y_min, y_max))
    ax1.grid(which='major', color='xkcd:cool grey',  linestyle='-',  alpha=0.7)
    ax1.grid(which='minor', color='xkcd:light grey', linestyle='--', alpha=0.5)
    if  single_legend:
        ax1.legend(custom_lines, ['Train', 'Valid'], loc='lower right')
    else:
        ax1.legend(loc='lower right', framealpha=0.5)
    if nb_epoch <= 10: ax1.set_xticks(np.arange(1, nb_epoch+1))
    
    #
    # Plot training & validation loss values
    #
    for (i, h) in enumerate(hist):
        
        elaps_time  = h.params.get('elaps', None)
        epoch_array = np.array(h.epoch)+1 if max_epoch is None else np.arange(1, max_epoch+1)
        nb_epoch    = h.params.get('nb_epoch', len(epoch_array))

        if elaps_time: elaps_time = elaps_time.split()[2]
        
        if h.history.get('loss') and plot_train:
            label = f'train-#{i+1:1d}'
            if elaps_time: label += f' - {elaps_time}'
            color = train_color[i % len(train_color)]
            ax2.plot(epoch_array, h.history['loss'][:nb_epoch], 'o-', markersize=4, color=color, label=label)
            
        if h.history.get('val_loss') and plot_valid:
            label = f'valid-#{i+1:1d}'
            if elaps_time: label += f' - {elaps_time}'
            color = val_color[i % len(val_color)]
            ax2.plot(epoch_array, h.history['val_loss'][:nb_epoch], 'o-', markersize=4, color=color, label=label)
    
    ax2.set_title(title_val[(plot_train, plot_valid)])
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_xlabel('Epoch', fontsize=10)
    y_min, y_max = ax2.get_ylim()
    if min_loss is not None: y_min = min_loss
    if max_loss is not None: y_max = max_loss
    ax2.set_ylim((y_min, y_max))
    ax2.grid(which='major', color='xkcd:cool grey',  linestyle='-',  alpha=0.7)
    ax2.grid(which='minor', color='xkcd:light grey', linestyle='--', alpha=0.5)
    if  single_legend:
        ax2.legend(custom_lines, ['Train', 'Valid'], loc='upper right')
    else:
        ax2.legend(loc='upper right', framealpha=0.5)
    if nb_epoch <= 10: ax2.set_xticks(np.arange(1, nb_epoch+1))
    
    if ret: return fig
        

def plot_loss_accuracy_vs_hyperparam(hist: list, 
                                     param: str='batch_size',
                                     min_acc: float=None,
                                     max_acc: float=None,
                                     min_loss: float=None,
                                     max_loss: float=None, 
                                     plot_train: bool=True, 
                                     plot_valid: bool=True,
                                     figsize: tuple=(15,5),   
                                     message: str='',
                                     device_info: bool=True,
                                     ret: bool=False):
    '''
    Plot training & validation loss & accuracy values, giving an argument
    'hist' of type: list of tensorflow.python.keras.callbacks.History. 

    Parameters:
      param: the name of the paramter used to display the legend
      min_acc, max_acc: min and max for plotting training accuracy
      min_loss, max_loss: min and max for plotting training loss
      plot_train: whether to plot train accuracy & loss or not (default: True)
      plot_val: whether to plot validation accuracy & loss or not (default: True)
    '''
    
    custom_lines = [Line2D([0], [0], color='blue', lw=1, marker='o'),
                    Line2D([0], [0], color='orange', lw=1, marker='o')]
    colors = ('red', 'green', 'blue', 'orange', 'cyan', 'magenta')

    title_acc = {(True, True): "Model Accuracy",
                 (True, False): "training Accuray",
                 (False, True): "Validation Accuracy"}
    title_val = {(True, True): "Model Loss",
                 (True, False): "training Loss",
                 (False, True): "Validation Loss"}

    if not isinstance(hist, list): hist = [hist]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.1)

    # the title:
    device = cpu_gpu()
    if message: fig.suptitle(message, fontsize=12)
    if device_info: 
        fig.text(0.5, 0.9, device, horizontalalignment='center', fontsize=10, color='gray')
    
    if plot_train and plot_valid: 
        line = 'o:'
    else:
        line = 'o-'

    # Plot training & validation accuracy
    for (i, h) in enumerate(hist):
        param_value = h.params[param]
        nb_epoch    = h.params['nb_epoch']
        elaps_time  = h.params['elaps'].split()[2]
        epoch_array = np.array(h.epoch)+1

        if h.history.get('accuracy') and plot_train:
            label_prefix = '' if not plot_valid else 'train-'
            label_prefix += f'{param}:{param_value}'
            ax1.plot(epoch_array, h.history['accuracy'], line, markersize=4,
                     color=colors[i], label=label_prefix + f' - {elaps_time}')
        if h.history.get('val_accuracy') and plot_valid:
            label_prefix = '' if not plot_train else 'valid-'
            label_prefix += f'{param}:{param_value}'
            ax1.plot(epoch_array, h.history['val_accuracy'], 'o-', markersize=4,
                     color=colors[i], label=label_prefix + f' - {elaps_time}')
    ax1.set_title(title_acc[(plot_train, plot_valid)])
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch') 
    y_min, y_max = ax1.get_ylim()
    if min_acc is not None: y_min = min_acc
    if max_acc is not None: y_max = max_acc
    ax1.set_ylim((y_min, y_max))
    ax1.grid(which='major', color='xkcd:cool grey',  linestyle='-',  alpha=0.7)
    ax1.grid(which='minor', color='xkcd:light grey', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')
    if nb_epoch <= 10: ax1.set_xticks(np.arange(1, len(h.epoch)+1))
    
    # Plot training & validation loss values
    for (i, h) in enumerate(hist):
        param_value = h.params[param]
        nb_epoch    = h.params['nb_epoch']
        elaps_time  = h.params['elaps'].split()[2]
        epoch_array = np.array(h.epoch)+1

        if h.history.get('loss') and plot_train:
            label_prefix = '' if not plot_valid else 'train '
            label_prefix += f'{param}:{param_value}'
            ax2.plot(epoch_array, h.history['loss'], line, markersize=4,
                     color=colors[i], label=label_prefix + f' - {elaps_time}')
        if h.history.get('val_loss') and plot_valid:
            label_prefix = '' if not plot_train else 'valid '
            label_prefix += f'{param}:{param_value}'
            ax2.plot(epoch_array, h.history['val_loss'], 'o-', markersize=4,
                     color=colors[i], label=label_prefix + f' - {elaps_time}')
    ax2.set_title(title_val[(plot_train, plot_valid)])
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    y_min, y_max = ax2.get_ylim()
    if min_loss is not None: y_min = min_loss
    if max_loss is not None: y_max = max_loss
    ax2.set_ylim((y_min, y_max))
    ax2.grid(which='major', color='xkcd:cool grey',  linestyle='-',  alpha=0.7)
    ax2.grid(which='minor', color='xkcd:light grey', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    if nb_epoch <= 10: ax2.set_xticks(np.arange(1, len(h.epoch)+1))

    if ret: return fig


def show_conf_matrix(actual_label, pred_label, class_label, figsize=(8,7), cmap=colormaps['Blues'], xticks_rot='horizontal', ret=False):
    ''' 
    To display the confusion matrix.

    Parameters:
      actual_label: the array of the actual labels (scalars)
      pred_label:   the array of the predicted labels (scalars)
      class_label:  the array of the label of the classes
      figsize:      the tuple(witdh, height) of the width and height of the figure
      cmap:         the desired color map (default=colormaps[name]['Blues'])
      xticks_rot:   whether to display the X axis labels 'horizontal' or 'vertical' (defaut: 'horizontal')
      ret:          whether to return the fig or not (useful for marimo)
    '''
    fig  = plt.figure(figsize=figsize)
    axis = plt.axes()
    plt.title('Confusion Matrix', fontsize=16, pad=15)
    plt.ylabel('True labels', fontsize=13)
    plt.gca().xaxis.set_label_position('top') 
    plt.xlabel('Predicted labels', fontsize=13)
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(bottom=0.2)
    ConfusionMatrixDisplay.from_predictions(actual_label, 
                                            pred_label, 
                                            ax=axis,
                                            cmap=cmap,
                                            display_labels=class_label, 
                                            xticks_rotation=xticks_rot,
                                            colorbar=False);
    if ret: return fig
