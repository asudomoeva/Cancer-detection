
import matplotlib.pyplot as plt
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
import os
from PIL import Image
from skimage.color import rgb2gray
import re
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils import class_weight

import os
import pandas as pd
import random

import tensorflow as tf
#from tensorflow.keras.applications import VGG16, InceptionV3, VGG19, ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam


def choose_image(image_number):
  
    global slide_path
    global tumor_mask_path
    global slide
    global tumor_mask

    slide_path = os.path.join(SLIDES_DIR, 'tumor_'+image_number+'.tif')
    tumor_mask_path =  os.path.join(SLIDES_DIR, 'tumor_'+image_number+'_mask.tif')

    slide = open_slide(slide_path)
    print ("Read WSI from %s with width: %d, height: %d" % (slide_path, 
                                                          slide.level_dimensions[0][0], 
                                                          slide.level_dimensions[0][1]))

    tumor_mask = open_slide(tumor_mask_path)
    print ("Read tumor mask from %s" % (tumor_mask_path))

    print("Slide includes %d levels", len(slide.level_dimensions))

    for i in range(len(slide.level_dimensions)-1):
        x = slide.level_dimensions[i][0]
        y = slide.level_dimensions[i][1]

        print("Level %d, dimensions: %s downsample factor %d" % (i, 
                                                               slide.level_dimensions[i], 
                                                               slide.level_downsamples[i]))

        downsample = 2**i
        dim = 299.

        dat.append({'image': image_number, 'level': i, 'downsample factor': downsample, 'x': x, 'y': y,                     'max windows': int(round((x*y)/dim**2,0)-1)})
        
    return slide_path, tumor_mask_path


def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

train_names = ['tumor_091', 'tumor_002', 'tumor_005', 'tumor_094', 'tumor_078', 'tumor_023', 
               'tumor_081', 'tumor_001', 'tumor_035', 'tumor_012', 'tumor_057', 'tumor_096', 
               'tumor_101', 'tumor_031', 'tumor_059', 'tumor_084','tumor_016', 'tumor_064']
holdout_names = ['tumor_110', 'tumor_075', 'tumor_019']


def is_tissue_in_window(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    #return zip(indices[0], indices[1])
    return len(indices) > 0


from keras.models import load_model

def run_inference(level, image_name, model, scaling = False):
    
    #reading in the slides and masks B@ODW
    slide_path = os.path.join(SLIDES_DIR, image_name+'.tif')
    tumor_mask_path =  os.path.join(SLIDES_DIR, image_name+'_mask.tif')
    print(slide_path)
    print(tumor_mask_path)

    #opening them
    print('opening slide & mask..')
    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)
    
    predicted_mask = np.zeros((tumor_mask.level_dimensions[level][1], tumor_mask.level_dimensions[level][0]))
    print('predicted mask shape: ' + str(predicted_mask.shape))
    
    slide_image = read_slide(slide, 
                         x=0, 
                         y=0, 
                         level=level, 
                         width=slide.level_dimensions[level][0], 
                         height=slide.level_dimensions[level][1])
    # showing the mask for test image at chosen level
    mask_image = read_slide(tumor_mask, 
                        x=0, 
                        y=0, 
                        level=level, 
                        width=tumor_mask.level_dimensions[level][0], 
                        height=tumor_mask.level_dimensions[level][1])[:,:,0]

    #dimensions at chosen level
    x_max = int(tumor_mask.level_dimensions[level][0]*.85)
    y_max = int(tumor_mask.level_dimensions[level][1]*.85)
    
    print('y: ' + str(y_max) + ' x: ' + str(x_max))

    # choosing downsample factor
    downsample_factor = int(slide.level_downsamples[level])

    #initiatize coordinates
    x0 = int(.15*x_max)
    y0 = int(.15*y_max)

    #size of window
    x_dim, y_dim = 299., 299.
    size = (int(x_dim), int(y_dim))

    #calculate how many steps we can take with 299x299 window
    #x_steps, y_steps = int(((x_max-x0) / x_dim)-1), int(((y_max-y0) / y_dim)-1)
    x_steps, y_steps = int(((x_max-x0) / x_dim)), int(((y_max-y0) / y_dim)) # I don't think we need the -1 above since int() already rounds down
    print('x steps: '+ str(x_steps) + ' y steps: '+ str(y_steps))

    for i in range(x_steps):
        
        # reset y0 to start
        y0 = int(.15*y_max)
        
        for j in range(y_steps): 
            #generating a window from the original slide
            window = read_slide(slide, 
                             x=int(x0)*downsample_factor, 
                             y=int(y0)*downsample_factor, 
                             level=level, 
                             width=int(x_dim), 
                             height=int(y_dim))

            if is_tissue_in_window(window) is True:
                window_reshaped = window.reshape(1, 299, 299, 3)
                
                if scaling == True:
                    window_reshaped = window_reshaped / 255 # scaling
                
                pred = model.predict(window_reshaped)
            
                if pred > 0.5:
                    predicted_mask[int(x0):(int(x0) + int(x_dim)), int(y0):(int(y0) + int(y_dim))] = 1

            #move the sliding window on y axis
            y0 = y0 + y_dim

        #move the sliding window on x axis
        x0 = x0 + x_dim
       
    return predicted_mask, slide_image, mask_image


SLIDES_DIR = '/domino/datasets/local/med_images/slides'
#print("Slides path within the drive: {}".format(SLIDES_DIR))
#holdout_names = ['tumor_110', 'tumor_075', 'tumor_019']

#model_inf = load_model('/domino/datasets/goyetc/medical-imaging/scratch/dense_model_baseline_level_3.h5')
model_inf = load_model('/domino/datasets/local/med_models/CNN__small_level_3.h5')
#/domino/datasets/local/med_models
image_selected = random.choice(holdout_names)

predicted_mask, slide_image, mask_image = run_inference(3, image_selected, model = model_inf)

plt.figure(figsize=(10,10), dpi=100)
plt.imshow(slide_image)
plt.imshow(mask_image, cmap='OrRd', alpha=0.5)
plt.imshow(predicted_mask, cmap='OrRd', alpha=0.3)
fig4 = plt.gcf()
fig4.savefig('/mnt/results/inference_prediction_example_baseline_level_3_'+str(image_selected+'.png'), dpi = 100)

