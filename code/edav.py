import json

import matplotlib.pyplot as plt
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
import os
from PIL import Image
from skimage.color import rgb2gray
import re
import itertools

import os
import pandas as pd
import random

import sys

import glob
 
# load all files from input datasets as pandas dataframes

SLIDES_DIR = "/domino/datasets/image_slides_input/slides/"
'''data_files_list = glob.glob(SLIDES_DIR+"*")

# saving image paths into a list
image_names = []

for images in data_files_list:
    img = re.findall(r'tumor_\d\d\d', images)
    for i in img:
        image_names.append(i)
image_names = list(set(image_names)) # removing duplicates

print(image_names)'''

image_names = ['tumor_035', 'tumor_101', 'tumor_096', 'tumor_023', 
                'tumor_059', 'tumor_012', 'tumor_031', 'tumor_094', 'tumor_091', 
                'tumor_016', 'tumor_005', 'tumor_110', 'tumor_078', 'tumor_019', 
                'tumor_001', 'tumor_057', 'tumor_084', 'tumor_075', 'tumor_081', 
                'tumor_064', 'tumor_002']

# ## Data Exploration & Visualization
# * Note that the sample code has been augmented to automate visualization
# * Note dataframe with downsampling factors and basic calculations on max # of windows available at each zoom level
# * Note POC of overlay and tissue identification

# #### functions

# Download an example slide and tumor mask

# Important note: the remainder are in a Google Drive folder, linked above.
# You will need to host them on your own, either in Google Drive, or by using
# the cloud provider of your choice.

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
        
        #assert tumor_mask.level_dimensions[i][0] == x
        #assert tumor_mask.level_dimensions[i][1] == y

        downsample = 2**i
        dim = 299.

        dat.append({'image': image_number, 'level': i, 'downsample factor': downsample, 'x': x, 'y': y,                     'max windows': int(round((x*y)/dim**2,0)-1)})
        #Dict[temp] = (float(slide.level_dimensions[i][0]), float(slide.level_dimensions[i][1]))

    # Verify downsampling works as expected
    #width, height = slide.level_dimensions[7]
    #assert width * slide.level_downsamples[7] == slide.level_dimensions[0][0]
    #assert height * slide.level_downsamples[7] == slide.level_dimensions[0][1]

    return slide_path, tumor_mask_path

# See https://openslide.org/api/python/#openslide.OpenSlide.read_region
# Note: x,y coords are with respect to level 0.
# There is an example below of working with coordinates
# with respect to a higher zoom level.

# Read a region from the slide
# Return a numpy RBG array

def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

def zoom(i): 
  
  slide_image = read_slide(slide, 
                         x=0, 
                         y=0, 
                         level=i, 
                         width=slide.level_dimensions[i][0], 
                         height=slide.level_dimensions[i][1])
     

  # Example: read the entire mask at the same zoom level
  mask_image = read_slide(tumor_mask, 
                       x=0, 
                       y=0, 
                       level=i, 
                       width=tumor_mask.level_dimensions[i][0], 
                       height=tumor_mask.level_dimensions[i][1])

  # Note: the program provided by the dataset authors generates a mask with R,G,B channels.
  # The mask info we need is in the first channel only.
  # If you skip this step, the mask will be displayed as all black.
  mask_image = mask_image[:,:,0]

  return slide_image, mask_image


def show_slide():
  
  plt.figure('slide', figsize=(10,10))
  plt.imshow(slide_image)
  plt.savefig('/mnt/results/base_'+str(selected_image)+'.png')

  return 'slide image'


def show_mask():

  plt.figure('mask', figsize=(10,10))
  plt.imshow(mask_image)
  plt.savefig('/mnt/results/mask_'+str(selected_image)+'.png')
  
  return 'mask'


def show_overlay(a):
  
  plt.figure(figsize=(10,10))
  #plt.gcf()
  plt.imshow(slide_image)
  plt.imshow(mask_image, cmap = 'OrRd', alpha = a)
  plt.savefig('/mnt/results/overlay_'+str(selected_image)+'.png')

  
  return 'overlay'

# As mentioned in class, we can improve efficiency by ignoring non-tissue areas 
# of the slide. We'll find these by looking for all gray regions.

def find_tissue_pixels(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return zip(indices[0], indices[1])


def apply_mask(im, mask, color=(255,0,0)):
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked


# #### Workflow with image

dat = []

try:
    selected_image = sys.argv[1]
except:
    selected_image = random.choice(image_names)
    
choose_image(selected_image[-3:])

df=pd.DataFrame(dat)
df

df.to_json(r'/mnt/dominostats.json')


# Example: read the entire slide at level 5
# 
# Higher zoom levels may not fit into memory. You can use the below function to extract regions from higher zoom levels without having to read the entire image into ram.
# 
# Use the sliding window approach discussed in class to collect training data for your classifier. E.g., slide a window across the slide (for starters, use a zoomed in view, so you're not working with giant images). Save each window to disk as an image. To find the label for that image, check to the tissue mask to see if the same region contains cancerous cells.
# 
# Important: this is tricky to get right. Carefully debug your pipeline beforetraining your model. Start with just a single image, and a relatively low zoom level.

# In[48]:

slide_image, mask_image = zoom(5)

# checking dimentions
slide_image.shape

# checking dimentions
mask_image.shape

show_slide()

show_mask()

show_overlay(0.6)

