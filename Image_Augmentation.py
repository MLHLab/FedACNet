#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from numpy import loadtxt
import os
import scipy
import collections
import pathlib
import matplotlib.pyplot as plt
import PIL.Image

from keras.preprocessing.image import ImageDataGenerator


# In[13]:


data_dir = pathlib.Path("C:/Users/tapom/Desktop/Research/2_Code/ImageAug")
image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[ ]:





# In[14]:


datagen = ImageDataGenerator( 
        rotation_range = 5, 
        brightness_range = (0.5, 1.5)) 


# In[15]:


#Setting DataGenerator with respective augmentation parameters
image_convert = ImageDataGenerator(rotation_range = 5)
                                   #brightness_range = (0.5, 1.5))


# In[16]:


image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[180, 180], batch_size=1000,seed=32)
data, labels = image_preprocess.next()
print(data.shape)


# In[17]:


i=0
for batch in datagen.flow(
    data,
    batch_size=1,
    save_to_dir='C:/Users/tapom/Desktop/Research/2_Code/ImageAug/Output',
    save_prefix='Augmented_image_A',
    save_format='jpeg'):
    i += 1
    if i > 99: # create mentioned no. of augmented images
        break  # otherwise


# In[ ]:




