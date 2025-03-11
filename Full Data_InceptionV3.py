#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import loadtxt
import os
import scipy
import collections
import pathlib
import matplotlib.pyplot as plt
import PIL.Image
from datetime import datetime

from sklearn.model_selection import train_test_split

import pickle
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,  classification_report

from numpy.random import seed

# Set seeds

seed(1)
tf.random.set_seed(2)


# In[2]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is starting =", current_time)
print('***************************** Iteration 1 Starts ***************************** ')


# In[3]:


#!pip install tf-slim


# In[4]:


data_dir = pathlib.Path("C:/Users/tapom/Desktop/FullData/FullData")
image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[5]:


#Rescaling the image
image_convert = ImageDataGenerator(rescale = 1./255)

#Flow the images from the input directory into an array, applies the pre-processing defined in image_convert function, resizes all images in same size 
#image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[3, 3], class_mode="binary", batch_size=1000,seed=32)
image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[180, 180], class_mode='categorical', batch_size=4000,seed=32)
data, labels = image_preprocess.next()


# In[6]:


labels = labels. astype(int)
print(type(data))
print(data.shape)
print(type(labels))
print(labels.shape)
print(sum(labels))

print(labels)


# In[7]:


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=35)

print(X_train.shape)
print(y_train.shape)
print(sum(y_train))

print(X_test.shape)
print(y_test.shape)
print(sum(y_test))


# In[8]:


#new code
#Importing Libraries
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
#from keras.models import Sequential,Model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,GlobalMaxPooling2D,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import random


# In[9]:


#Inception Model (Pre-Trained)
#local_weights_file = "../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model = InceptionV3(input_shape=(180,180,3),include_top=False,weights=None)
#pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable=False
pre_trained_model.summary()


# In[10]:


last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
x = GlobalMaxPooling2D()(last_output)
x = Dense(1024,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(3,activation='softmax')(x) 

model = Model(pre_trained_model.input,x)
model.summary()


# In[11]:


model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Train the model

r = model.fit(X_train,y_train, epochs=10, batch_size=100, shuffle=True)


# In[12]:


# Print final model performance

print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[13]:


# predict probabilities for test set
predictions =model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

y_pred


# In[14]:


# print image labels
labels = {value: key for key, value in image_preprocess.class_indices.items()}
labels


# In[15]:


# convert test values into categories
y_test2 = np.argmax(y_test, axis=1)
y_test2


# In[16]:


print(classification_report(y_test2, y_pred, target_names=labels.values()))


# In[17]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')


# In[ ]:





# In[ ]:




