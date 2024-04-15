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
from sklearn.metrics import confusion_matrix

from numpy.random import seed

# Set seeds

seed(1)
tf.random.set_seed(2)


# # Data Viral

# In[2]:


data_dir = pathlib.Path("C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovid")
image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[3]:


#Rescaling the image
image_convert = ImageDataGenerator(rescale = 1./255)

#Flow the images from the input directory into an array, applies the pre-processing defined in image_convert function, resizes all images in same size 
#image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[3, 3], class_mode="binary", batch_size=1000,seed=32)
image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[180, 180], class_mode="binary", batch_size=1000,seed=32)
data, labels = image_preprocess.next()


# In[4]:


labels = labels. astype(int)
print(type(data))
print(data.shape)
print(type(labels))
print(labels.shape)
print(sum(labels))

print(labels)


# In[5]:


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=35)

print(X_train.shape)
print(y_train.shape)
print(sum(y_train))

print(X_test.shape)
print(y_test.shape)
print(sum(y_test))


# In[6]:


#new code
import keras
#from keras.models import Model
#from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
#    BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D,     BatchNormalization, concatenate, AveragePooling2D

vgg16_weight_path = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = tf.keras.applications.VGG16(
    #weights=vgg16_weight_path,
    include_top=False,
    #input_shape=IMG_SIZE + (3,)
    input_shape=(180, 180, 3)
)

model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.layers[0].trainable = False

model.summary()


# In[7]:


model.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Train the model

r = model.fit(X_train,y_train, epochs=10, batch_size=20, shuffle=True)


# In[ ]:





# In[9]:


# Print final model performance

print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[10]:


# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
#yhat_classes = model.predict_classes(X_test, verbose=0)


yhat_classes_1 =model.predict(X_test, verbose=0)

yhat_classes = np.where(yhat_classes_1 > 0.5, 1,0)

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# In[11]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')


# In[ ]:





# In[ ]:




