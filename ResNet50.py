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



import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow import reshape, nest, config
from tensorflow.keras import losses, metrics, optimizers

from sklearn.model_selection import train_test_split

import pickle

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

from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.metrics import confusion_matrix,classification_report
from keras.optimizers import Adam

from tensorflow.keras.models import Model, Sequential

from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.metrics import confusion_matrix,classification_report
from keras.optimizers import Adam
#from tensorflow.python.keras import Adam


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
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=40)

print(X_train.shape)
print(y_train.shape)
print(sum(y_train))

print(X_test.shape)
print(y_test.shape)
print(sum(y_test))


# In[7]:


#new code
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(2019)


# In[9]:


#build the Resnet model 
resnet = ResNet50(weights='imagenet',
                      input_shape= (180,180,3),
                      include_top= False)  


#show the base model summary 
resnet.summary()
#show how manay layers in the Resnet Network
layers = resnet.layers
print(f'Number of Layers: {len(layers)} ')


# In[10]:


# let's train our Model 
inputs = resnet.input
# add an average pooling layer
x = resnet.output
x = GlobalAveragePooling2D()(x)
#first dense layer
x = Dense(512, activation='relu')(x)
#dropout 
x = Dropout(0.5)(x)
# output layer
outputs = Dense(1, activation ='softmax')(x)
# this is the model we will train
model = Model(inputs=inputs, outputs=outputs)

# freeze all convolutional Resnet layers
#for layer in layers:
#    layer.trainable = False
# compile the model 
model.compile(optimizer= 'adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[11]:


# Train the model
r = model.fit(X_train, y_train, epochs=5,batch_size=10)


# In[12]:


# Print final model performance

print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[13]:


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


# In[14]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')

