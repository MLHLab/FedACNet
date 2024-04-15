#!/usr/bin/env python
# coding: utf-8

# In[16]:


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

# In[17]:


data_dir = pathlib.Path("C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViral")
image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[18]:


#Rescaling the image
image_convert = ImageDataGenerator(rescale = 1./255)

#Flow the images from the input directory into an array, applies the pre-processing defined in image_convert function, resizes all images in same size 
#image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[3, 3], class_mode="binary", batch_size=1000,seed=32)
image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[180, 180], class_mode="binary", batch_size=1000,seed=32)
data, labels = image_preprocess.next()


# In[19]:


labels = labels. astype(int)
print(type(data))
print(data.shape)
print(type(labels))
print(labels.shape)
print(sum(labels))

print(labels)


# In[20]:


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=40)

print(X_train.shape)
print(y_train.shape)
print(sum(y_train))

print(X_test.shape)
print(y_test.shape)
print(sum(y_test))


# In[21]:


#new code
import keras
#from keras.models import Model
#from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
#    BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D,     BatchNormalization, concatenate, AveragePooling2D

def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(filters, growth_rate, classes, dense_block_size, layers_in_block):
    input_img = Input(shape=(180, 180, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(1, activation='softmax')(dense_x)

    return Model(input_img, output)


# In[22]:


dense_block_size = 3
layers_in_block = 4

growth_rate = 12
classes = 2
model = dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
model.summary()


# training
batch_size = 32
epochs = 10
#optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


# In[23]:


model.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Train the model

r = model.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, shuffle=True)


# In[60]:


# Print final model performance

print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[64]:


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


# In[65]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')

