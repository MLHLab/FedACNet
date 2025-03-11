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
from sklearn.metrics import confusion_matrix, classification_report

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


# In[2]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is starting =", current_time)
print('***************************** Iteration 1 Starts ***************************** ')


# In[3]:


data_dir = pathlib.Path("C:/Users/tapom/Desktop/FullData/FullData")
image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[4]:


#Rescaling the image
image_convert = ImageDataGenerator(rescale = 1./255)

#Flow the images from the input directory into an array, applies the pre-processing defined in image_convert function, resizes all images in same size 
#image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[3, 3], class_mode="binary", batch_size=1000,seed=32)
image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[180, 180], class_mode='categorical', batch_size=4000,seed=32)
data, labels = image_preprocess.next()


# In[5]:


labels = labels. astype(int)
print(type(data))
print(data.shape)
print(type(labels))
print(labels.shape)
print(sum(labels))

print(labels)


# In[6]:


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=30)

print(X_train.shape)
print(y_train.shape)
print(sum(y_train))

print(X_test.shape)
print(y_test.shape)
print(sum(y_test))


# In[7]:


'''
# Define the model

model = tf.keras.Sequential([                             
  tf.keras.layers.Flatten(input_shape=(3, 3, 3)),
  tf.keras.layers.Dense(4, activation='relu'),
  tf.keras.layers.Dense(2)
])

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''


# In[8]:


#new code
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(2019)


# In[9]:


'''
#new code
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (180,180,3)) ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(550,activation="relu"),      #Adding the Hidden layer
    tf.keras.layers.Dropout(0.1,seed = 2019),
    tf.keras.layers.Dense(400,activation ="relu"),
    tf.keras.layers.Dropout(0.3,seed = 2019),
#    tf.keras.layers.Dense(300,activation="relu"),
#    tf.keras.layers.Dropout(0.4,seed = 2019),
    tf.keras.layers.Dense(200,activation ="relu"),
    tf.keras.layers.Dropout(0.2,seed = 2019),
    tf.keras.layers.Dense(1,activation = "sigmoid")   #Adding the Output Layer
])
'''


# In[10]:


#build the Resnet model 
resnet = ResNet50(weights='imagenet',
                      input_shape= (180,180,3),
                      include_top= False)  


#show the base model summary 
resnet.summary()
#show how manay layers in the Resnet Network
layers = resnet.layers
print(f'Number of Layers: {len(layers)} ')


# In[11]:


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
outputs = Dense(3, activation ='softmax')(x)
# this is the model we will train
model = Model(inputs=inputs, outputs=outputs)

# freeze all convolutional Resnet layers
#for layer in layers:
#    layer.trainable = False
# compile the model 
model.compile(optimizer= 'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[12]:


# Train the model
r = model.fit(X_train, y_train, epochs=10,batch_size=100)


# In[13]:


# Print final model performance

print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[14]:


# predict probabilities for test set
predictions =model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

y_pred


# In[15]:


# print image labels
labels = {value: key for key, value in image_preprocess.class_indices.items()}
labels


# In[16]:


# convert test values into categories
y_test2 = np.argmax(y_test, axis=1)
y_test2


# In[17]:


print(classification_report(y_test2, y_pred, target_names=labels.values()))


# In[18]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')


# In[ ]:





# In[ ]:




