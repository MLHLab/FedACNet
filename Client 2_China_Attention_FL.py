#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import loadtxt
import os
import scipy
import collections
import pathlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import PIL.Image
from datetime import datetime

import tensorflow as tf

import keras
#from keras.models import Sequential, Model
#from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout, Input, multiply

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Activation 
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
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
tf.compat.v1.set_random_seed(2019)


# # ITERATION 1

# In[2]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is starting =", current_time)
print('***************************** Iteration 1 Starts ***************************** ')


# In[3]:


data_dir = pathlib.Path("C:/Users/tapom/Desktop/FullData/China")
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


def SqueezeAndExcitation(inputs, ratio=8):
    b, _, _, c = inputs.shape
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(c//ratio, activation="relu", use_bias=False)(x)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = multiply([inputs,x])
    return x


# In[8]:


input_img = Input(shape=(180, 180, 3))

x = Conv2D(16, (3, 3), activation='relu')(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = SqueezeAndExcitation(x)                                    #Squeeze and Excitation Attention Layer

x = Flatten()(x)

x = Dense(550, activation='relu')(x)
x = Dropout(0.1,seed=2019)(x)

x = Dense(400, activation='relu')(x)
x = Dropout(0.3,seed=2019)(x)

x = Dense(200, activation='relu')(x)
x = Dropout(0.2,seed=2019)(x)

output = Dense(3, activation='softmax')(x)                  # Add Output layer

model = Model(inputs=input_img, outputs=output)


# In[9]:


# Print the model summary
model.summary()


# In[10]:


model.compile(optimizer='Adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model

r = model.fit(X_train,y_train, epochs=40, batch_size=1000, shuffle=True)


# In[11]:


# Print final model performance
print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[12]:


# predict probabilities for test set
predictions =model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

y_pred


# In[13]:


# print image labels
labels = {value: key for key, value in image_preprocess.class_indices.items()}
labels


# In[14]:


# convert test values into categories
y_test2 = np.argmax(y_test, axis=1)
y_test2


# In[15]:


print(classification_report(y_test2, y_pred, target_names=labels.values()))


# In[16]:


# Save model weights
pickle.dump(model.get_weights(), open('Client2_China.pkl', 'wb'))


# In[17]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')


# In[ ]:





# In[ ]:





# In[ ]:





# # ITERATION 2

# In[18]:


print('***************************** Iteration 2 Starts ***************************** ')
now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Iteration 2 is starting =", current_time)


# In[19]:


# Read model weight from Central Server
Central_weight = pickle.load(open('Central_Server.pkl', 'rb'))

#apply new weights to model
model.set_weights(Central_weight)


# In[20]:


print('*****************************************************************************')
print(model.weights[0])


# In[21]:


print('*****************************************************************************')
print(model.weights[16])


# In[22]:


# Train the model
r = model.fit(X_train,y_train, epochs=40, batch_size=1000, shuffle=True)


# In[23]:


# Print final model performance
print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[24]:


# predict probabilities for test set
predictions =model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

y_pred


# In[25]:


# print image labels
labels = {value: key for key, value in image_preprocess.class_indices.items()}
labels


# In[26]:


# convert test values into categories
y_test2 = np.argmax(y_test, axis=1)
y_test2


# In[27]:


print(classification_report(y_test2, y_pred, target_names=labels.values()))


# In[28]:


# Save model weights
pickle.dump(model.get_weights(), open('Client2_China2.pkl', 'wb'))


# In[29]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Iteration 2 is ending =", current_time)
print('***************************** Iteration 2 Ends ***************************** ')


# In[ ]:





# # Test with Individual Models

# In[30]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Test is starting =", current_time)
print('***************************** Test Starts ***************************** ')


# # Test 1, Model - Central

# In[31]:


# Read model weight
weight = pickle.load(open('Central_Server2.pkl', 'rb'))

#apply new weights to model
model.set_weights(weight)


# In[32]:


print("Test score:", model.evaluate(X_test, y_test))


# In[33]:


# predict probabilities for test set
predictions =model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

y_pred


# In[34]:


# print image labels
labels = {value: key for key, value in image_preprocess.class_indices.items()}
labels


# In[35]:


# print image labels
labels = {value: key for key, value in image_preprocess.class_indices.items()}
labels


# In[36]:


# convert test values into categories
y_test2 = np.argmax(y_test, axis=1)
y_test2


# In[37]:


print(classification_report(y_test2, y_pred, target_names=labels.values()))


# # Test 2, Model - USA

# In[38]:


# Read model weight
weight = pickle.load(open('Client1_USA2.pkl', 'rb'))

#apply new weights to model
model.set_weights(weight)


# In[39]:


print("Test score:", model.evaluate(X_test, y_test))


# In[40]:


# predict probabilities for test set
predictions =model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

y_pred


# In[41]:


# print image labels
labels = {value: key for key, value in image_preprocess.class_indices.items()}
labels


# In[42]:


# convert test values into categories
y_test2 = np.argmax(y_test, axis=1)
y_test2


# In[43]:


print(classification_report(y_test2, y_pred, target_names=labels.values()))


# In[ ]:




