#!/usr/bin/env python
# coding: utf-8

# In[33]:


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
from sklearn.metrics import confusion_matrix,  classification_report

from numpy.random import seed

# Set seeds

seed(1)
tf.random.set_seed(2)
tf.compat.v1.set_random_seed(2019)


# In[34]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is starting =", current_time)
print('***************************** Iteration 1 Starts ***************************** ')


# In[35]:


data_dir = pathlib.Path("C:/Users/tapom/Desktop/FullData/FullData")
image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[36]:


#Rescaling the image
image_convert = ImageDataGenerator(rescale = 1./255)

#Flow the images from the input directory into an array, applies the pre-processing defined in image_convert function, resizes all images in same size 
#image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[3, 3], class_mode="binary", batch_size=1000,seed=32)
image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[180, 180], class_mode='categorical', batch_size=4000,seed=32)
data, labels = image_preprocess.next()


# In[37]:


labels = labels. astype(int)
print(type(data))
print(data.shape)
print(type(labels))
print(labels.shape)
print(sum(labels))

print(labels)


# In[38]:


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=30)

print(X_train.shape)
print(y_train.shape)
print(sum(y_train))

print(X_test.shape)
print(y_test.shape)
print(sum(y_test))


# In[39]:


def SqueezeAndExcitation(inputs, ratio=8):
    b, _, _, c = inputs.shape
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(c//ratio, activation="relu", use_bias=False)(x)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = multiply([inputs,x])
    return x


# In[40]:


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


# In[41]:


# Print the model summary
model.summary()


# In[42]:


model.compile(optimizer='Adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
r = model.fit(X_train,y_train, epochs=40, batch_size=1000, shuffle=True)


# In[43]:


# Print final model performance

print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[44]:


# predict probabilities for test set
predictions =model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

y_pred


# In[45]:


# print image labels
labels = {value: key for key, value in image_preprocess.class_indices.items()}
labels


# In[46]:


# convert test values into categories
y_test2 = np.argmax(y_test, axis=1)
y_test2


# In[47]:


print(classification_report(y_test2, y_pred, target_names=labels.values()))


# In[48]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')


# In[55]:


from sklearn.metrics import roc_curve, auc
Y_true_bin = y_test
Y_true_bin


# In[56]:


Y_pred=predictions
Y_pred


# In[61]:


# Number of classes
n_classes = 3

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(Y_true_bin[:, i], Y_pred[:, i])
    print('*****************')
    print('**',i)
    print(fpr)
    print(tpr)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

# Plot the diagonal line (chance line)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest) for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




