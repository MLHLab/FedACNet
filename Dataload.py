#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Installing libraries - Run once

# !pip -q install torchxrayvision


# In[2]:


# Import libraries

import torchxrayvision as xrv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# Clone the git respository in local

get_ipython().system('git clone https://github.com/ieee8023/covid-chestxray-dataset')


# In[4]:


d = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset/images/",csvpath="covid-chestxray-dataset/metadata.csv")


# In[6]:


sample = d[10]


# In[7]:


plt.imshow(sample["img"][0], cmap="gray");

