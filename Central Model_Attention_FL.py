#!/usr/bin/env python
# coding: utf-8

# In[1]:


import socket
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime


# # ITERATION 1

# In[2]:


print('***************************** Server Execution Starts ***************************** ')
now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution is starting =", current_time)


# In[3]:


# Read model weight from Clinet1
Client1_weight = pickle.load(open('Client1_USA.pkl', 'rb'))


# In[4]:


print('*****************************************************************************')
print(Client1_weight[16])


# In[ ]:





# In[5]:


# Read model weight from Clinet2
Client2_weight = pickle.load(open('Client2_China.pkl', 'rb'))

print('*****************************************************************************')
print(Client2_weight[16])


# In[ ]:





# In[6]:


# Derive model weight from the weights received from the clinets
agg_weight_arr = np.array(Client1_weight) + np.array(Client2_weight)
Central_weight=(1/2)*np.array(agg_weight_arr)


# In[7]:


print('*****************************************************************************')
print(Central_weight[0])


# In[8]:


print('*****************************************************************************')
print(Central_weight[16])


# In[9]:


# Save model weights
pickle.dump(Central_weight, open('Central_Server.pkl', 'wb'))


# In[10]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution is ending =", current_time)
print('***************************** Server Execution Ends ***************************** ')


# In[ ]:





# # ITERATION 2

# In[11]:


print('***************************** Server Execution Starts ***************************** ')
now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution 2 is starting =", current_time)


# In[12]:


# Read model weight from Clinet1
Client1_weight = pickle.load(open('Client1_USA2.pkl', 'rb'))

print('*****************************************************************************')
print(Client1_weight[0])


# In[13]:


print('*****************************************************************************')
print(Client1_weight[16])


# In[14]:


# Read model weight from Clinet2
Client2_weight = pickle.load(open('Client2_China2.pkl', 'rb'))

print('*****************************************************************************')
print(Client2_weight[0])


# In[15]:


print('*****************************************************************************')
print(Client2_weight[16])


# In[ ]:





# In[16]:


# Derive model weight from the weights received from the clinets
agg_weight_arr = np.array(Client1_weight) + np.array(Client2_weight)
Central_weight=(1/2)*np.array(agg_weight_arr)

print('*****************************************************************************')
print(Central_weight[0])


# In[17]:


print('*****************************************************************************')
print(Central_weight[16])


# In[18]:


# Save model weights
pickle.dump(Central_weight, open('Central_Server2.pkl', 'wb'))


# In[19]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution 2 is ending =", current_time)
print('***************************** Server Execution Ends ***************************** ')


# In[ ]:





# In[ ]:





# In[ ]:




