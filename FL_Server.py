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
Client1_weight = pickle.load(open('Client1_Europe.pkl', 'rb'))


# In[17]:


print('*****************************************************************************')
print(Client1_weight[16])


# In[ ]:





# In[19]:


# Read model weight from Clinet2
Client2_weight = pickle.load(open('Client2_AsiaPacific.pkl', 'rb'))

print('*****************************************************************************')
print(Client2_weight[16])


# In[ ]:





# In[20]:


# Read model weight from Clinet3
Client3_weight = pickle.load(open('Client3_America.pkl', 'rb'))

print('*****************************************************************************')
print(Client3_weight[16])


# In[21]:


# Derive model weight from the weights received from the clinets
agg_weight_arr = np.array(Client1_weight) + np.array(Client2_weight) + np.array(Client3_weight)
Central_weight=(1/3)*np.array(agg_weight_arr)


# In[22]:


print('*****************************************************************************')
print(Central_weight[0])


# In[23]:


print('*****************************************************************************')
print(Central_weight[16])


# In[24]:


# Save model weights
pickle.dump(Central_weight, open('Central_Server.pkl', 'wb'))


# In[25]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution is ending =", current_time)
print('***************************** Server Execution Ends ***************************** ')


# In[ ]:





# # ITERATION 2

# In[26]:


print('***************************** Server Execution Starts ***************************** ')
now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution 2 is starting =", current_time)


# In[27]:


# Read model weight from Clinet1
Client1_weight = pickle.load(open('Client1_Europe2.pkl', 'rb'))

print('*****************************************************************************')
print(Client1_weight[0])


# In[28]:


print('*****************************************************************************')
print(Client1_weight[16])


# In[29]:


# Read model weight from Clinet2
Client2_weight = pickle.load(open('Client2_AsiaPacific2.pkl', 'rb'))

print('*****************************************************************************')
print(Client2_weight[0])


# In[30]:


print('*****************************************************************************')
print(Client2_weight[16])


# In[31]:


# Read model weight from Clinet3
Client3_weight = pickle.load(open('Client3_America2.pkl', 'rb'))

print('*****************************************************************************')
print(Client3_weight[0])


# In[32]:



print('*****************************************************************************')
print(Client3_weight[16])


# In[ ]:





# In[33]:


# Derive model weight from the weights received from the clinets
agg_weight_arr = np.array(Client1_weight) + np.array(Client2_weight) + np.array(Client3_weight)
Central_weight=(1/3)*np.array(agg_weight_arr)

print('*****************************************************************************')
print(Central_weight[0])


# In[34]:


print('*****************************************************************************')
print(Central_weight[16])


# In[35]:


# Save model weights
pickle.dump(Central_weight, open('Central_Server2.pkl', 'wb'))


# In[36]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution 2 is ending =", current_time)
print('***************************** Server Execution Ends ***************************** ')


# In[ ]:





# In[ ]:





# In[ ]:




