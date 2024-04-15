#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import os
import shutil


# In[2]:


df_input = pd.read_excel("Reffile1.xlsx")
print(df_input.shape)
df_input.head()


# In[3]:


df_mod = df_input[df_input['file .gz extention']!='Y']
df_mod.shape


# In[4]:


df_mod = df_mod[df_mod['disease_hierarch1']=='NonViral']
df_mod.shape


# In[5]:


# get distinct countries
print(df_mod['Country_mod2'].value_counts())
print(df_mod['Country_mod2'].value_counts(normalize=True))


# In[6]:


df_2 = df_mod[df_mod['Country_mod2'].isin(['Australia','Spain'])]
print(df_2['Country_mod2'].value_counts())
print('\n')

print(df_2['Country_mod2'].value_counts(normalize=True))
print('\n')

distinct_Country_mod2 = df_2['Country_mod2'].unique()
print(distinct_Country_mod2)


# In[7]:


# Create new folder for each Country if the folder down't exist already

for Country_mod2 in distinct_Country_mod2:
    newPath1 = os.path.join("Countrywise", Country_mod2, "TargetBacterial")
    newPath2 = os.path.join("Countrywise", Country_mod2, "OtherFungalLipoid")
    if not os.path.exists(newPath1):
        os.makedirs(newPath1)
    if not os.path.exists(newPath2):
        os.makedirs(newPath2)


# In[8]:


# Read each filename from the given directory, get the continent name of each file using the reference file 
# and then copy to the newly created continent specific folder

dirName = "C:/Users/tapom/Desktop/Research/2_Code/covid-chestxray-dataset/images"

for root, dirs, files in os.walk(dirName, topdown=False):
    for file in files:        
        if len(file)!=0:
            cont1 = df_2[df_2['filename']==file]['Country_mod2']
            cont2 = df_2[df_2['filename']==file]['disease_hierarch0']
            if not cont1.empty:                
                if cont2.values[0]=='Bacterial':
                    shutil.copy2(os.path.join(root,file), os.path.join("Countrywise", cont1.values[0],"TargetBacterial"))
                else:
                    shutil.copy2(os.path.join(root,file), os.path.join("Countrywise", cont1.values[0],"OtherFungalLipoid"))

