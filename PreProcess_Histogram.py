#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import glob
import numpy as np
import os


# In[5]:



root1="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViral/TargetViral/*"
root2="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViralPre/TargetViral/"

root3="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViral/OtherNonviral/*"
root4="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViralPre/OtherNonviral/"

root5="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViralHoldOut/TargetViral/*"
root6="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViralHoldOutPre/TargetViral/"

root7="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViralHoldOut/OtherNonviral/*"
root8="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataViralHoldOutPre/OtherNonviral/"


# In[12]:


'''
root1="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovid/TargetCovid/*"
root2="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovidPre/TargetCovid/"

root3="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovid/OtherNoncovid/*"
root4="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovidPre/OtherNoncovid/"

root5="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovidHoldOut/TargetCovid/*"
root6="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovidHoldOutPre/TargetCovid/"

root7="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovidHoldOut/OtherNoncovid/*"
root8="C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovidHoldOutPre/OtherNoncovid/"
'''


# In[13]:


file=0

ext=".jpeg"

for img in glob.glob(root1 + ext):
    inp= cv2.imread(img)

    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root2 + str(file) + ext)
    cv2.imwrite(path, mod)
    
ext=".jpg"

for img in glob.glob(root1 + ext):
    inp= cv2.imread(img)
    
    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root2 + str(file) + ext)
    cv2.imwrite(path, mod)
    
ext=".png"

for img in glob.glob(root1 + ext):
    inp= cv2.imread(img)
        
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root2 + str(file) + ext)
    cv2.imwrite(path, mod)


# In[14]:


file=0

ext=".jpeg"

for img in glob.glob(root3 + ext):
    inp= cv2.imread(img)

    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root4 + str(file) + ext)
    cv2.imwrite(path, mod)
    
ext=".jpg"

for img in glob.glob(root3 + ext):
    inp= cv2.imread(img)
    
    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root4 + str(file) + ext)
    cv2.imwrite(path, mod)
    
ext=".png"

for img in glob.glob(root3 + ext):
    inp= cv2.imread(img)
    
        
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root4 + str(file) + ext)
    cv2.imwrite(path, mod)


# In[15]:


file=0

ext=".jpeg"

for img in glob.glob(root5 + ext):
    inp= cv2.imread(img)

    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root6 + str(file) + ext)
    cv2.imwrite(path, mod)
    
ext=".jpg"

for img in glob.glob(root5 + ext):
    inp= cv2.imread(img)
    
    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root6 + str(file) + ext)
    cv2.imwrite(path, mod)
    
ext=".png"

for img in glob.glob(root5 + ext):
    inp= cv2.imread(img)
    
    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------

    
    file=file+1
    path=(root6 + str(file) + ext)
    cv2.imwrite(path, mod)


# In[16]:


file=0

ext=".jpeg"

for img in glob.glob(root7 + ext):
    inp= cv2.imread(img)

   
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------

    
    file=file+1
    path=(root8 + str(file) + ext)
    cv2.imwrite(path, mod)
    
ext=".jpg"

for img in glob.glob(root7 + ext):
    inp= cv2.imread(img)
    
    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------
    
    file=file+1
    path=(root8 + str(file) + ext)
    cv2.imwrite(path, mod)
    
ext=".png"

for img in glob.glob(root7 + ext):
    inp= cv2.imread(img)
    
    
    #Histogram block ----------------------------------------
    R, G, B = cv2.split(inp)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    mod = cv2.merge((output1_R, output1_G, output1_B))
    #--------------------------------------------------------

    
    file=file+1
    path=(root8 + str(file) + ext)
    cv2.imwrite(path, mod)


# In[ ]:




