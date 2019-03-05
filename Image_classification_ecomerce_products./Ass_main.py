
# coding: utf-8

# In[1]:


from img2vec.img_to_vec import Img2Vec
from PIL import Image


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


# In[2]:


# loading Data
# loadind data into pandas
filename ="/media/parth/New Volume/p_data/word2vec_test.csv"

data = pd.read_csv(filename,error_bad_lines=False)


# removing extra string in the imageurl

data['imageUrlStr'] = data['imageUrlStr'].str.split(';').str[0]

# combining all the text into one column and droping the columns

data['combined'] = data['imageUrlStr']+data['productBrand']+data['color']+data['keySpecsStr']+data['sellerName']
data.drop(data.columns[[0,2,4,5,6,7,8,9,10]],inplace=True, axis=1)


# In[3]:


#Taking 500 rows in the complete data set
#data.shape


# In[4]:


data.head()


# In[5]:


data_10=data[:100]


# In[6]:


# using the image to vector code as library

img2vec = Img2Vec(cuda=True)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from random import randint
from time import sleep

#import IPython
#from IPython.display import Image
#from IPython.display import display
import urllib.request
import io


# In[7]:


# writing a function to real the imageurl and output image 

def openurl(urls):
    fd = urllib.request.urlopen(urls)
    sleep(randint(0,2))
    image_file = io.BytesIO(fd.read())
    return Image.open(image_file)


# converting the read images to vector form and extracting the 512 length array.
data_10['vector'] = data_10.apply(lambda x: img2vec.get_vec(openurl(x['imageUrlStr'])),axis=1)


# In[ ]:


# checking the shape of the output vector data
data_10['vector'].shape


# In[8]:


# Here we compare the cosine similaritybetween the image vector
# after comparing taking the productid as a key and append the similar imageids to it

data_dict = dict()

for i in range(len(data_10['vector'])):
    for j in range(i,len(data_10['vector'])):
        if i !=j:
            similarity = cosine_similarity(data_10['vector'][i].reshape(1,-1),data_10['vector'][j].reshape(1,-1))
            if similarity >0.99:
                if data_10['productId'][i] in data_dict:
                    data_dict[data_10['productId'][i]].append(data_10['productId'][j])
                else:
                    data_dict[data_10['productId'][i]] = [data_10['productId'][j]]


# In[9]:


# view the dictionary file
print(data_dict)


# In[10]:


# this piece of code is to view the images

import IPython
from IPython.display import Image
from IPython.display import display

#for i,urls in enumerate(data_10['imageUrlStr']):
#    display(Image(urls, width=50), i,data_10['productId'][i])


# In[12]:


## this piece of code is to see the images that are same by key and value

import IPython
from IPython.display import Image
from IPython.display import display


listof = ["TOPE7U8MGBCRV7QG","TOPE7U8MYVRRTMSU", "TOPE7U8MVFCJB6Z3"]

y = data[data['productId'].isin(listof)]

#for urls in y['imageUrlStr']:
#    display(Image(urls, width=50))
#y['imageUrlStr'][80]


# In[13]:


## writing the dictionary into output
import json
with open('data_10.json', 'w') as outfile:
    json.dump(data_dict, outfile)


# In[ ]:


@@
# I consered only 500 rows because i was facing the issues with the http 503 request errors.
# The is a high scope for optimising this code and solve in better ways with a bit of research.


# In[4]:


jupyter nbconvert Ass_main.ipynb --to python

