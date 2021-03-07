#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading the Data

# In[2]:


d=pd.read_csv('train.csv',sep=',',encoding='latin')


# In[3]:


d.head()


# In[4]:


d.drop(columns=['keyword','location'],inplace=True)


# In[5]:


d.head()


# In[6]:


d.shape


# In[7]:


import keras
from keras.layers import Dense,Dropout


# In[8]:


x=d['text']
y=d['target']


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation


# # Cleaning the data

# In[12]:


def Tokenizer(string):
    words=nltk.word_tokenize(string)
    return ' '.join(words)

def Removestopwords(string):
    for i in punctuation:
        string=string.replace(i,'')
    eng_stop=stopwords.words('english')
    words=nltk.word_tokenize(string)
    k=[]
    for each in words:
        if each not in eng_stop:
            k.append(each.lower())
    return ' '.join(k)

def Lammetization(string):
    words=nltk.word_tokenize(string)
    ws=WordNetLemmatizer()
    l=[]
    for each in words:
        l.append(ws.lemmatize(each))
    return ' '.join(l)
            


# In[13]:


def Refine(string):
    return Lammetization(Removestopwords(Tokenizer(string)))


# In[14]:


d['Processed']=d['text'].apply(lambda x: Refine(x))


# In[15]:


d.head()


# In[16]:


x=d['Processed']
y=d['target']


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# In[18]:


x_train


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# In[20]:


cv=CountVectorizer(max_features=1000,ngram_range=(1,1),max_df=0.3,min_df=2)


# In[21]:


x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)


# In[22]:


y_train


# In[23]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[26]:


model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(1000,)))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# # Model Training

# In[27]:


model.fit(x_train,y_train,epochs=20,batch_size=32,validation_data=(x_test,y_test))


# In[28]:


results=pd.DataFrame(model.history.history)


# # Final Result

# In[32]:


results.plot(legend=True,figsize=(12,8))

