import numpy as np  
import pandas as pd 
import re           
from bs4 import BeautifulSoup 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
 
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from keras.models import Model,Sequential
from keras.callbacks import EarlyStopping
import warnings
from sklearn.model_selection import train_test_split
import random
from keras.models import load_model

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

ls=[0,1]

data1=pd.read_excel('data1.xlsx')
data1=data1[['text','label']]

data3=pd.read_excel('data3.xlsx')
data3=data3[['text','label']]
data2=pd.read_excel('data2 .xlsx')
data2=data2[['text','label']]
data=pd.concat([data1,data2])
data=pd.concat([data,data3])
data= data[data.text.apply(lambda x: x !="")]
data= data[data.label.apply(lambda x: x !="" and x!=10)]
data=data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
data['text']=data['text'].astype(str)
data['label']=data['label'].astype(int)

print(data.head())
from sklearn.utils import resample
df_majority = data[data.label==0]
df_minority = data[data.label==1]
print(data.label.value_counts())
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=3427,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
data = pd.concat([df_majority, df_minority_upsampled])
#print(data.text[0])
x_tr,x_val,y_tr,y_val=train_test_split(data['text'],data['label'],test_size=0.4,random_state=0,shuffle=True)

print(data['label'].value_counts())

import pickle
#prepare a tokenizer for text on training data
with open('tokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)

x_val =  x_tokenizer.texts_to_sequences(x_val)   
x_val = pad_sequences(x_val, maxlen=50)

model=load_model('model_text.md')

score=model.evaluate(x_val,y_val)
print("Accuracy : %.2f%%" %(score[1]*100))


