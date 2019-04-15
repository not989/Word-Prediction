#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import array
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import pickle
import sys
import heapq


# In[2]:


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)


# In[3]:


# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
    in_text, result = seed_text, seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = array(encoded)
        # predict a word in the vocabulary
        yhat = model.predict_classes(encoded, verbose=0)
        
        #This is necessary( customized part)
        preds = model.predict(encoded, verbose=0)[0]
        
        
        next_indices = sample(preds, 5)
#         print(next_indices)
        ar = []
        for indd in next_indices:
            # map predicted word index to word
            yhat = indd
            out_word = ''
            for word, index in tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            # append to input
            #print(out_word)
            ar.append(out_word)
            #in_text, result = out_word, result + ' ' + out_word
        return ar
 


# In[4]:


loaded_model = pickle.load(open("best_model_pkl_format.pkl", 'rb'))
tokenizer_1 = pickle.load(open("tokenizer.pkl", 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)


# evaluate
res = generate_seq(loaded_model, tokenizer_1, 'আমরা', 1)
print(res)

