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



# data = """ Jack and Jill went up the hill\n
# 		To fetch a pail of water\n
# 		Jack fell down and broke his crown\n
# 		And Jill came tumbling after\n """

# data = """ আমার নাম আবুল \n
# 		আবুলের জন্ম হইসে অনেক আগে \n
# 		এখন কেমন আছিস ?\n
# 		আবুল এখন কি করবো?\n """

data = open('test_corpus_bn.txt', 'r', encoding='utf-8').read()


# In[7]:


# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]

# # saving
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # loading
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

filename = "tokenizer.pkl"  
with open(filename, 'wb') as file:  
    pickle.dump(tokenizer, file)


# In[8]:


# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# In[13]:


# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i-1:i+1]
    sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))


# In[14]:


# split into X and y elements
sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]


# In[15]:


# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)


# In[16]:


# define model
model = Sequential()
model.add(Embedding(vocab_size, 2000, input_length=1))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))
# print(model.summary())


# In[17]:


# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[19]:


# fit network
model.fit(X, y, epochs=10, verbose=2)

# from keras.callbacks import ModelCheckpoint
# filepath = "best_model_pkl_format.pkl"
# checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# history = model.fit(X, y, validation_split=0.60, epochs=15, batch_size=15, callbacks=[checkpointer])
# print(history)


# In[20]:


pkl_filename = "best_model_pkl_format.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)


# In[21]:


loaded_model = pickle.load(open("best_model_pkl_format.pkl", 'rb'))
tokenizer_1 = pickle.load(open("tokenizer.pkl", 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)


# evaluate
res = generate_seq(loaded_model, tokenizer_1, 'আমরা', 1)
print(res)

