#! /bin/env python
# -*- coding: utf-8 -*-
import importlib

import pandas as pd
import numpy as np

from gensim.models.word2vec import Word2Vec
from keras_preprocessing import sequence
import keras.utils
from keras import utils as np_utils
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers.core import Embedding
from keras.layers.rnn import LSTM
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

import yaml
import sys
import multiprocessing

sys.setrecursionlimit(1000000)
# importlib.reload(sys)
# sys.setdefaultencoding('utf8')

np.random.seed()

#参数配置
cpu_count = multiprocessing.cpu_count()  # 4CPU数量
voc_dim = 150 #word的向量维度
min_out = 10 #单词出现次数
window_size = 7 #WordVec中的滑动窗口大小

lstm_input = 150#lstm输入维度
epoch_time = 10#epoch
batch_size = 16 #batch


def loadfile():
    #文件输入
    neg = []
    pos = []
    with open('../data/pos.txt', 'r') as f:
        for line in f.readlines():
            pos.append(line)
        f.close()
    with open('../data/neg.txt', 'r') as f:
        for line in f.readlines():
            neg.append(line)
        f.close()
        X_Vec = np.concatenate((pos, neg))
    y = np.concatenate((np.ones(len(pos), dtype=int),
                        np.zeros(len(neg), dtype=int)))
    # print X_Vec,y
    return X_Vec, y

def onecut(doc):
    # 将中文分成一个一个的字
    ret = list(doc.strip())
    '''
    for i in range(len(ret)):
        print ret[i]
        if (i>=2):
            break;
    '''
    return ret


def one_seq(text):
    text1=[]
    for document in text:
        # if len(document)<3:
        #     continue
        text1.append(onecut(document.replace('\n', '')) )
    return text1

def word2vec_train(X_Vec):
    model_word = Word2Vec(vector_size=voc_dim,
                     min_count=min_out,
                     window=window_size,
                     workers=cpu_count,
                     epochs=5)
    model_word.build_vocab(X_Vec)
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.epochs)
    model_word.save('../model/Word2Vec.pkl')

    #print model_word.wv.vocab.keys()[54],model_word.wv.vocab.keys()
    #print len(model_word.wv.vocab.keys())
    #print model_word ['有']
    input_dim = len(model_word.wv.index_to_key) + 1 #下标0空出来给不够10的字
    embedding_weights = np.zeros((input_dim, voc_dim)) 
    w2dic={}
    for i in range(len(model_word.wv.index_to_key)):
        embedding_weights[i+1, :] = model_word.wv[model_word.wv.index_to_key[i]]
        w2dic[model_word.wv.index_to_key[i]]=i+1
    #print embedding_weights
    return input_dim,embedding_weights,w2dic

def data2inx(w2indx,X_Vec):
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    return data 


def train_lstm(input_dim, embedding_weights, x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,
                        input_dim=input_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=lstm_input)) 
    model.add(LSTM(128, activation='softsign'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',#hinge
                  optimizer='adam', metrics=['mae', 'acc'])

    print("Train...")  # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time, verbose=1)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)

    yaml_string = model.to_json()
    with open('../model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('../model/lstm.h5')
    print('Test score:', score)


X_Vec, y = loadfile()
X_Vec = one_seq(X_Vec)
input_dim,embedding_weights,w2dic = word2vec_train(X_Vec)

index = data2inx(w2dic,X_Vec)
index2 = sequence.pad_sequences(index, maxlen=voc_dim )

x_train, x_test, y_train, y_test = train_test_split(index2, y, test_size=0.2)
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)
train_lstm(input_dim, embedding_weights, x_train, y_train, x_test, y_test)
