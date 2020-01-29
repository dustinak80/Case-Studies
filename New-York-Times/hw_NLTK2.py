# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:31:26 2020

@author: Dustin

ref: 
https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
##
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
import nltk
from nltk import tokenize
from collections import Counter
from nltk.corpus import stopwords as sw
##
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.ensemble import  RandomForestClassifier as rfc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.naive_bayes import ComplementNB as cnb
##
from sklearn.metrics import confusion_matrix as cm


#######################################################################
#Get Data

#Bring in the X - Values
infile = open('data_train.txt', 'r')
X = {'X' : [x for x in infile]}
infile.close()

#Bring in the Y - Values
infile = open('labels_train_original.txt', 'r')
replace = {'News': 0,
           'Opinion': 1,
           'Classifieds': 2,
           'Features': 3}
Y = {'Y' : [replace[y.rstrip('\n')] for y in infile]}
infile.close()

#Combine them
data = pd.concat([pd.DataFrame(Y), pd.DataFrame(X)], axis = 1)

#######################################################################
#Transform X using a pipline

class length( BaseEstimator, TransformerMixin ):
    """
    This will return the length of the Article
    """
    
    #Class Constructor 
    def __init__( self, get_length = True ):
        self._get_length = get_length
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        if self._get_length:
            X.loc[:,'length'] = X['X'].apply(lambda x: len(x))
        
        print('Done Getting length\n')
        
        return X
    
class starts( BaseEstimator, TransformerMixin ):
    """
    This will create a column with a 1 if starts with 'to the'
    and 0 if it does not. (opinion articles)
    """
    #Class Constructor 
    def __init__( self, to_the = True ):
        self._to_the = to_the
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        if self._to_the:
            X.loc[:,'to the'] = X['X'].apply(lambda x: 1 if x[:6] == 'to the' else 0)

        print('Done Getting "to the"\n')

        return X

class NLTK ( BaseEstimator, TransformerMixin ):
    """
    
    """
    #Class Constructor 
    def __init__( self, stopwords = None, keys = None ):
        self._stopwords  = stopwords or set(sw.words('english'))
        self._keys = keys
        
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that converts tagged dictionary to frequency of nouns and verbs
    def transform( self, X, y = None ):
        
        def tokenize_and_tag(X):
            
            #Get the tokens - tokenize, drop stop words, drop repeat words (mostly to reduce time)
            tokens = tokenize.word_tokenize(X)
            key_words = [words for words in tokens if words not in self._stopwords]
            key_words = list(set([words for words in key_words if words.isalpha()]))
            
            for words in key_words:
                tags = nltk.pos_tag(words)
                counts = dict(Counter(tag for word, tag in tags if tag.startswith('N') or tag.startswith('V')))
            
            return counts
        
        print('Start Tokenizing and Tagging')
        counts = X['X'].apply(tokenize_and_tag)
        print('Done Tokenizing and Tagging\n')
        
        print('keys: {}'.format(self._keys))
        #for training set, self._keys == None, after, the keys will be
        #set so that the test set has same column names
        if self._keys == None:
            keys = []
            for i in range(len(X)-1):
                interest = list(counts[i].keys())
                keys += [x for x in interest if x not in keys]
            self._keys = keys
            print('Got Keys:\n{}\n'.format(self._keys)) 
        
        nvs = {}
        for i in self._keys:
            nvs[i] = []
        
        #Getting the frequencies of each of the Columns in Data
        for i in counts.index:
            #sum of verbs (startswith('V')) and nouns (startswith('N'))
            vs = sum([counts[i][k] for k in counts[i].keys() if k.startswith('V')])
            ns = sum([counts[i][k] for k in counts[i].keys() if k.startswith('N')])
            for j in nvs.keys():
                #go through column names and if that article has that tag, get frequency
                if j in counts[i].keys():
                    t = counts[i][j]
                    nvs[j] += [t / vs if j.startswith('V') else t / ns]
                #if not, set frequency = 0
                else:
                    nvs[j] += [0]        
        print('Got frequencies')
        X = pd.concat([X,pd.DataFrame(nvs)], axis = 1)

        return X
       
#pipeline - create and transform
pipeline = Pipeline(steps = [('length', length()),
                             ('starts', starts()),
                             ('NLTK', NLTK())])

data = pipeline.transform(data) # Had to set data = pipeline because
                                # not editing dataframe in NLTK transform

#Write training set data to a file for future import
data.to_csv('data_train_unique_tag.csv', index = False)

#######################################################################
scores = { 'test' : {},
          'train' : {}
    }

#Split test and train of training set
X_train, X_test, y_train, y_test = tts(data.loc[:,'length':], data['Y'], stratify = data['Y'])
X_train = pd.DataFrame(X_train).reset_index(drop = True)
X_test = pd.DataFrame(X_test).reset_index(drop = True)
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

#kNN (Base) - fit, get scores, store
neigh = knc()
neigh.fit(X_train, y_train)

n_train_score = neigh.score(X_train, y_train)
n_test_score = neigh.score(X_test, y_test)

scores['test']['knn'] = n_test_score
scores['train']['knn'] = n_train_score

#Random Forrest (Base) - fit, get scores, store
rfc = rfc()
rfc.fit(X_train, y_train)

rfc_train_score = rfc.score(X_train, y_train)
rfc_test_score = rfc.score(X_test, y_test)

scores['test']['rfc'] = rfc_test_score
scores['train']['rfc'] = rfc_train_score

#Gaussian Naive Bayes (Base) - fit, get scores, store
nbg = gnb()
nbg.fit(X_train, y_train)

nbg_train_score = nbg.score(X_train, y_train)
nbg_test_score = nbg.score(X_test, y_test)

scores['test']['nbg'] = nbg_test_score
scores['train']['nbg'] = nbg_train_score

#Multinomial Naive Bayes (Base) - fit, get scores, store
nbc = cnb()
nbc.fit(X_train, y_train)

nbc_train_score = nbc.score(X_train, y_train)
nbc_test_score = nbc.score(X_test, y_test)

scores['test']['nbc'] = nbc_test_score
scores['train']['nbc'] = nbc_train_score

# print(pd.DataFrame(scores))
#       test     train
# knn  0.346  0.582667
# nbc  0.346  0.373333
# nbg  0.446  0.460000
# rfc  0.384  0.950000
#######################################################################
#training set
X_train_final = data.loc[:,'length':]
y_train_final = data['Y']

#fit the desired Algorithm
nbg = gnb()
nbg.fit(X_train_final, y_train_final)

#Import Validation Data
infile = open('data_valid.txt', 'r')
X_test_final = pd.DataFrame({'X' : [x for x in infile]})
infile.close()

infile = open('labels_valid_original.txt', 'r')
Y_test_final = pd.DataFrame({'Y' : [replace[y.rstrip('\n')] for y in infile]})
infile.close()

data_test = pd.concat([Y_test_final, X_test_final], axis = 1)

#Save Transformed Validation Data
data_test = pipeline.transform(data_test)
data_test.to_csv('data_valid_unique_tag.csv', index = False)

X_test_final = data_test.loc[:, 'length':]
Y_test_final = data_test['Y']

#Predict
Y_test_predict = nbg.predict(X_test_final)

#Confusion Matrix - Standard Confusion, Percent Confusion
confusion = cm(Y_test_final, Y_test_predict)
fun = lambda x: x/sum(x)
cm_perc = np.apply_along_axis(fun, 1, confusion)
# print(cm_perc)
# [[0.         0.0625     0.93164062 0.00585938]
#  [0.         0.78500986 0.21499014 0.        ]
#  [0.         0.0106383  0.98510638 0.00425532]
#  [0.         0.16438356 0.82778865 0.00782779]]
global_acc = sum(cm_perc.diagonal())/4
# print(global_acc)
# 0.4444860083903422

#######################################################################
#Extra
rfc = rfc()
rfc.fit(X_train_final, y_train_final)

Y_test_predict = rfc.predict(X_test_final)

confusion = cm(Y_test_final, Y_test_predict)
cm_perc = np.apply_along_axis(fun, 1, confusion)
# print(cm_perc)
# [[0.359375   0.09765625 0.29882812 0.24414062]
#  [0.12426036 0.65285996 0.05325444 0.16962525]
#  [0.31489362 0.04680851 0.41489362 0.22340426]
#  [0.27984344 0.17221135 0.29941292 0.24853229]]
global_acc = sum(cm_perc.diagonal())/4
# print(global_acc)
# 0.4189152168004312