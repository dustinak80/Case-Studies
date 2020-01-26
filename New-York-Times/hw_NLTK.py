# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:16:01 2020

@author: Dustin
"""

import nltk
from nltk import tokenize
from collections import Counter
#nltk.download()
import pandas as pd
import numpy as np


def main(filename, filename2 = 'No', add = 'No', replace = 'No'):
    # Open a file named philosophers.txt.
    infile = open(filename, 'r')
    
    if filename2 != 'No':
        outfile = open(filename2, 'w')

    # Read the file's contents.
    if add != 'No':
        count = 0
        for line in infile:
            count += 1
            print(count)
            #print(len(line))
            if filename2 != 'No':
                line = line.rstrip('\n') +'.\n'
                outfile.write(line)
    elif replace != 'No':
        count = 0
        for line in infile:
            count += 1
            print(count)
            #print(len(line))
            if filename2 != 'No':
                line = str(replace[line.rstrip('\n')])+'\n'
                outfile.write(line)
    else:
        count = 0
        for line in infile:
            count += 1
            print(count)


    # Close the file.
    infile.close()
    
    if filename2 != 'No':
        outfile.close()

    # Print the data that was read into
    # memory.
    

# Call the main function.
train_name = 'data_train.txt'
# train_name2 = 'data_train_art.txt'
# #main(filename)
# main(train_name, train_name2, add = 'Yes')

# goal_name = "labels_train_original.txt"
goal_name2 = 'labels_train_numbers.txt'
replace = {'News': 0,
           'Opinion': 1,
           'Classifieds': 2,
           'Features': 3}
# #main(filename)
# main(goal_name, goal_name2, replace = replace)

########################################################################
#Get Training Data Set Values

def data_manip(x_file, y_file):
    #bring in the values from the document
    infile = open(x_file, 'r')
    data = []
    for line in infile:
        data += [line]

    infile.close()

    #Tokenize the verbs
    verbs = [   'VB',
            'VBD',
            'VBG',
            'VBN',
            'VBP',
            'VBZ',
            ]

    X = {}
    for i in range(len(data)):
        tokens = tokenize.word_tokenize(data[i])
        tags = nltk.pos_tag(tokens)
        #Collect the Verbs
        counts = dict(Counter(tag for word, tag in tags if tag in verbs))
        for j in verbs:
            if j not in X.keys():
                X[j] = [0]*(i)
            X[j] += [counts[j]/sum(counts.values()) if j in counts.keys() else 0]
        print('{} finished'.format(i))

    X['length'] = []
    X['to the'] = []
        
    for line in data:
        X['length'] += [len(line)] #Length of each article
        if line[:6] == 'to the': #if the article starts with 'to the' then give 1
            X['to the'] += [1]
        else:
            X['to the'] += [0]


    #Get Y Values
    infile = open(y_file, 'r')
    y = []
    for line in infile:
        y += [int(line)]
    infile.close()

    #Get X and Y
    x = pd.DataFrame(X)
    y = pd.Series(y)
    
    return x, y

#Get Training Set X and Y
train_x, train_y = data_manip(train_name, goal_name2)

# #All Data
# All = pd.concat([train_y, train_x], axis = 1)

# for i in verbs:
#     All[[0,i]].hist(by = 0)
    
##########################################################################
#Get Validation/Test Set

#Numbers for Validation y
# goal_name = 'labels_valid_original.txt'
goal_name2 = 'labels_valid_numbers.txt'
# main(goal_name, goal_name2, replace = replace)

test_name = 'data_valid.txt'

#Get test set values
test_x, test_y = data_manip(test_name, goal_name2)


##########################################################################
#kNN 
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import confusion_matrix as cm

score = {'train': [], 'test':[]}

for i in [12, 15, 18, 21, 24]: #12 was the best with all the data
    #set up and fit
    neigh = knc(n_neighbors = i)
    neigh.fit(train_x, train_y)

    #Test the training set
    train_score = neigh.score(train_x, train_y)
    test_score = neigh.score(test_x, test_y)
    
    #Store
    score['train'] += [train_score]
    score['test'] += [test_score]

#Only Verbs
#neighbors = [3, 6, 9, 12, 15]
# test [0.3575, 0.377, 0.3845, 0.3855, 0.393]
# train [0.6475, 0.5535, 0.516, 0.4915, 0.4915]

# neighbors = [12, 15, 18, 21, 24]   
# test [0.3855, 0.393, 0.4, 0.4005, 0.3855, 0.393, 0.4, 0.4005, 0.402]
# train [0.4915, 0.4915, 0.482, 0.4715, 0.4915, 0.4915, 0.482, 0.4715, 0.471]

#Use neighbors = 24
neigh = knc(n_neighbors = 24)
neigh.fit(train_x, train_y)

#Predictions
train_pred = neigh.predict(train_x)
test_pred = neigh.predict(test_x)

#confusion matrix
#Training Set
confusion = cm(train_y, train_pred)
fun = lambda x: x/sum(x)
cm_perc = np.apply_along_axis(fun, 1, confusion)

# 0.444444	0.184524	0.202381	0.168651
# 0.138144	0.717526	0.0762887	0.0680412
# 0.248566	0.170172	0.439771	0.141491
# 0.284836	0.229508	0.19877	0.286885

global_acc = sum(cm_perc.diagonal())/4 #.4721565

#Test Set
confusion2 = cm(test_y, test_pred)
cm_perc2 = np.apply_along_axis(fun, 1, confusion2)

# 0.40625	0.150391	0.255859	0.1875
# 0.145957	0.674556	0.0986193	0.0808679
# 0.280851	0.193617	0.325532	0.2
# 0.32681	0.260274	0.215264	0.197652

global_acc2 = sum(cm_perc2.diagonal())/4 #.4009


##########################################################################
# #Extra Stuff

# #for adding all of the frequency of token tags
# for j in set(list(counts.keys()) + list(X.keys())):
#     if j not in X.keys():
#         X[j] = [0]*(i)
#     X[j] += [counts[j]/sum(counts.values()) if j in counts.keys() else 0]

# #Add ['.' , 'SYM']
# add = {'.' : [0]*2000,
#        'SYM' : [0]* 2000
#        }

# test_x = pd.concat([pd.DataFrame(add), test_x], axis = 1)