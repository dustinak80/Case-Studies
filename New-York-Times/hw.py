# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:16:01 2020

@author: Dustin
"""

import nltk
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
train_name2 = 'data_train_art.txt'
#main(filename)
main(train_name, train_name2, add = 'Yes')

goal_name = "labels_train_original.txt"
goal_name2 = 'labels_train_numbers.txt'
replace = {'News': 0,
           'Opinion': 1,
           'Classifieds': 2,
           'Features': 3}
#main(filename)
main(goal_name, goal_name2, replace = replace)

########################################################################

#Get X Values
infile = open(train_name, 'r')
data = {'length':[],
        'to the': []
        }
for line in infile:
    data['length'] += [len(line)]
    if line[:6] == 'to the':
        data['to the'] += [1]
    else:
        data['to the'] += [0]

infile.close()

#Get Y Values
infile = open(goal_name2, 'r')
y = []
for line in infile:
    y += [line]
infile.close()

#Get X and Y
train_x = pd.DataFrame(data)
train_y = pd.Series(y)

##########################################################################
#kNN 
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import confusion_matrix as cm

#set up and fit
neigh = knc()
neigh.fit(train_x, train_y)

#Test the training set
train_pred = neigh.predict(train_x)

#confusion matrix
confusion = cm(train_y, train_pred)
fun = lambda x: x/sum(x)
cm_perc = np.apply_along_axis(fun, 1, confusion)

# array([[0.60714286, 0.15079365, 0.15873016, 0.08333333],
#        [0.09896907, 0.78762887, 0.06597938, 0.04742268],
#        [0.21988528, 0.12810707, 0.56978967, 0.08221797],
#        [0.23360656, 0.17213115, 0.23155738, 0.36270492]])

global_acc = sum(cm_perc.diagonal())/4

##########################################################################
#kNN for the articles that dont start with 'to the'

train_pred2 = []
for i in train_x.index:
    if train_x.loc[i, 'to the'] == 1:
        train_pred2 += ['1']
    else:
        train_pred2 += [str(int(neigh.predict(np.array(train_x.loc[i,:]).reshape(1,-1))))]
train_pred2 = np.array(train_pred2)

confusion2 = cm(train_y, train_pred2)
confusion2 = confusion2[[1,3,5,1], :]
confusion2 = confusion2[:, [0,2,4,6]]
cm_perc2 = np.apply_along_axis(fun, 1, confusion2)

# 0.583333	0.186508	0.152778	0.077381
# 0.0618557	0.876289	0.0350515	0.0268041
# 0.219885	0.128107	0.56979	    0.082218
# 0.583333	0.186508	0.152778	0.077381

global_acc2 = sum(cm_perc2.diagonal())/4