import math
import operator
import csv
import random
import numpy  as np
import pandas as pd
from copy import deepcopy

#recursive function
#at each stage divide training set into two list based on catorgory
#find which catagory the test node belongs to
#repeat function and send test node, 1/2 catagory(which belongs with test node), catagory +1 

def get_entropy(column):
    uniques = get_unique_values(column)
    values = number_of_items(uniques, column)
    for v in values:
        v[0] = v[0]/len(column)

    entropy = 0.000
    for v in values:
        if v[0] > 0:
            entropy = -1*(v[0]*math.log2(v[0])) + entropy
    return entropy

def get_unique_values(set):
    uniques = []
    for item in set:
        if item not in uniques:
            uniques.append(item)
    return uniques

def number_of_items(uniques, set): #np.bincount()
    occurrence = []
    for x in uniques:
        occurrence.append([x, 0])
    for i in set:
        for j in range(len(occurrence)):
            if i == occurrence[j][0]:
                occurrence[j][1] = occurrence[j][1] + 1
    for x in occurrence:
        del x[0]
    return occurrence

def highest_index(list):
    highest = 0
    HI = 0
    for x in range(len(list)):
        if list[x] > highest:
            HI = x
    return HI

def split_list(data_set, split_index):
    left_split = []
    right_split = []
    for item in data_set:
        if item[split_index] == 0:
            left_split.append(item)
        else:
            right_split.append(item)
    return left_split, right_split


def info_gain(data, split_column, target_column):
    orginal_entropy = get_entropy(target_column)
    left_split = []
    right_split = []

    #see if i can remove this because im spliting then putting back together
    subset = []
    subset.append(target_column)
    subset.append(split_column)
    left_split, right_split = split_list(list(zip(*subset)), 1)
    split_entropy = 0
    
    for set in [left_split, right_split]:
        if len(set) == 0:
            continue
        prob = 0.0 #(set.shape[0]/data.shape[0])
        #print('set: ', set)
        for row in split_column:
            if row == set[0][1]:
                prob = prob  + 1
        prob = prob/len(split_column)
        turn_list = list(zip(*set))
        split_entropy = prob*get_entropy(turn_list[0]) #set[0]

    return orginal_entropy - split_entropy

def highest_info_gain(data_set):
    turn_list = list(zip(*data_set))
    list_of_IG = []
    index = 1
    while index < len(data_set[0]):
        list_of_IG.append(info_gain(data_set, turn_list[index], turn_list[0]))
        index = index + 1
    return list_of_IG.index(max(list_of_IG)) + 1

def remove_column(index, list):
    new_list = []
    for row in list:
        #del item[index] #item.pop(index)
        new_row = []
        for i in range(len(row)):
            if i != index:
                new_row.append(row[i])
        new_list.append(new_row)
    if len(list) == 1:
        return new_list[0]
    return new_list

def make_prediction(results): #make pred
    turn_list = list(zip(*results))
    prediction_list = turn_list[0]

    class_list = [['republican', 0.00], ['democrat', 0.00]]
    for x in prediction_list:
        if x == class_list[0][0]:
            class_list[0][1] = class_list[0][1] + 1
        else:
            class_list[1][1] = class_list[1][1] + 1
    
    occ = []
    turn_list = list(zip(*class_list))
    turn_list = turn_list[1]
    for x in turn_list:
        if x != 0:
            occ.append(x/len(prediction_list))
        else:
            occ.append(x)
    return occ

def decision_tree(training_set, test_point):
    #print(training_set)
    #print(' --- ')
    turn_list = list(zip(*training_set))
    entropy = get_entropy(turn_list[0])

    '''return turn_list[0] #may have to 
    elif len(training_set[0]) == 1: #elif cant do any more splits
        return training_set

    '''
    
    if entropy == 0 or len(training_set[0]) == 1: #if 100% in list (entropy = 0?) - stop condition
        return training_set
    else:
        #check info gain for each cat
        #split training list
        #add used attribute to a used list
        #check which list test point belongs to
        #call decision tree() again with new list as training list
        split_index = highest_info_gain(training_set)
        left_split = []
        right_split = []
        left_split, right_split = split_list(training_set, split_index)

        if test_point[split_index] == 0:
            new_training_set = remove_column(split_index, left_split)
        else:
            new_training_set = remove_column(split_index, right_split)

        new_test_point = remove_column(split_index, [test_point])
        result = decision_tree(new_training_set, new_test_point)
        return result

def split_data(split):
    training_set = []
    test_set = []

    dataset = pd.read_csv('votes.csv')
    for x in range(len(dataset)-1):
        #for y in range(4):
            #dataset[x][y] = float(dataset[x][y])
        if random.random() < split:
            training_set.append(dataset.iloc[x,:])
        else:
            test_set.append(dataset.iloc[x,:])

    return training_set, test_set



training_set = []
test_set = []
prediction_list = []
split = 0.67

training_set, test_set = split_data(split)

for i in range(len(test_set)):
    result = decision_tree(training_set, test_set[i])
    prediction = make_prediction(result)
    print('>Prediction: R = ', prediction[0], ' D = ', prediction[1], ', Actual: ', test_set[i][0])
    #add to pred list
    #print data

#dataset = pd.read_csv('votes.csv')
#print(len(training_set[0]))

#test = [['republican'], ['republican'], ['democrat']]
#test2 = [['republican'], ['republican'], ['republican']]
#print(make_prediction(test2))

#print(test_set[0])
#new_test = remove_column(0, [test_set[0]])
#print(new_test)

