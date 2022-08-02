import random
import numpy  as np
import time
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import math
import operator
import csv
from numpy.linalg import norm

class K_Nearest_Neighbours():
    def __init__(self, input_k):
        self.k = input_k

    def train(self, X, Y): #fit data
        self.X_train = X
        self.Y_train = Y

    def predict(self, X_test):
        distances = self.get_distance(X_test)
        return self.predict_labels(distances)

    def get_distance(self, X_test):
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]
        distances = np.zeros((n_test, n_train))

        for i in range(n_test):
            for j in range(n_train):
                distances[i,j] = np.sqrt(np.sum((X_test[i,:] - self.X_train[j,:])**2))

        return distances
    
    def predict_labels(self, distances):
        n_test = distances.shape[0]
        y_pred = np.zeros(n_test)

        for i in range(n_test):
            y_inices = np.argsort(distances[i,:])
            K_closest_classes = self.Y_train[y_inices[:self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(K_closest_classes))

        return y_pred

#---------------------------------------------------------------------------------------------------------------------------
    
def euclidean_distance(test_point, Training_Set_point, length):
    distance = 0
    test_point_vec = np.array([test_point[0], test_point[1], test_point[2], test_point[3]])
    training_point_vec = np.array([Training_Set_point[0], Training_Set_point[1], Training_Set_point[2], Training_Set_point[3]])
    distance = norm(test_point_vec - training_point_vec)
    return distance #np.sqrt(distance)

def KNN(Training_Set, Test_Set, k):
    distances = []
    K_Neighbours = []
    length = len(Test_Set)-1
    for i in range(len(Training_Set)):
        distance_between_points = euclidean_distance(Test_Set, Training_Set[i], length)
        distances.append((Training_Set[i], distance_between_points))
    #sort distances
    distances.sort(key=operator.itemgetter(1))
    for i in range(k):
        K_Neighbours.append(distances[i][0])
    return K_Neighbours

def get_response(neighbours):#out of k nearst it picks the most popular and returns predicted result
    votes = []
    in_list = False
    for point in neighbours:
        new = [point[4], 1]
        if len(votes) == 0:
            votes.append(new)
            continue
        for x in range(len(votes)):
            if point[4] == votes[x][0]:
                votes[x][1] = votes[x][1] + 1
                in_list = True
        if in_list == False:
            votes.append(new)
        in_list = False
    #sort votes
    highest = votes[0]
    for x in votes:
        if x[1] > highest[1]:
            highest = x

    #print("votes", votes)
    #print(neighbours)
    return highest[0]

def accuracy(test_set, pred):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == pred[i]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0

def split_data(split):
    training_set = []
    test_set = []
    #with open('iris.csv', 'rb') as csvfile:
        #line = csv.reader(csvfile)
        #for row in csvfile:
            #print(row)
            #continue
        #dataset = list(line)

    dataset = pd.read_csv('iris.csv')
    for x in range(len(dataset)-1):
        #for y in range(4):
            #dataset[x][y] = float(dataset[x][y])
        if random.random() < split:
            training_set.append(dataset.iloc[x,:])
        else:
            test_set.append(dataset.iloc[x,:])

    return training_set, test_set

def correct(pred, actual):
    if pred == actual:
        return True
    else:
        return False

#dataset = pd.read_csv('iris.csv')
#print(dataset.head())

training_set = []
test_set = []
predictions = []
split = 0.67
k = 3

training_set, test_set = split_data(split)

print('--')
for i in range(len(test_set)):
    neighbours = KNN(training_set, test_set[i], k)
    result = get_response(neighbours)
    predictions.append(result)
    print('Sample Class =', test_set[i][-1] + ', Prediction Class =', result + ', Prediction Correct:', correct(result, test_set[i][-1]))
percentage = accuracy(test_set, predictions)
print('--\n' + 'Training Set Accuracy: ' + repr(percentage) + '%')
