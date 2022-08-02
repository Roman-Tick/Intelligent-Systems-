import numpy as np
import pandas as pd
import matplotlib as plt
from keras.utils.np_utils import to_categorical
from copy import deepcopy
import time
import math
import random

class NN():
    def __init__(self, hidden_layer_nodes, alpha, batches, epochs, sizes):
    self.hidden_layer_nodes = hidden_layer_nodes
    self.alpha = alpha
    self.batches = batches
    self.epochs = epochs
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #creates matrix of weights
    self.biases = [np.random.randn() for y in sizes[1:]] #creates matrix of baises 
    self.error = []
    self.accuracy_tracker = []
    self.loss_tracker = []
    self.predictions = []

    def feed_forward(self, X, y):
        #calculate all activations 
        Activation_func = x
        Activation_list = [x] #list of all activations
        nets_of_nodes = [] #stores all the nets/in sum of each node
        for b, w in zip(self.biases, self.weights):
            net = np.dot(w,Activation_func) + b
            nets_of_nodes.append(net)
            Activation_func = Sigmoid(net)
            Activation_list.append(Activation_func)

    def back_prop(self, X, Y):
        """
            x is tuple of all inputs/previous activations
        """
        Activation_list, nets_of_nodes = self.feed_forword(x, y)

        delta = self.cost_derivative(Activation_list[-1], y) * Sigmoid_derivative(nets_of_nodes[-1])
        b = [np.zeros(bias.shape) for bias in self.biases] #layed numpy array
        w = [np.zeros(weight.shape) for weight in self.weights] #layed numpy array
        b[-1] = delta
        w[-1] = np.dot(delta, Activation_list[-2].transpose())

        for layer in xrange(2, self.num_of_layers):
            net = nets_of_nodes[-layer]
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * Sigmoid_derivative(net)
            b[-layer] = delta
            w[-layer] = np.dot(delta, Activation_list[-layer-1].transpose())

        return b, w

    def error(self, y_actual, y_predictions):
        error = Cross_Entropy(y_actual, y_predictions)
        return 0.5 * np.mean(error)

    def Stochastic_GD(self, tests, test_data, data, ins_outs, epochs): #ins and outs reps the inputs and corresponging dirsted outputs
        """
            each epoch - partitions trianing data into mini batches (easy way to sample randomly) 
                        then each mini bacth is applied to single step of gradient descent
        """
        batch_size = 10 #place holder
        
        for epoch in range(epochs):
            mini_batch = [data[k:k+batch_size] for k in xrange(0, len(data), batch_size)] #change to another way
            for set in mini_batch:
                self.update(set)
            
            if data: #debugging
                print("epoch not complete")
            else:
                print("epoch complete")

    def activations(self, a):
        """
            returns output of network
        """
        for b, w in zip(self.biases, self.weights):
            a = Sigmoid(np.dot(w, a)+b)
        return a

    def update(self, mini_batch): 
        """
            updates weights and biases using gradient descent on a single mini batch
        """
        temp_weight = [np.zeros(w.shape) for w in self.weights]
        temp_bias = [np.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            div_bias, div_weight = self.Back_Prop(x, y)

    def total_cost(self, data):
        cost = 0.0
        for x, y in data:
            activation = self.activations(x)
            cost += (L_rate*np.linalg.norm(a-y)**2)/len(data)

        cost += L_rate*(lmbda/len(data)) * sum(np.linalg.norm(weight)**2 for weight in self.weights)

        return cost

    def cost_derivative(self, node_out, y):
        return (node_out-y)

    def predict(self, x):
        z = self.weights.dot(x.T)
        act = Sigmoid(z)
        return Argmax(act.T)

    def score(self, predictions, actuals):
        return np.sum(actuals == predictions, axis=0) / float(actuals.shape[0])

    def fit_model(self, X, Y):
        Y_batch = np.array_split(Y, self.batches)
        X_batch = np.array_split(X, self.batches)
        for i in range(self.epochs):
            list_of_errors = []
            predictions = []
            for X, Y in zip(X_batch, Y_batch):
                z2, a2 = self.feed_forward(X)
                self.back_prop(X, Y, a2, z2)
                error = Cross_Entropy(Y, a2.T)
                list_of_errors.append(0.5 * np.mean(error))

                Y_predictions = Argmax(a2.T)
                Y_actual = Argmax(Y)
                predictions.append(score(Y_predictions, Y_actual))

            self.accuracy_tracker.append(np.mean(predictions))
            self.loss_tracker.append(np.mean(list_of_errors))

    def plot_loss(self):
        plt.title("Loss vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(self.loss_tracker, label="Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        plt.title("Accuracy vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(self.accuracy_tracker, label="Accuracy")
        plt.legend()
        plt.show()

def vectorise(y):
    x = np.zeros((10,1))
    x[y] = 1.0
    return x

def Sigmoid(x): #function for calculating sigmoid for a certain input x
    return 1.0/(1.0+np.exp(-x))

def Sigmoid_Prime(x):
    a1 = Sigmoid(x)
    return a1 * (1 - a1)

def Cross_Entropy(y_target, outputs):
    return -np.sum(sp.log(outputs) * y_target, axis=1)

def Argmax(y):
    return np.argmax(y, axis=1)

def softmax(z):
    sum = np.sum(np.exp(z.T), axis=1).reshape(-1,1)
    return (np.exp(z.T) / sum).T

def load_data():
    #load data sets
    training_data = pd.read_csv('fashion-mnist_train.csv')
    test_data = pd.read_csv('fashion-mnist_test.csv')

    #drop class values and add to there own list
    Y_train = training_data["label"].values
    X_train  = training_data.drop(labels = ["label"], axis=1)
    Y_test = training_data["label"].values
    X_test  = training_data.drop(labels = ["label"], axis=1)

    #normalization to reduce effect of illuminations differences
    X_train = X_train/255.0
    X_test = X_test/255.0

    #reshape to 28x28
    X_train = X_train.values.reshape(-1,28,28,1)
    X_test = X_test.values.reshape(-1,28,28,1)

    #encode labels to one hot vectors
    Y_test = to_categorical(Y_test, num_classes = 10)
    Y_train = to_categorical(Y_train, num_classes = 10)

    return X_train, Y_train, X_test, Y_test

#------------------------------

hidden_layer_nodes = 30
epochs = 30
LR = 3
alpha = 10e-4
batches = 30
sizes = [784, 30, 10]

X_train, Y_train, X_test, Y_test = load_data()

fashion_NN = NN(size[1], alpha, batches, epochs, sizes)
fashion_NN.fit_model(deepcopy(X_train), deepcopy(Y_train))
fashion_NN.plot_loss()
fashion_NN.plot_accuracy()

predictions = fashion_NN.predict(X_test)
Accuracy = fashion_NN(predictions, Y_test)
