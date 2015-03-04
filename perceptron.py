import numpy as np
from numpy import *
from random import *
###############################################################################
## Choose here the columns of the training data that will be used by Perceptron
types_clf = ['Pclass', 'Sex', 'Age'];


###############################################################################

#train_perceptron: returns a vector (w) for the dot product to predict test data
def train_perceptron (train_data):
    
    
    #subset of useful data
    learning_data = to_ndarray(train_data, types_clf);
    
    #Num columns and rows
    num_feats = len(learning_data[0])
    num_inputs = len(learning_data) 

    #The eta value from the algorithm: how much to rotate on each iteration
    eta = 0.0025
    
    #The output initialization (all zeros)
    w = zeros (num_feats)
    errors = 0
    
    for nrep in range(1000):
        #calculates the error rate (to find theta)
        errors = 0

        for i in range (num_inputs):
            expected = train_data[i]['Survived']
            
            #attempts to predict the output
            predicted = np.dot(w, learning_data[i]);
                
            predicted = 1 if predicted > 0 else -1 
            if predicted != expected:
                errors += 1
            
            #rotate the hyperplane (if prediction was wrong)
            w += eta * (expected - predicted) * learning_data[i];
       
    rate = 100*errors/num_inputs
    return w

# Converts tupple to ndarray
def to_ndarray(tupple, tupple_type):
    i0 = np.shape(tupple[tupple_type])[0];
    j0 = np.shape(tupple_type)[0];
    mat = np.ndarray((i0,j0));

    for i in range(i0):
        for j in range(j0):
            mat[i][j] = tupple[tupple_type][i][j];

    return mat;


#Perceptron algorithm: Calls training set and returns the array of 0s and 1s predicting test dataset
def perceptron (train_data, test_data):
    
    #first, trains the vector w
    w = train_perceptron (train_data)

    #gets subset of useful data
    prediction_data = to_ndarray(test_data, types_clf);
    
    #runs the calculation for each test element
    num_inputs = len(prediction_data)
    ans = zeros (num_inputs)
    for i in range(num_inputs):
        ans[i] = 1 if np.dot(w, prediction_data[i]) > 0 else 0;

    return ans
