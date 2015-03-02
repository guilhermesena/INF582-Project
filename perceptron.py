from numpy import *
from random import *
###############################################################################
## Choose here the columns of the training data that will be used by Perceptron
types_clf = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'];


###############################################################################

#Step function
def heaviside(x):
    if x > 0:
        return 1
    return 0

#train_perceptron: returns a vector (w) for the dot product to predict test data
def train_perceptron (train_data):
    
    
    #subset of useful data
    learning_data = matrix(train_data[types_clf]).T.getA1();
    print(learning_data)
    
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
            expected = heaviside(train_data[i]['Survived'])
            
            #attempts to predict the output
            predicted = 0
            for j in range(num_feats):
                predicted += w[j]*learning_data[i][j]
                
            predicted = heaviside(predicted)
            #print("predicted = %i expected = %i" % (predicted, expected))
            if predicted != expected:
                errors += 1
            
            #rotate the hyperplane (if prediction was wrong)
            for j in range(num_feats):
                w[j] += eta*(expected - predicted)*learning_data[i][j]
        
       
    rate = 100*errors/num_inputs
    print("error rate = %f" % rate)
    return w

#predicts output with vector w from training set
def calc_perceptron(w, data_row):
    ans = 0
    
    #basically we get the sign of the linear combination
    for i in range (len(data_row)):
        ans = ans + w[i]*data_row[i]
    
    return heaviside (ans);
        

#Perceptron algorithm: Calls training set and returns the array of 0s and 1s predicting test dataset
def perceptron (train_data, test_data):
    
    #first, trains the vector w
    w = train_perceptron (train_data)
    
    #gets subset of useful data
    prediction_data = matrix(test_data[types_clf]).T.getA1();
    
    #runs the calculation for each test element
    num_inputs = len(prediction_data)
    ans = zeros (num_inputs)
    for i in range(num_inputs):
        ans[i] = calc_perceptron(w, prediction_data[i])
    
    return ans
