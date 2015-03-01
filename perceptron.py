from numpy import *
from random import *
###############################################################################
## Choose here the columns of the training data that will be used by Perceptron
types_clf = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 
             'Fare', 'Embarked'];


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
    
    #Num columns and rows
    num_feats = len(learning_data[0])
    num_inputs = len(learning_data) 
    print("features = %i inputs = %i" % (num_feats, num_inputs))

    #The eta value from the algorithm: how much to rotate on each iteration
    eta = 0.0025
    
    #The output initialization (all zeros)
    w = zeros (num_feats)
    
    
    for i in range (num_inputs):
        survived = heaviside(train_data[i]['Survived'])
        
        #attempts to predict the output
        y = 0
        for j in range(num_feats):
            y = y + w[j]*learning_data[i][j]
        y = heaviside(y)
        
        #rotate the hyperplane (if prediction was wrong)
        for j in range(num_feats):
            
            #skip the ID column
            if j == 0:
                continue
            
            w[j] = w[j] + eta*(survived - y)*learning_data[i][j]
            print("row = %i survived: %i y = %i w[%i] = %f" % (i, survived, y, j, w[j]))
            
    return w

def calc_perceptron(w, data_row):
    ans = 0
    print("len (data_row) = %i" % (len(data_row)))
    for i in range (len(data_row) - 1):
        print("w[%i] = %f, data_row[%i] = %i" % (i, w[i], i, data_row[i]))
        ans = ans + w[i]*data_row[i]
        
    return heaviside (ans);
        

def perceptron (train_data, test_data):
    w = train_perceptron (train_data)
    
    #subset of useful data
    prediction_data = matrix(test_data[types_clf]).T.getA1();
    
    num_inputs = len(prediction_data)
    ans = zeros (num_inputs)
    for i in range(num_inputs):
        ans[i] = calc_perceptron(w, prediction_data[i])
    
    return ans
