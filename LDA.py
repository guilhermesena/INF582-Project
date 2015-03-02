import numpy as np
import scipy as sp

###############################################################################
## Choose here the columns of the training data that will be used by LDA
types_clf = ['Pclass', 'Sex', 'Age'];


def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    
    classLabels = np.unique(Y)
    classNum = len(classLabels)
    dim = len(X[0])
    totalMean = np.mean(X,0)

    # partition class labels per label - list of arrays per label
    partition = [np.where(Y==label)[0] for label in classLabels] 

    # find mean value per class (per attribute) - list of arrays per label
    classMean = [(np.mean(X[idx],0),len(idx)) for idx in partition]

    # Compute the within-class scatter matrix
    Sw = np.zeros((dim,dim))
    for idx in partition:
        Sw += np.cov(X[idx],rowvar=0) * len(idx)# covariance matrix * fraction of instances per class 

    # Compute the between-class scatter matrix
    Sb = np.zeros((dim,dim))
    for mu,class_size in classMean:
        mu=mu.reshape(dim,1) #make column vector
        Sb += class_size * np.dot((mu - totalMean), np.transpose((mu - totalMean)))    

    # Solve the eigenvalue problem for discriminant directions to maximize class seperability while simultaneously minimizing
    # the variance within each class
    # The exception code can be ignored for the example dataset
    try:
        S = np.dot(linalg.inv(Sw), Sb)
        eigval, eigvec = linalg.eig(S)
    except: #Singular Matrix
        print("Singular matrix")
        eigval, eigvec = linalg.eig(Sb, Sw+Sb)

    idx = eigval.argsort()[::-1] # Sort eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
    W = np.real(eigvec[:,:classNum-1]) # eigenvectors correspond to k-1 largest eigenvalues


    # Project data onto the new LDA space
    X_lda = np.real(np.dot(X, np.real(W)))

    # project the mean vectors of each class onto the LDA space
    projected_centroid = [np.dot(mu, np.real(W)) for mu,class_size in classMean]

    return W, projected_centroid, X_lda

def predict(X, projected_centroid, W):
    """Apply the trained LDA classifier on the test data 
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """

    # Project test data onto the LDA space defined by W 
    projected_data  = np.dot(X, W)
    
    # Compute distances from centroid vectors
    dist = [linalg.norm(data-centroid) for data in projected_data for centroid in projected_centroid]
    Y_raw = np.reshape(np.array(dist), (len(X),len(projected_centroid)))
    
    # Assign the label of the closed centroid vector
    label = Y_raw.argmin(axis=1)

    return label

def LDA (training_data, test_data):
        learning_data = np.matrix(training_data[types_clf]).T.getA1()
        W, projected_centroid, X_lda = my_LDA(learning_data, training_data['Survived'])
        
        learning_test = np.matrix(test_data[types_clf]).T.getA1()
        return predict(learning_test, projected_centroid, W)