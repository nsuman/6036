import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import Decimal
def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta, tempParameter):
   
    #YOUR CODE HERE
    n, d = X.shape
    k , d1 = theta.shape
    assert(d == d1)
    return_list =[]
    for feature_vector in X:
        max = 0
        append_matrix = []
        for theta_i in theta:
            dot_product = np.dot(feature_vector,theta_i)/tempParameter
            if dot_product > max: max = dot_product
            append_matrix.append(dot_product)
        append_matrix = np.array(append_matrix)
        append_matrix = append_matrix-max
        append_matrix = np.exp(append_matrix)
        append_matrix = append_matrix/sum(append_matrix)
        return_list.append(append_matrix)
    return np.transpose(np.array(return_list));


def computeCostFunction(X, Y, theta, lambdaFactor, tempParameter):
     #YOUR CODE HERE
    n, d = X.shape
    n2, = Y.shape
    #print(n,n2)
    assert(n== n2)
    k , d1 = theta.shape
    
    prob_matrix = computeProbabilities(X,theta, tempParameter)
    
    outer_sum = 0
    for i in range(n):
        label  =  Y[i]
        inner_sum = 0
        for j in range(k):
            if j ==label:
                prob_j_i = Decimal(prob_matrix[j][i])
                inner_sum+= prob_j_i.ln()
        
        outer_sum+= inner_sum
        
    theta_square = 0
    for vector in theta:
        for component in vector:
            theta_square += component**2
        
    return float(-outer_sum/n) + lambdaFactor*theta_square/2
    

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter):
    #YOUR CODE HERE
    
    n, d = X.shape
    n2, = Y.shape
    #print(n, n2)
    assert(n== n2)
    k , d1 = theta.shape
    
    prob_matrix = computeProbabilities(X, theta, tempParameter)
    
    descent_matrix = []
    
    for j in range(k):
        theta_j = theta[j]
        inner_sum = np.zeros(d1)
        for i in range(n):
            vector = X[i]
            if j==Y[i]:
                inner_sum += (1-prob_matrix[j][i])*vector
            else:
                inner_sum += (0-prob_matrix[j][i])*vector
        descent_matrix.append((-1/(tempParameter*n))*inner_sum + lambdaFactor *theta_j)
    
    descent_array = np.array(descent_matrix)
    
    return theta- alpha*descent_array


def updateY(trainY, testY):
    #YOUR CODE HERE
    return trainY%3, testY%3

def computeTestErrorMod3(X, Y, theta, tempParameter):
    #YOUR CODE HERE
    classification = getClassification(X, theta, tempParameter)%3
    return 1- np.mean(classification == Y)
    

def softmaxRegression(X, Y, tempParameter, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor, tempParameter))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter)
    return theta, costFunctionProgression
    
def getClassification(X, theta, tempParameter):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta, tempParameter)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def computeTestError(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels == Y)
