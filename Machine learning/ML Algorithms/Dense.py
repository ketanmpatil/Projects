import numpy as np


class Dense():
    def __init__(self,learning_rt,iterations):
        self.lr = learning_rt
        self.iterations = iterations
        self.weights = None
        self.bias = None


    def fit(self,X,y):
        X_ = X.copy()
        Y_ = y.copy()

        n_samples, n_features = X.shape
        
        # Initialize Weights and Bias

        self.weights = np.random.randn(n_features)
        self.bias = 0

        for _ in range(self.iterations):



            # Hypothesis Function

            activation = self.sigmoid(np.dot(X,self.weights) + self.bias)



    def predict(self,X):
        y_predicted = self.sigmoid(np.dot(X,self.weights) + self.bias)
        return y_predicted

    def sigmoid(self,x):
        val = 1 / (1 + np.exp(-x))
        return val