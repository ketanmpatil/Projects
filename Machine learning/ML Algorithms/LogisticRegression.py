import numpy as np
import math

class LogisticReg():
    def __init__(self,learning_r=1e-3,iterations=1000):
        self.learning_r=learning_r
        self.iter=iterations
        self.weights=None
        self.bias=None
       
    def fit(self,X,y):      
        n,m=X.shape
        
        # Define the Hypothesis(h(x)), h(x)=weights*features.transpose()+bias
        self.weights=np.ones(m) # Weights, 1xm 
        self.bias = 0 # bias, 1x1

        # Gradient Descent Alogrithm
        for _ in range(self.iter):
            
            # Hypothesis Function
            y_predicted= np.dot(X, self.weights) + self.bias
            h = 1/(1+ math.exp(y_predicted))
            
            # Derivating weight and biases
#             dw = (1 / n) * np.dot(X.transpose(),np.sum(y_predicted-y))
#             db = (1 / n) * np.sum(y_predicted-y)
            cost = (-1/m) * np.sum(y*np.log(h)+(1-y)*np.log(1-h))

            dw = (1/m)*np.dot(X.T,np.dot(h-y))
            db = (1/m)*(h-y)


            # Updating weights and biases
            self.weights -= self.learning_r * dw
            self.bias -= self.learning_r * db
            
    def predict(self,X_test):
        predictions = np.dot(self.weights,X_test.transpose())+self.bias
        return predictions
    
    


