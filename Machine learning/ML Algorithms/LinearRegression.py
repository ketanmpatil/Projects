#!/usr/bin/env python
# coding: utf-8

# In[167]:


import numpy as np

class LinearReg():
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
            
            # Derivating weight and biases
#             dw = (1 / n) * np.dot(X.transpose(),np.sum(y_predicted-y))
#             db = (1 / n) * np.sum(y_predicted-y)

            dw = (1 / n) * np.dot(X.T, (y_predicted - y))
            db = (1 / n) * np.sum(y_predicted - y)


            # Updating weights and biases
            self.weights -= self.learning_r * dw
            self.bias -= self.learning_r * db
            
    def predict(self,X_test):
        predictions = np.dot(self.weights,X_test.transpose())+self.bias
        return predictions
    
    

            
        
    
    
    


# In[ ]:




