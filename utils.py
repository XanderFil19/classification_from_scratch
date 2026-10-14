import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import pandas as pd

class Classifier():

    def __init__(self , lr , epochs , num_class , lam = 0 , batch_size = None , w = None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = w
        self.lam = lam
        self.q = num_class

    @staticmethod
    def softmax(x):
        exp = np.exp(x)
        return exp / np.sum(exp, axis = 0 , keepdims=True)
    
    @staticmethod
    def cross_entropy(p,q,eps=1e-9):
        q = np.clip(q, eps, 1 - eps)
        s = - np.sum( p * np.log(q) , axis=0 )
        return np.mean( s )

    @staticmethod
    def add_bias(X):
        n = X.shape[1]
        return np.concat([X, np.ones((1,n))], axis=0)
    
    @staticmethod
    def accuracy(y, y_hat):
        max = np.argmax( y_hat , axis=0)
        return np.mean(y == max)

    def onehot(self,y):
        Y = np.zeros((self.q,len(y)))
        Y[ y , np.arange( len(y) ) ] = 1
        return Y

    def initialize(self,w_init):
        self.w = w_init

    def forward(self,x):        
        return self.softmax( self.w @ x )

    def fit(self, X, y):

        d , n = X.shape

        if self.w is None:
            self.w = np.zeros((self.q , d + 1))

        Xt = self.add_bias(X)
        Yt = self.onehot(y)

        if self.batch_size is None:
            self.batch_size = n
    
        batch_size = self.batch_size
    
        for step in range(self.epochs):
    
            indexes = np.random.permutation(n)
    
            for s in range(0,n,batch_size):
    
                batch_idx = indexes[ s : s + batch_size ]
    
                X_batch = Xt[ : , batch_idx ]
                y_batch = Yt[ : , batch_idx ]

                m = len( batch_idx )
    
                y_pred = self.forward( X_batch )
    
                loss = self.cross_entropy( y_batch, y_pred )
                #acc = self.accuracy(y_batch,y_pred)
    
                #dreg = self.w.copy()
                #dreg[-1] = 0
                dw = ( (y_pred - y_batch) @ X_batch.T ) / m
                
                self.w -= self.lr * dw
    
            if step % 100 == 0:
                y_pred = self.forward(Xt)
                loss = self.cross_entropy( Yt , y_pred )
                acc = 100 * self.accuracy( y , y_pred )
                print(f'Epochs {step + 1} | Loss {loss:.4f} | Accuracy {acc:.2f} %')