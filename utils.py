import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import pandas as pd
import os

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
        exp = np.exp(x - np.max(x,axis=0))
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
        return np.mean(y == y_hat)

    def onehot(self,y):
        Y = np.zeros((self.q,len(y)))
        Y[ y , np.arange( len(y) ) ] = 1
        return Y

    def classify(self,x):
        y_pred = self.forward(x)
        max = np.argmax( y_pred , axis=0)
        return max

    def initialize(self,w_init):
        self.w = w_init

    def forward(self,x):  
        if x.shape[0] == self.w.shape[-1] - 1:
            x = self.add_bias(x)
        return self.softmax( self.w @ x )

    def fit(self, X, y,return_metrics = False, plot_metrics = True ):

        d , n = X.shape

        if self.w is None:
            self.w = np.zeros((self.q , d + 1))

        Xt = self.add_bias(X)
        Yt = self.onehot(y)

        batch_size = n if self.batch_size is None else self.batch_size

        if return_metrics:
            loss_curve = { }
            acc_curve = { }
            

        for step in range(self.epochs):
    
            indexes = np.random.permutation(n)
    
            for s in range(0,n,batch_size):
    
                batch_idx = indexes[ s : s + batch_size ]
    
                X_batch = Xt[ : , batch_idx ]
                y_batch = Yt[ : , batch_idx ]

                m = len( batch_idx )
    
                y_pred = self.forward( X_batch )
    
                loss = self.cross_entropy( y_batch, y_pred )
    
                #dreg = self.w.copy()
                #dreg[-1] = 0
                dw = ( (y_pred - y_batch) @ X_batch.T ) / m
                
                self.w -= self.lr * dw
    
            if step % 10 == 0:
                y_pred = self.forward(Xt)
                y_classified = self.classify(Xt)
                loss = self.cross_entropy( Yt , y_pred )
                acc = 100 * self.accuracy(y,y_classified)
                if return_metrics:
                    loss_curve[step + 1] = loss
                    acc_curve[step + 1] = acc
                print(f'Epochs {step + 1} | Loss {loss:.4f} | Accuracy {acc:.2f} %')

        if return_metrics:
            loss_df = pd.Series(loss_curve,name='Training loss')
            acc_df = pd.Series(acc_curve,name='Training accuracy')
            return loss_df, acc_df

    def evaluate(self,X,y):

        y_pred = self.forward(X)
        y_pred_classified = self.classify(X)

        if len(y) == y.size:
            y_one_hot = self.onehot(y)

            loss = self.cross_entropy( y_one_hot , y_pred )
            accuracy = 100 * self.accuracy( y , y_pred_classified)
        else:
            y_class = self.classify(y)

            loss = self.cross_entropy( y , y_pred )
            accuracy = 100 * self.accuracy( y_class , y_pred_classified)

        print(f'Evaluation loss: {loss:.4f} | Evaluation accuracy: {accuracy:.2f} %')

        return loss, accuracy