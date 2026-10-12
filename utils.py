import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def cross_entropy(y,y_hat):
    epsilon = 1e-9
    return - np.mean( ( y * np.log( y_hat + epsilon ) + (1 - y) * np.log( 1 - y_hat + epsilon ) ) )

def accuracy(y,y_hat):
    n = len(y)
    acc = np.sum(y == (y_hat >= 0.5).astype(np.int64))
    return acc / n

def forward(w,x):
    return sigmoid( x @ w )

def classify(w,x):

    n, d = x.shape
    xt = np.concat( [x, np.ones((n,1))] , axis=1 )
    out = forward(w,xt)
    return np.round( out, 0)

def binary_classification(X,y,lr,epochs,batch_size=None):
    n, d = X.shape
    w = np.random.randn(d + 1)
    Xt = np.concat( [X, np.ones((n,1))] , axis=1 )

    if batch_size is None:
        batch_size = n

    for step in range(epochs):
        
        y_hat = forward(w,Xt)

        loss = cross_entropy(y,y_hat)
        acc = accuracy(y,y_hat)

        dw = Xt.T @ (y_hat - y) / batch_size
        
        w -= lr * dw

        if step % 100 == 0:
            print(f'Epochs {step + 1} | Loss {loss} | Accuracy {acc}')
    return w