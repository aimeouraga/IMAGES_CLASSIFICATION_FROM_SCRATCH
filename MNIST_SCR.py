#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from random import shuffle
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

device = torch.device('mps')
import time
from jupyterthemes import jtplot
jtplot.style(theme='gruvboxd', context='notebook', ticks=True, grid=False)
from sklearn.preprocessing import OneHotEncoder

############################## START DOWNLOAD AND PROCESS THE DATA  ################################
transform = transforms.Compose([transforms.ToTensor(),])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, )

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


def retrive_data(loader):
    for images, labels in loader:
        images_pixels = images
        labels = labels
    
    images_flattened = images_pixels.reshape(images.shape[0], -1)
    
    data = pd.DataFrame(images_flattened)
    data['label'] = pd.Series(labels)

    return data


train_data = retrive_data(train_loader)
test_data = retrive_data(test_loader)


X_train = torch.tensor(train_data.drop(columns=['label']).values, dtype=torch.float32).to(device)
Y_train = torch.tensor(train_data['label'].values, dtype=torch.long).reshape(-1,1)
encod = OneHotEncoder()
Y_train_EN = torch.tensor(encod.fit_transform(Y_train).todense(), dtype=torch.float32).to(device)

X_test = torch.tensor(test_data.drop(columns=['label']).values, dtype=torch.float32).to(device)
Y_test = torch.tensor(test_data['label'].values, dtype=torch.long).reshape(-1,1)
Y_test_EN = torch.tensor(encod.fit_transform(Y_test).todense(), dtype=torch.float32).to(device)


X_train = X_train.T
Y_train_EN = Y_train_EN.T
X_test = X_test.T
Y_test_EN = Y_test_EN.T

######################################## END ############################################


class MNIST_SCRATH:

    def __init__(self, nbr_unit, nbr_epochs, learning_rate, nbr_classes):
        
        self.nb_unit_H1 = nbr_unit
        self.nb_unit_H2 = nbr_unit
        self.nb_output = nbr_classes
        self.nbr_epochs = nbr_epochs
        self.learning_rate = learning_rate
        self.train_loss = []
        self.test_loss = []


    def relu(self, z):
        return torch.maximum(torch.tensor(0).to(device), z)

    def d_relu(self, z):
        return torch.where(z <= 0, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))


    def softmax(self, z):
        expo = torch.exp(z)
        return expo / torch.sum(expo, dim=0)

    def loss(self, y_true, y_pred):
        return -((torch.tensor(1).to(device))/y_true.shape[1]) * torch.sum(y_true*torch.log(y_pred))

 

    def initialization(self, X):
        self.nb_input = X.shape[0]
        W1 = torch.normal(0, np.sqrt(2/(self.nb_output + self.nb_input)), size=(self.nb_unit_H1, self.nb_input)).to(device)
        W2 = torch.normal(0, np.sqrt(2/(self.nb_output + self.nb_input)), size=(self.nb_unit_H2, self.nb_unit_H1)).to(device)
        W3 = torch.normal(0, np.sqrt(2/(self.nb_output + self.nb_input)), size=(self.nb_output, self.nb_unit_H2)).to(device)
        b1 = torch.normal(0, np.sqrt(2/(self.nb_output + self.nb_input)), size=(self.nb_unit_H1, 1)).to(device)
        b2 = torch.normal(0, np.sqrt(2/(self.nb_output + self.nb_input)), size=(self.nb_unit_H2, 1)).to(device)
        b3 = torch.normal(0, np.sqrt(2/(self.nb_output + self.nb_input)), size=(self.nb_output, 1)).to(device)
        
        return W1, W2, W3, b1, b2, b3

    def forward_pass(self, X, W1, W2, W3, b1, b2, b3):
        Z1 = W1 @ X + b1
        A1 = self.relu(Z1)
        Z2 = W2 @ A1 + b2
        A2 = self.relu(Z2)
        Z3 = W3 @ A2 + b3
        A3 = self.softmax(Z3)
    
        return Z1, A1, Z2, A2, Z3, A3


    def backward_pass(self, X, Y, A1, A2, A3, W2, W3, b1, b2, b3, Z1, Z2, Z3):
        m = X.shape[1]
        
        dW3 = (1/m) * (A3-Y) @ A2.T
        
        dW2 = (1/m) * (W3.T @ (A3-Y)) * self.d_relu(Z2) @ A1.T
        
        dW1 = (1/m) * W2 @ W3.T @ (A3-Y) * self.d_relu(Z2)  * self.d_relu(Z1) @ X.T
        
        db1 = (1/m) * torch.sum(W2 @ W3.T @ (A3-Y) * self.d_relu(Z2)  * self.d_relu(Z1), dim=1, keepdims=True)
        
        db2 = (1/m) * torch.sum((W3.T @ (A3-Y)) * self.d_relu(Z2), dim=1, keepdims=True)
        
        db3 = (1/m) * torch.sum(A3-Y, axis=1, keepdims=True)
        
        return dW1, dW2, dW3, db1, db2, db3


    def update(self, X, Y, A1, A2, A3, Z1, Z2, Z3, W1, W2, W3, b1, b2, b3):
        
        dW1, dW2, dW3, db1, db2, db3 = self.backward_pass(X, Y, A1, A2, A3, W2, W3, b1, b2, b3, Z1, Z2, Z3)
        
        W1 = W1 - self.learning_rate * dW1
        W2 = W2 - self.learning_rate * dW2
        W3 = W3 - self.learning_rate * dW3
        b1 = b1 - self.learning_rate * db1
        b2 = b2 - self.learning_rate * db2
        b3 = b3 - self.learning_rate * db3
        
        return W1, W2, W3, b1, b2, b3


    def predict(self, X, W1, W2, W3, b1, b2, b3):
        _, _, _, _, _, A3 = self.forward_pass(X, W1, W2, W3, b1, b2, b3)
        return A3

    def function_accuracy(self, Y_true, Y_pred):
        max_indices = torch.argmax(Y_pred, dim=0)
        pred = torch.zeros_like(Y_pred)
        pred[max_indices, torch.arange(Y_pred.size(1))] = 1
    
        Y_true = Y_true.cpu().numpy()
        pred = pred.cpu().numpy()
       
        correct_predictions = np.sum([1 for i in range(Y_true.shape[1]) if  np.argmax(Y_true[:,i]) == np.argmax(pred[:,i])])
        
        accuracy = (correct_predictions / Y_true.shape[1])*100
        return accuracy

    
    def fit(self, X_train, y_train, X_test, y_test):

        W1, W2, W3, b1, b2, b3 = self.initialization(X_train)

        for epoch in range(self.nbr_epochs):
            #forward pass or make a prediction
            Z1, A1, Z2, A2, Z3, A3 = self.forward_pass(X_train, W1, W2, W3, b1, b2, b3)
            
            #backward pass
            dW1, dW2, dW3, db1, db2, db3 = self.backward_pass(X_train, y_train, A1, A2, A3, W2, W3, b1, b2, b3, Z1, Z2, Z3)
            
            #update the parameters
            W1, W2, W3, b1, b2, b3 = self.update(X_train, y_train, A1, A2, A3, Z1, Z2, Z3, W1, W2, W3, b1, b2, b3)
            
            #train loss
            tr_loss = self.loss(y_train, A3).cpu().numpy()
            self.train_loss.append(tr_loss)
            
            #test loss
            _, _, _, _, _, y_tpred =  self.forward_pass(X_test, W1, W2, W3, b1, b2, b3)
            tes_loss = self.loss(y_test, y_tpred).cpu().numpy()
            self.test_loss.append(tes_loss)
        
            #train accuracy
            y_TRpred = self.predict(X_train, W1, W2, W3, b1, b2, b3)
            tr_acc = self.function_accuracy(y_train, y_TRpred)
        
            #test accuracy
            y_TEpred = self.predict(X_test, W1, W2, W3, b1, b2, b3)
            te_acc = self.function_accuracy(y_test, y_TEpred)
            
            #display the losses
            if epoch%100==0:
                print(f"epoch {epoch}/{self.nbr_epochs}: train_loss {tr_loss:.4f} ==== train_accuracy {tr_acc:.2f}%  ==== test_loss {tes_loss:.4f} ==== val_accuracy {te_acc:.2f}% ")
            
        
        
        #plot train and test losses
        plt.plot(self.train_loss, label = 'train_loss')
        plt.plot(self.test_loss, label = "test_loss")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.show()
        
########################## MODEL INSTANTIATION ############################
nbr_unit = 1600
nbr_epochs = 1601
learning_rate = 0.1
nbr_classes = 10
model = MNIST_SCRATH(nbr_unit, nbr_epochs, learning_rate, nbr_classes)
model.fit(X_train, Y_train_EN, X_test, Y_test_EN)

