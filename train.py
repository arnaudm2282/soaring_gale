import torch
import sklearn
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import csv
import datetime
import matplotlib.pyplot as plt
import data_process as datap
import model

def get_accuracy(model, test):
    correct, total = 0, 0
    for input, output in test:
        out = model(input)
        correct+=np.sum(np.abs(out.detach().numpy() -output.detach().numpy()))
        total+=output.shape[0]
    return correct/total


def train_RNN(model, train, valid, num_epochs=5, learning_rate=1e-5):
    # TODO
    final=np.empty((0,))
    #playing with different loss functions
    #criterion = nn.MSELoss(reduction='mean')
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):
        for input, output in train:
            #input = input.permute(0,2,1)
            pred = model(input)
            loss = criterion(pred,output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #to plot final epoch of predictions
            if epoch==num_epochs-1:
                final = np.append(final,pred.detach().numpy()[:,1])
        
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train))
        valid_acc.append(get_accuracy(model, valid))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
                epoch+1, loss, train_acc[-1], valid_acc[-1]))
    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, 'b-',label="Train")
    
    plt.xlabel("Epoch")
    plt.ylabel("T Accuracy", color='b')
    
    plt.legend(loc='best')
    plt.show()

    plt.plot(epochs, valid_acc, 'y-',label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("V Accuracy", color='y')
    plt.legend(loc='best')
    plt.show()

    plt.plot(range(int(len(final)/4)),final[1::4],label='Prediction')
    plt.xlabel("Day")
    plt.ylabel("Prediction", color='y')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    
    data = []

    _, data = datap.load_price_data_into_numpy_array('aadr.us.txt', 
                                       '../Data/ETFs')

    data = datap.remove_volume_open_interest(data)  
    x_t_pairs = datap.make_x_t_tuple_tensor_pairs_in_place(data, 30, 5)  
    train_loader = DataLoader(x_t_pairs[:1000],batch_size=30,shuffle=False,drop_last=True)
    valid_loader = DataLoader(x_t_pairs[1000:],batch_size=30,shuffle=False,drop_last=True)

    LSTMModel = model.LSTM(4,4,50,3)

    train_RNN(LSTMModel,train_loader,valid_loader,num_epochs=100,learning_rate=0.001)