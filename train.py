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

def get_accuracy(model, data):
    data_loader = DataLoader(data, batch_size=512)
    model.eval()
    
    correct, total = 0, 0
    for x, t in iter(data_loader):
        out = model.forward(x)
        correct+=np.sum(np.abs(out.detach().numpy() - t.detach().numpy()))
        total+=t.shape[0]
    return correct/total


def train_model(model, train, valid, num_epochs=5, learning_rate=1e-5, 
              batch_size=256, criteria='mse', verbose=True):
    final=np.empty((0,))
    '''
    Optimize model on train data.
    
    Arguments:
        train, valid : list of (x,t) tuple
            x - inputs
            t - targets
            
        num_epochs : int
            number of epochs to perform
            
        batch_size : int
            number of samples per batch
            
        criteria : string
            specify the loss function.
            'mse' => MSE
            SmoothL1 otherwise
            
        verbose : bool
            if True, prints model performance at the end of each epoch
    '''
    # loss function
    criterion = None
    if criteria == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    else:
        criterion = nn.SmoothL1Loss()
        
    # data loader
    train_loader = DataLoader(train, batch_size=batch_size,shuffle=True,
                              drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    # variables to track model performance
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    
    # optimize model
    for epoch in range(num_epochs):
        for x, t in iter(train_loader):
            model.train()
            # for input, output in train:
            
            # model output
            pred = model.forward(x)
            
            # training step
            loss = criterion(pred,t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # to plot final epoch of predictions
            if epoch==num_epochs-1:
                final = np.append(final,pred.detach().numpy()[:,1])
        
        # add performance to tracking variables
        losses.append(float(loss))
        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train))
        valid_acc.append(get_accuracy(model, valid))
        if verbose:
            print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
                    epoch+1, loss, train_acc[-1], valid_acc[-1]))
        

    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, 'b-',label="Train")
    
    plt.title('Train loss')
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy", color='b')
    
    plt.legend(loc='best')
    plt.show()

    plt.title('Valid Loss')
    plt.plot(epochs, valid_acc, 'y-',label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Valid Accuracy", color='y')
    plt.legend(loc='best')
    plt.show()

    # TODO seperate function for one timeseries of price data.
    # data above can be from many ETFs during training.
    plt.title('Forecast TODO')
    plt.plot(range(int(len(final)/4)),final[1::4],label='Prediction')
    plt.xlabel("Day")
    plt.ylabel("Prediction", color='y')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    if True:
        data = []
    
        _, data = datap.load_price_data_into_numpy_array('aadr.us.txt', 
                                           './data/ETFs')
    
        data = datap.remove_volume_open_interest(data)  
        x_t_pairs = datap.make_x_t_tuple_tensor_pairs_in_place(data, 30, 5)  
        train_data = x_t_pairs[:1000]
        valid_data = x_t_pairs[1000:]
    
        # LSTMModel = model.LSTM(4,4,50,3)
        # train_model(LSTMModel,train_loader,valid_loader,num_epochs=10,
        #           learning_rate=0.001)
    
        mod = model.Forecaster(input_features=4,encoder_hidden_features=10,
                               forecaster_hidden_features=4,output_length=5)
        train_model(mod,train_data,valid_data,num_epochs=1000,
                    learning_rate=0.001)
    
        