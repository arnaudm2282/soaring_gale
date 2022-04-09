'''
This file contains the main functions used for optimizaing the model on 
training data, checkpointing models, and charting model forecasts for 
qualitative analysis.
'''

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


def checkpoint(model, path, checkpoint_name):
    save_path = '{}/{}.pt'.format(path, checkpoint_name)
    torch.save(model.state_dict(), save_path)
    return


def average_model_error(model, data):
    data_loader = DataLoader(data, batch_size=512)
    model.eval()
    
    error = 0
    N = len(data)
    for x, t in iter(data_loader):
        out = model.forward(x)
        error += torch.sum((t.detach() - out.detach()) ** 2)
    return error / N


def train_model(model, train, valid, num_epochs=5, learning_rate=1e-5, 
              batch_size=256, criteria='mse', verbose=True,
              checkpoint_path=None, checkpoint_name=None):
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
    losses, train_error, valid_error = [], [], []
    epochs = []
    
    # optimize model
    for epoch in range(num_epochs):
        for x, t in iter(train_loader):
            model.train()
            
            # model output
            pred = model.forward(x)
            
            # training step
            loss = criterion(pred,t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # add performance to tracking variables
        losses.append(float(loss))
        epochs.append(epoch)
        train_error.append(average_model_error(model, train))
        valid_error.append(average_model_error(model, valid))
        if verbose:
            print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
                    epoch+1, loss, train_error[-1], valid_error[-1]))
            
        if checkpoint_path and checkpoint_name:
            checkpoint(model, checkpoint_path, checkpoint_name + f'epoch_{epoch}')

    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_error, 'b-',label="Train")
    
    plt.title('Train Error')
    plt.xlabel("Epoch")
    plt.ylabel("Train Error", color='b')
    
    plt.legend(loc='best')
    plt.show()

    plt.title('Valid Error')
    plt.plot(epochs, valid_error, 'y-',label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Valid Error", color='y')
    plt.legend(loc='best')
    plt.show()
    
# %% misc functions

def plot_ohlc_timeseries(data, title='timeseries'):
    '''
    data - array shape (N,4)
        - N timesteps of 4 features (Open, High, Low, Close)
    '''
    plt.title(title)
    N_steps = torch.arange(data.shape[0]) + 1
    O = data[:,0]
    H = data[:,1]
    L = data[:,2]
    C = data[:,3]
    
    plt.plot(N_steps,O,label='Open')
    plt.plot(N_steps,H,label='High')
    plt.plot(N_steps,L,label='Low')
    plt.plot(N_steps,C,label='Close')
    
    plt.xlabel('timestep')
    plt.ylabel('price')
    plt.legend(loc='best')
    plt.show()
    

def plot_model_forecast(model, data, title='model forecast', 
                        context_length=30):
    '''
    Plot models forecast for data based on step through slicing data into 
    inputs of context length

    Parameters
    ----------
    model : torch model
    data : numpy array float32
        - shape (N,M)
    title : string
        Title for pyplot
    context_length : int
        input length to feed model from data
    '''
    model.eval()
    final=np.empty((0,))
    
    N = data.shape[0]
    
    for i in range(N - context_length):
        x = data[i:i+context_length]
        x = x[None,:,:]
        x = torch.from_numpy(x)
        pred = model.forward(x)
        
        final = np.append(final,pred.detach().numpy()[:,1])
        
    plt.title('Forecast')
    plt.plot(range(int(len(final)/4)),final[1::4],label='Prediction')
    plt.xlabel("Day")
    plt.ylabel("Prediction", color='y')
    plt.legend(loc='best')
    plt.show()


# %% Testing code if running this file alone

if __name__ == '__main__':
    if False:
        data = []
    
        _, data = datap.load_price_data_into_numpy_array('aadr.us.txt', 
                                           './data/ETFs')
    
        data = datap.remove_volume_open_interest(data)  
        x_t_pairs = datap.make_x_t_tuple_tensor_pairs_in_place(data, 30, 5)  
        train_data = x_t_pairs[:1000]
        valid_data = x_t_pairs[1000:]
        
        # LSTM model test
        LSTMModel = model.LSTM(4,4,50,3)
        train_model(LSTMModel,train_data,valid_data,num_epochs=10,
                  learning_rate=0.001)
    
    if False:
        data = []
        _, data = datap.load_price_data_into_numpy_array('aadr.us.txt', 
                                           './data/ETFs')
        data = datap.remove_volume_open_interest(data)  
        x_t_pairs = datap.make_x_t_tuple_tensor_pairs_in_place(data, 30, 5)  
        train_data = x_t_pairs[:1000]
        valid_data = x_t_pairs[1000:]
        
        # Forecaster model test
        mod = model.Forecaster(input_features=4,encoder_hidden_features=10,
                               forecaster_hidden_features=4,output_length=5)
        train_model(mod,train_data,valid_data,num_epochs=25,
                    learning_rate=0.001)
        
        plot_model_forecast(mod, data)
        
    # test checkpointing
    if False:
        mod = nn.Linear(10,10)
        
        x = torch.randn(10,10)
        t = torch.randn(10,10)
        
        train_model(mod, [(x,t)], [(x,t)], 
                    checkpoint_path='/home/am/Desktop/temp',
                    checkpoint_name='testtest', num_epochs=10, batch_size=1)
