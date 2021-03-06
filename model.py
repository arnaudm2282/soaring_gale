'''
This filel contains the model architectures we explore in this project. The
final model architecture we settled on is the Forecaster_fc_hidden class. Below
that are some other architectures we tried (kept here becuase we reference them
in the README).
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
#from google.colab import drive

# %% Final Model Architecture

class Forecaster_fc_hidden(nn.Module):
    def __init__(self, input_features, encoder_hidden_features, fc_hidden,
                 output_length, encoder_layers=1):
        super(Forecaster_fc_hidden, self).__init__()
        
        self.encoder = nn.RNN(input_size=input_features,
                              hidden_size=encoder_hidden_features,
                              num_layers=encoder_layers,
                              batch_first=True)
        
        self.fc_hidden = nn.Linear(encoder_hidden_features, fc_hidden)
        self.fc_activation = nn.ReLU()
        self.fc_out = nn.Linear(fc_hidden, output_length * input_features)
        
        self.output_length = output_length
        self.input_features = input_features
        
    def forward(self, x):
        N = x.shape[0]
        encoder_o, encoder_h_n = self.encoder(x)
        
        fc_input = torch.flatten(encoder_h_n.permute(1,0,2), start_dim=1,
                                 end_dim=-1)
        out = self.fc_hidden(fc_input)
        out = self.fc_activation(out)
        out = self.fc_out(out)
        
        out = out.reshape(N, self.output_length, self.input_features)
        return out
    

# %% Other mdoel architectures that were tried


class RNN(nn.Module):
    def __init__(self,input_size,hidden,output_size, processing_func=None):
        super(RNN,self).__init__()
        self.func = processing_func
        self.hidden = hidden
        self.rnn = nn.RNN(input_size,hidden,batch_first=True)
        self.fc = nn.Linear(hidden,output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:,:5,:])
        return out


class LSTM(nn.Module):
    def __init__(self,input_size,output_size,hidden,layers):
        super(LSTM,self).__init__()
        self.hidden = hidden
        self.layers=layers
        self.norm = nn.BatchNorm1d(30)
        self.lstm = nn.LSTM(input_size,hidden,layers,batch_first=True)
        
        self.fc = nn.Linear(hidden,output_size)

    def forward(self,x):
        h0 = torch.zeros(self.layers, x.size(0), self.hidden).requires_grad_()
        c0 = torch.zeros(self.layers, x.size(0), self.hidden).requires_grad_()
        
        out, _ = self.lstm(x,(h0.detach(),c0.detach()))
        out = self.norm(out)
        out=self.fc(out[:,:5,:])
        return out


class Forecaster_fc(nn.Module):
    def __init__(self, input_features, encoder_hidden_features, 
                  forecaster_hidden_features, output_size,encoder_layers=1, 
                  forecaster_layers=1):
        super(Forecaster, self).__init__()
        
        self.encoder = nn.RNN(input_size=input_features,
                              hidden_size=encoder_hidden_features,
                              num_layers=encoder_layers,
                              batch_first=True)
        
        self.forecaster = nn.RNN(input_size=encoder_hidden_features,
                                  hidden_size=forecaster_hidden_features,
                                  num_layers=forecaster_layers,
                                  batch_first=True)
        
        self.fc = nn.Linear(forecaster_hidden_features,output_size)
        
    def forward(self, x):
        encoder_o, encoder_h_n = self.encoder(x)
        forecaster_o, forecaster_h_n = self.forecaster(encoder_o)
        out = self.fc(forecaster_o[:,:5,:])
        return out


class Forecaster(nn.Module):
    def __init__(self, input_features, encoder_hidden_features, 
                  forecaster_hidden_features, output_length,encoder_layers=1, 
                  forecaster_layers=1):
        super(Forecaster, self).__init__()
        
        self.encoder = nn.RNN(input_size=input_features,
                              hidden_size=encoder_hidden_features,
                              num_layers=encoder_layers,
                              batch_first=True)
        
        self.forecaster = nn.RNN(input_size=encoder_hidden_features,
                                  hidden_size=forecaster_hidden_features,
                                  num_layers=forecaster_layers,
                                  batch_first=True)
        
        self.output_length=output_length
        
    def forward(self, x):
        encoder_o, encoder_h_n = self.encoder(x)
        forecaster_o, forecaster_h_n = self.forecaster(encoder_o)
        out = forecaster_o[:,:self.output_length,:]
        return out


# %% Testing code if running this file alone

if __name__ == '__main__':
    if False:
        data = []
        for i in range(100):
            x = torch.randn(30,4)
            t = torch.randn(5,4)
            
            data.append((x,t))
            
        dl = torch.utils.data.DataLoader(data, batch_size=20, shuffle=False)
        for x, t in iter(dl):
            x = x
            t = t
            break
        
        print('data shape', x.shape, t.shape)
        
        mod = Forecaster(input_features=4, encoder_hidden_features=10,
                  forecaster_hidden_features=4, output_size=4)
        #o, h_n = mod.forward(x)
        o = mod.forward(x)
        print('output shape', o.shape)
        
    if False:
        x = torch.randn(20,30,4)
        m = Forecaster_fc_hidden(4,10,50,5)
        
        y = m.forward(x)
        print(x.shape)
        print(y.shape)
