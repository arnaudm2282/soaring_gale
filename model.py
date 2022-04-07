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
import math
#from google.colab import drive


class RNN(nn.Module):
    def __init__(self,input_size,hidden,output_size, processing_func=None ):
        # TODO
        super(RNN,self).__init__()
        self.func = processing_func
        self.hidden = hidden
        self.rnn = nn.RNN(input_size,hidden,batch_first=True)
        self.fc = nn.Linear(hidden,output_size)
        
    def forward(self, x):
        # TODO
        
        #x = self.func(x)
        #x=x
        #x = torch.reshape(x,(1,30,6))
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
        
        # TODO
        self.fc = nn.Linear(forecaster_hidden_features,output_size)
        
    def forward(self, x):
        encoder_o, encoder_h_n = self.encoder(x)
        forecaster_o, forecaster_h_n = self.forecaster(encoder_o)
        #print('x shape', x.shape)
        #print('e shape', encoder_o.shape, encoder_h_n.shape)
        #print('f shape', forecaster_o.shape, forecaster_h_n.shape)
        out = self.fc(forecaster_o[:,:5,:]) # TODO
        return out#, forecaster_h_n

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, dropout,max_len=1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pose = torch.zeros(max_len,1,d_model)
        pos = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0)/ d_model))
        pose[:, 0,0::2] = torch.sin(pos * div)
        pose[:,0,1::2] = torch.cos(pos*div)
        #pose = pose.unsqueeze(0)
        self.register_buffer('pose',pose)

    def forward(self, x):
        t = self.pose[:, :x.size(1)]
        x = x + t
        return x

class TransformerModel(nn.Module):

    def __init__(self,ntoken,d_model,nhead,d_hid,nlayers,dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid,dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src:torch.Tensor, src_mask: torch.Tensor):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        #src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output[:,:5,:]

def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

if __name__ == '__main__':
    
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


class Forecaster_fc_hidden(nn.Module):
    def __init__(self, input_features, encoder_hidden_features, fc_hidden,
                 output_length, encoder_layers=1):
        super(Forecaster_fc_hidden, self).__init__()
        
        self.encoder = nn.RNN(input_size=input_features,
                              hidden_size=encoder_hidden_features,
                              num_layers=encoder_layers,
                              batch_first=True)
        
        self.fc_hidden = nn.Linear(encoder_hidden_features, fc_hidden)
        
        self.fc_out = nn.Linear(fc_hidden, output_length * input_features)
        
        self.output_length = output_length
        self.input_features = input_features
        
    def forward(self, x):
        N = x.shape[0]
        encoder_o, encoder_h_n = self.encoder(x)
        
        fc_input = torch.flatten(encoder_h_n.permute(1,0,2), start_dim=1,
                                 end_dim=-1)
        out = self.fc_hidden(fc_input)
        out = self.fc_out(out)
        
        out = out.reshape(N, self.output_length, self.input_features)
        return out
    

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
