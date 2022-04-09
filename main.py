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
import model as model
import train as train

# augment data full dataset ***
if False:
    etfs_path = './data/ETFs'
    etf_files = os.listdir(etfs_path)
    
    def only_close(data):
        return data[:,3].reshape(-1,1)
    
    train_start_date, train_end_date = '2010-01-01', '2013-01-01'
    val_start_date, val_end_date = train_end_date, '2015-01-01'
    test_start_date, test_end_date = val_end_date, '2030-01-01'
    
    train_data, val_data, test_data = \
      datap.date_make_train_val_test_data(etfs_path,
                                          train_start_date=train_start_date,
                                          train_end_date=train_end_date,
                                          val_start_date=val_start_date,
                                          val_end_date=val_end_date,
                                          test_start_date=test_start_date,
                                          test_end_date=test_end_date,
                                          process_data_func=only_close)
    
    print('len(train):', len(train_data))
    print('len(val):', len(val_data))
    print('len(test):', len(test_data))
    
    print('x,t shape', train_data[0][0].shape, train_data[0][1].shape)
    
    #augment data
    datap.augment(train_data, augment_func=datap.translate_price)
    
    print('len(train):', len(train_data))
    print('len(val):', len(val_data))
    print('len(test):', len(test_data))
    
    print('x,t shape', train_data[0][0].shape, train_data[0][1].shape)
    
    # model
    m = model.Forecaster_fc_hidden(input_features=1,
                                    encoder_hidden_features=100,
                                    fc_hidden=100,
                                    output_length=5)
    
    # train
    train.train_model(m, train_data, val_data, num_epochs=50, 
                      batch_size=10000,
                      learning_rate=1e-5)
    
# load torch model, analysis code
if False:
    m = model.Forecaster_fc_hidden(input_features=1,
                                    encoder_hidden_features=100,
                                    fc_hidden=100,
                                    output_length=5)
    
    model_dict = './checkpoints/forecast_fc_50_relu/model.Forecaster_fc_hidden(input_features=1,encoder_hidden_features=100,fc_hidden=100,output_length=5)train.train_model(m, train_data, val_data, num_epochs=50,batch_size=10000,learning_rate=1e-5).pt'
    m.load_state_dict(torch.load(model_dict))
    
    def find_best_test_point(model, test_data):
        '''
        Return the best performing test point (lowest MSE) of model on
        test_data.
        '''
        min_mse = float('inf')
        min_example_index = None
        N = len(test_data)
        for i in range(N):
            x, t = test_data[i]
            x_len = x.shape[0]
            y = m.forward(x[None,:,:])
            
            mse = ((y - t) ** 2).sum() / x_len
            
            if mse < min_mse:
                min_mse = mse
                min_example_index = i
                
        min_x, min_t = test_data[min_example_index]
        print('model forecast', m.forward(min_x[None,:,:]))
        print('actual target', min_t)
        
    find_best_test_point(m, test_data)
    
    
# %% 250 epoch forecaster_fc_hidden, lstm

if False:
    etfs_path = './data/ETFs'
    etf_files = os.listdir(etfs_path)
    
    def only_close(data):
        return data[:,3].reshape(-1,1)
    
    train_start_date, train_end_date = '2010-01-01', '2013-01-01'
    val_start_date, val_end_date = train_end_date, '2015-01-01'
    test_start_date, test_end_date = val_end_date, '2030-01-01'
    
    train_data, val_data, test_data = \
      datap.date_make_train_val_test_data(etfs_path,
                                          train_start_date=train_start_date,
                                          train_end_date=train_end_date,
                                          val_start_date=val_start_date,
                                          val_end_date=val_end_date,
                                          test_start_date=test_start_date,
                                          test_end_date=test_end_date,
                                          process_data_func=only_close)
    
    print('len(train):', len(train_data))
    print('len(val):', len(val_data))
    print('len(test):', len(test_data))
    
    print('x,t shape', train_data[0][0].shape, train_data[0][1].shape)
    
    #augment data
    datap.augment(train_data, augment_func=datap.translate_price)
    
    print('len(train):', len(train_data))
    print('len(val):', len(val_data))
    print('len(test):', len(test_data))
    
    print('x,t shape', train_data[0][0].shape, train_data[0][1].shape)
    
    # model forecaster
    m = model.Forecaster_fc_hidden(input_features=1,
                                    encoder_hidden_features=100,
                                    fc_hidden=100,
                                    output_length=5)
    
    # train forecaster
    model_name = 'forecaster_fc_hidden(1,100,100,5)'
    checkpoint_path = './checkpoints/forecaster_fc_hidden_250'
    train.train_model(m, train_data, val_data, num_epochs=250, 
                      batch_size=10000,
                      learning_rate=1e-5,
                      checkpoint_path=checkpoint_path,
                      checkpoint_name=model_name)
    
    # model lstm
    m = model.LSTM(input_size=1,
                   output_size=1,
                   hidden=100,
                   layers=1)
    
    # train lstm
    model_name = 'LSTM(1,1,100,1)'
    checkpoint_path = './checkpoints/lstm_250'
    train.train_model(m, train_data, val_data, num_epochs=250, 
                      batch_size=10000,
                      learning_rate=1e-5,
                      checkpoint_path=checkpoint_path,
                      checkpoint_name=model_name)
    
# model prediction examples
if False:
    mod = model.Forecaster_fc_hidden(input_features=1,
                                    encoder_hidden_features=100,
                                    fc_hidden=100,
                                    output_length=5)
    
    saved_params_dict = './checkpoints/forecaster_fc_hidden_250/forecaster_fc_hidden(1,100,100,5)epoch_249.pt'
    mod.load_state_dict(torch.load(saved_params_dict))
    
    t_data = test_data[:5000]
    def plot_forecast(model, x, t):
        N = t.shape[0]
        y = model.forward(x[None,:,:]).detach().numpy()
        
        plt.title('Actual')
        plt.plot(np.arange(N).reshape(-1,1), t)
        plt.show()
        
        plt.title('Predicted')
        plt.plot(np.arange(N).reshape(-1,1), y.reshape(-1,1))
        plt.show()
        
    for i in range(20):
        x,t = t_data[i]
        plot_forecast(mod, x, t)
    
    pass
    