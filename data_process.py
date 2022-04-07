#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import csv
import math
import datetime
import random

# %%

def remove_volume_open_interest(data):
    '''
    remove volume and open interest columns from data (columns 5 and 6)
    '''
    return data[:,:-2]

def load_price_data_into_numpy_array(file_name, file_path, process_data=None):
    '''
    Load csv text file into numpy array
    
    where text file has data in the shape (N,M) 
        - first row is column headers
        - first column is date stamps
        
    if process_data is not none,
        - apply process_data to b
        - process_data should be a function that accepts array shape (N-1,M)
          and return an array of shape (N-1,Q)
    
    returns tuple (a, b)
        a - shape (N - 1, 1)
            - array of date strings
            - '<U10' datatype
        b - shape (N - 1, M - 1)
            - rest of the array 
            - 'float32' datatype
    '''
    path = file_path + '/' + file_name
    with open(path, 'rt') as file:
        csv_reader = csv.reader(file)
        array_list = np.array(list(csv_reader))
        array_list = array_list[1:] # remove column headers
        dates_array = array_list[:,0].astype('<U10').reshape(-1,1)
        data_array = array_list[:,1:].astype('float32')
        
        if process_data:
            data_array = process_data(data_array)
        
        return (dates_array, data_array)


# %%

def make_x_t_tuple_tensor_pairs_in_place(data, input_length=10, 
                                         output_length=5):
    ''' 
    Make list of (x,t) tensor pairs from numpy.array (float32), uses same 
    memory space as data
    
    Arguments:
        data - numpy array float32, shape [Q,M] 
         - Q days of M features of price data
         
        input_length, output_length - int
     
    returns:
        list of (x, t):
            x shape [input_length, M]
            t shape [output_length, M]
            x, t are torch.tensor (float32 dtype)
    '''
    N = data.shape[0]
    x_t_pairs = []
    for i in range(N - input_length - output_length):
        start = i
        end = i + input_length
        
        # extract example from data, and convert to torch.tensor with float32
        # dtype
        x = data[start:end]
        x = torch.from_numpy(x)
        t = data[end:end + output_length]
        t = torch.from_numpy(t)
        
        
        x_t_pairs.append((x,t))
        
    return x_t_pairs


# %% Functions that split data based off of specified dates

def date_add_to_train_val_test(date_data, data, train, val, test,
                               train_start_date, train_end_date, 
                               val_start_date, val_end_date, test_start_date,
                               test_end_date, x_length=30, t_length=5):
    '''
    Arguments:
        date_data - timeseries data shape [M,1]
            - M days of string dates 'YYYY-mm-dd'
        
        data - timeseries data shape [M,Q]
            - M days of timesteps
            - Q features per day
    
        train, test, val 
            - list of tuple (x, t) pairs
        
        train_start_date, train_end_date, val_start_date, val_end_date,
         test_start_date, test_end_date
            - string formatted date 'YYYY-mm-dd', e.g. '2022-01-15'
            
        x_length, t_length
            - integers to specify (x,t) sequence length
        
    Function does:
        - put timeseries data before train_end_date into train.
        - put timeseries data from train_end_date to val_end_date into val.
        - put timeseries data from val_end_date onwards into test data.
    '''
    N = data.shape[0]
    train_start_date = datetime.datetime.strptime(train_start_date, '%Y-%m-%d')
    train_end_date = datetime.datetime.strptime(train_end_date, '%Y-%m-%d')
    val_start_date = datetime.datetime.strptime(val_start_date, '%Y-%m-%d')
    val_end_date = datetime.datetime.strptime(val_end_date, '%Y-%m-%d')
    test_start_date = datetime.datetime.strptime(test_start_date, '%Y-%m-%d')
    test_end_date = datetime.datetime.strptime(test_end_date, '%Y-%m-%d')
    
    # find indexes to split data for train_end_date and val_end_date
    train_start_index = -1
    train_end_index = -1
    val_start_index = -1
    val_end_index = -1
    test_start_index = -1
    test_end_index = -1
    for i in range(N):
        date_i = datetime.datetime.strptime(date_data[i,0], '%Y-%m-%d')
        
        # train dates
        if date_i >= train_start_date and train_start_index == -1:
            train_start_index = int(i)
        elif date_i >= train_end_date and train_end_index == -1:
            train_end_index = int(i)
            
        # valid dates
        elif date_i >= val_start_date and val_start_index == -1:
            val_start_index = int(i)
        elif date_i >= val_end_date and val_end_index == -1:
            val_end_index = int(i)
            
        # test dates
        elif date_i >= test_start_date and test_start_index == -1:
            test_start_index = int(i)
        elif date_i >= test_end_date and test_end_index == -1:
            test_end_index = int(i)
            
    # split data into train, val, and test subsets
    train_to_add = data[train_start_index:train_end_index] # remove date column
    val_to_add = data[val_start_index:val_end_index]
    test_to_add = data[val_end_index:]
    
    # extend train, val, and test lists with data
    train += make_x_t_tuple_tensor_pairs_in_place(train_to_add, x_length,
                                                  t_length)
    val += make_x_t_tuple_tensor_pairs_in_place(val_to_add, x_length, 
                                                t_length)
    test += make_x_t_tuple_tensor_pairs_in_place(test_to_add, x_length, 
                                                 t_length)
    
    return train, val, test


def date_make_train_val_test_data(data_directory, x_length=30, t_length=5,
                                  train_start_date='2000-01-01',
                                  train_end_date='2014-01-01',
                                  val_start_date='2014-01-02',
                                  val_end_date='2015-06-01',
                                  test_start_date='2015-06-01',
                                  test_end_date='2030-01-01',
                                  process_data_func=None):
    '''
    data_directory - string representing path to data folder
        - e.g. './data/ETFs'
        
    separate all data in data_directy by date ranges specified in arguments.
    Returns tuple  (x,t) pairs for training, validating, and testing
    '''
    # individual data file names
    data_files = os.listdir(data_directory)
    train, val, test = [], [], []
    
    for file in data_files:
        file_dates, file_data = load_price_data_into_numpy_array(file,
                                data_directory, process_data=process_data_func)
        
        date_add_to_train_val_test(file_dates, file_data, train, val, test, 
                                   train_start_date, train_end_date,
                                   val_start_date, val_end_date,
                                   test_start_date, test_end_date,
                                   x_length, t_length)
    
    return train, val, test


# %% Split ETFs into train, val, test by symbol and date

def split_etfs(etfs, use_seed=True, random_seed=42):
    '''
    etfs - list of string etf file names
    
    splits into proportion 20,20,60:
        ([test], [val], [train])
    '''
    split = len(etfs) // 10
    train_start = 4 * split
    val_start = 2 * split
    val_end = train_start
    test_end = val_start
    
    if use_seed:
        random.seed(random_seed)
        random.shuffle(etfs)
    
    test = etfs[:test_end]
    val = etfs[val_start:val_end]
    train = etfs[train_start:]
    
    return test, val, train


def get_data_within_date_range(date_data, data, x_length=30, t_length=5, 
                               start_date='2000-01-01', end_date='2020-01-01'):
    '''
    Parameters
    ----------
    date_data : numpy array dtype '<U10'
        shape (N,1)
    data : numpy array dtype 'float32'
        shape (N,M)
    x_length : int
    t_length : int
    start_date : string date 'YYYY-mm-dd'
    end_date : string date 'YYYY-mm-dd'

    Returns
    -------
    x_t_pairs_in_range : list of tuples (x,t)
        x - shape (x_length, M)
        t - shape (t_length, M)
    '''
    N = data.shape[0]
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # start, end indexes corresponding to dates
    start_index = -1
    end_index = -1
    for i in range(N):
        date_i = datetime.datetime.strptime(date_data[i,0], '%Y-%m-%d')
        
        # train dates
        if date_i >= start_date and start_index == -1:
            start_index = int(i)
        elif date_i >= end_date and end_index == -1:
            end_index = int(i)
    
    data_in_range = data[start_index:end_index]
    x_t_pairs_in_range = make_x_t_tuple_tensor_pairs_in_place(data_in_range,
                                                              x_length,
                                                              t_length)
    
    return x_t_pairs_in_range


def data_split_symbol_and_date(train_files, val_files, test_files, path,
                               train_start_date, train_end_date, 
                               val_start_date, val_end_date,
                               test_start_date, test_end_date, x_length=30,
                               t_length=5, process_data_func=None):
    '''
    Construct training, validation, and test sets from lists specifying file
    names for each, only including (x,t) pairs within date ranges.
    
    Arguments:
        train, val, test 
         - list of text filenames
        
        path 
         - string directory location of data files
        
        train_start_date, train_end_date, val_start_date, val_end_date, 
         test_start_date, test_end_date
          - string formatted dates 'YYYY-mm-dd'
          
        x_length, t_length
         - int
         
        process_data_func - function, behaviour: (N,M) -> (N,Q)
    '''
    train, val, test = [], [], []
    
    # helper
    def add_x_t_data(files, path, x_length, t_length, start_date, 
                     end_date, add_to_list):
        '''
        Add (x,t) pairs to add_to_list
        '''
        for file_name in files:
            # data from file
            dates_array, data_array = load_price_data_into_numpy_array(file_name,
                                            path, process_data=process_data_func)
            
            # (x,t) pairs
            to_add = get_data_within_date_range(dates_array, data_array,
                                                x_length, t_length,
                                                train_start_date,
                                                train_end_date)
            
            # append (x,t) pairs
            add_to_list += to_add
        
        return
    
    # train data
    add_x_t_data(train_files, path, x_length, t_length, train_start_date,
                 train_end_date, train)
    
    # val data
    add_x_t_data(val_files, path, x_length, t_length, val_start_date,
                 val_end_date, val)
    
    # test data
    add_x_t_data(test_files, path, x_length, t_length, test_start_date,
                 test_end_date, test)
    
    return train, val, test


# %% Normalize data functions

def get_specific_date_data(date_data, data, start_date='2000-01-01', end_date='2020-01-01'):
    '''
    Parameters
    ----------
    date_data : numpy array dtype '<U10'
        - shape (N,1)
        - array of string formatted dates 'YYYY-mm-dd'
    data : numpy array dtype 'float32'
        - shape (N,M)
        - N timesteps of M data features
    start_date : string date 'YYYY-mm-dd'
    end_date : string date 'YYYY-mm-dd'

    Returns
    -------
    data_in_range : array shape (Q,M) - timesteps within date range
    '''
    N = data.shape[0]
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # start, end indexes corresponding to dates
    start_index = -1
    end_index = -1
    for i in range(N):
        date_i = datetime.datetime.strptime(date_data[i,0], '%Y-%m-%d')
        
        # train dates
        if date_i >= start_date and start_index == -1:
            start_index = int(i)
        elif date_i >= end_date and end_index == -1:
            end_index = int(i)
    
    data_in_range = data[start_index:end_index]

    return data_in_range


def normalize_single_stock(data):
    '''
    normalize each day of prices in data.
    
    Parameters
    ----------
    data_date : numpy array dtype 'float32'
        shape (N, M)

    Returns
    -------
    norm_data : numpy array dtype 'float32'
        shape (N, M)
    '''
    norm_data = np.empty((0,data.shape[1]))

    for days in data:
        min_price = min(days)
        max_price = max(days)
        normalize_data = (days-min_price)/(max_price-min_price)
        norm_data = np.append(norm_data,np.array([normalize_data]),axis=0)

    return norm_data


def normalize_as_avg_price(data):
    '''
    Parameters
    ----------
    data : array shape (N,4)
        N timesteps of Open, High, Low, Close data
    '''
    avg_daily_price = np.average(data, axis=1)
    
    norm_data = data / avg_daily_price.reshape(-1,1)
    
    return norm_data


def normalize_train_data(train_data, path_name, train_start_date, train_end_date,
                         x_length=30, t_length=5, 
                         normalizer=normalize_single_stock):
    '''
    return normalized (x,t) pairs from train_data.
    
    Parameters
    ----------
    train_data : list CSV files, in which each file represents a unique stock.
    
    path_name : directory containing data CSV files.

    train_start_date: string formatted date 'YYYY-mm-dd', e.g. '2022-01-15'

    train_end_date: string formatted date 'YYYY-mm-dd', e.g. '2022-01-15'

    normalizer - function(data: array shape (N,M))
        - should return normalized version of its array argument

    Returns
    -------
    normalized_data : List of (x,t) tuple tensor_pairs
        - x shape (x_length, 4)
        - t shape (t_length, 4)
    '''
    normalized_data = []

    
    for stock in train_data:
        numpy_data = load_price_data_into_numpy_array(stock,path_name,
                                                      process_data=remove_volume_open_interest)
  
        date_data = get_specific_date_data(numpy_data[0], numpy_data[1],
                                           train_start_date, train_end_date)
  
  
        norm_data = normalizer(date_data)
  
        pairs=make_x_t_tuple_tensor_pairs_in_place(norm_data,
                                                   input_length=x_length,
                                                   output_length=t_length)
        normalized_data += pairs
      
    return normalized_data


# %% Augment data functions

def add_noise_to_data_point(data_point):
    '''
    Return augmented datapoint by adding noise~N(0,1)
    '''
    x_shape = data_point[0].shape
    t_shape = data_point[1].shape
    
    new_x = torch.randn(x_shape)
    new_t = torch.randn(t_shape)
    
    return (new_x, new_t)


def translate_price(data_point, min_trans=10, max_trans=1000):
    '''
    Return augmented datapoint by translating values by a fixed amount while
    preserving the proportion between timesteps based on the first open price.
    
    Arguments:
        data_point - tuple (x,t)
            - x shape (M,N) - OHLC price format
            - t shape (O,P) - OHLC price format
    '''
    x = data_point[0]
    t = data_point[1]
    
    # proportion is based off the first open price of the input data
    start_price = x[0][0]
    x_proportion = x / start_price
    t_proportion = t / start_price
    
    # amount to translate datapoint by
    new_base_price = torch.tensor(random.uniform(min_trans, max_trans))
    
    new_x = x_proportion * new_base_price
    new_t = t_proportion * new_base_price
    
    return (new_x, new_t)
    

def augment(data, augment_func=add_noise_to_data_point, augment_proportion=0.5,
            random_seed=42):
    '''
    data - list of (x,t) pairs, length=N
        - x tensor shape (M,Q)
        - t tensor shape (P,R)
        
    augment_function((x,t))
        - returns augmented (x,t)
        
    augment_proportion - float
        - proportion to increase data by
            - i.e. add augment_proportion * N examples to data
    '''
    N = len(data)
    num_new_examples = math.floor(augment_proportion * N)
    random.seed(random_seed)
    
    for i in range(num_new_examples):
        index = random.randint(0,N-1)
        data_point = data[index]
        data.append(augment_func(data_point))
    
    return

def single_stock_data(single_stock, file_path, single_point_start_date, single_point_end_date):
  '''
  Parameters
  ----------
  single_stock: single stock csv files
  file_path: String representation of file path
  single_point_start_date: Start date of stock data
  single_point_end_date: End date of stock data

  Returns
  -------
  single_training_point: List of tuple_tensor_pair
  '''

  num_array = load_price_data_into_numpy_array(single_stock,file_path, process_data=remove_volume_open_interest)

  date_range_data = get_specific_date_data(num_array[0],num_array[1],single_point_start_date,single_point_end_date)

  single_training_point = make_x_t_tuple_tensor_pairs_in_place(date_range_data, input_length=10, output_length=5)

  return single_training_point


def small_data(training_data, file_path, stock_start_date, stock_end_date,n_stocks):
  '''
  Parameters
  ----------
  training_data: list of csv files where each csv file represents a single stock.
  file_path: String representation of file path
  stock_start_date: Start date of stock data
  stock_end_date: End date of stock data
  n_stock: The number of stocks 

  Returns
  -------
  training_points: List of tuple_tensor_pairs
  '''
  training_points = []
  stock_data = random.sample(training_data,n_stocks)

  for stocks in stock_data:

    single_stock_training_points = single_stock_data(stocks, file_path, stock_start_date, stock_end_date)
    training_points += single_stock_training_points

  return training_points


# %% If running this file standalone

if __name__ == '__main__':
    etfs_path = './data/ETFs'
    etf_files = os.listdir(etfs_path)
    
    # Split data by date range
    if False:
        train_data, val_data, test_data = date_make_train_val_test_data(etfs_path)
        
        print('len(train):', len(train_data))
        print('len(val):', len(val_data))
        print('len(test):', len(test_data))
        
        
    # split data by symbol and date range
    if False:
        test_etf_files, val_etf_files, train_etf_files = split_etfs(etf_files)
        print(len(etf_files), len(test_etf_files), 
              len(val_etf_files), len(train_etf_files))
        
        # example date ranges
        train_start, train_end = '2010-01-01', '2011-01-01'
        val_start, val_end = '2011-01-01', '2012-01-01'
        test_start, test_end = '2012-01-01', '2013-01-01'
        
        train_data, val_data, test_data = data_split_symbol_and_date(train_etf_files, val_etf_files, test_etf_files, etfs_path, train_start, train_end, val_start, val_end, test_start, test_end)
        
        print('len(train):', len(train_data))
        print('len(val):', len(val_data))
        print('len(test):', len(test_data))
        
    
    # Only OHLC split by date range
    if False:
        train_data, val_data, test_data = date_make_train_val_test_data(etfs_path, process_data_func=remove_volume_open_interest)
        
        print('len(train):', len(train_data))
        print('len(val):', len(val_data))
        print('len(test):', len(test_data))
        
    
    # normalize data example
    if False:
        test, val, train = split_etfs(etf_files)
        train_sample = train[1:5]
        train_start_date = '2010-01-01'
        train_end_date = '2017-02-02'
        
        # normalize_single_stock normalizer
        normalized_data = normalize_train_data(train_sample,
                                               etfs_path,
                                               train_start_date,
                                               train_end_date)
        print('len(normalized_data):', len(normalized_data))
        
        data_point = normalized_data[0]
        x, t = data_point
        print(x.shape)
        print(t.shape)
        
        # normalize_as_avg_price normalizer
        normalized_data = normalize_train_data(train_sample,
                                               etfs_path,
                                               train_start_date,
                                               train_end_date,
                                               normalizer=normalize_as_avg_price)
        print('len(normalized_data):', len(normalized_data))
        
        data_point = normalized_data[0]
        x, t = data_point
        print(x.shape)
        print(t.shape)

    # Small data example
    if False:
        test, val, train = split_etfs(etf_files)
        train_start_date = '2010-01-01'
        train_end_date = '2017-02-02'

        # Single data point
        single_stock_training_data = small_data(train,etfs_path,train_start_date,train_end_date,n_stocks=1)
        print('len(single_stock_training_data):', len(single_stock_training_data))
        
        data_point = single_stock_training_data[0]
        x, t = data_point
        print(x.shape)
        print(t.shape)

        # Small data set containing data of 5 stocks
        small_data_set = small_data(train,etfs_path,train_start_date,train_end_date,n_stocks=5)
        print('len(small_data_set):', len(small_data_set))
        
        stocks_data_point = small_data_set[0]
        one_stock_x, one_stock_t = stocks_data_point
        print(one_stock_x.shape)
        print(one_stock_t.shape)



# %%
