import math
import time
import pandas as pd
import numpy  as np 
import tensorflow as tf
import os
import ast
import pickle
from tensorflow import keras
from numpy import array 
from keras.layers import LSTM,GRU,ConvLSTM2D
from keras.layers import Dense,Dropout,Flatten,TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras import regularizers
from keras import optimizers
from keras.models import Model 
from tensorflow.keras.utils import Sequence
from keras.utils.vis_utils import plot_model

import json
import random
import uuid
import sys
import gc

def is_locked(filepath):
    """Checks if a file is locked by opening it in append mode.
    If no exception thrown, then the file is not locked.
    """
    locked = None
    file_object = None
    if os.path.exists(filepath):
        try:
            print("Trying to open " +  filepath)
            buffer_size = 8
            # Opening file in append mode and read the first 8 characters.
            file_object = open(filepath, 'a', buffer_size)
            if file_object:
                print (filepath + " is not locked.")
                locked = False
        except IOError as e:
            print("File is locked (unable to open in append mode.) " + e)
            locked = True
        finally:
            if file_object:
                file_object.close()
                print(filepath + " closed.")
    else:
        print(filepath + " not found.")
    return locked

def wait_for_files(filepaths):
    """Checks if the files are ready.

    For a file to be ready it must exist and can be opened in append
    mode.
    """
    wait_time = 5
    for filepath in filepaths:
        # If the file doesn't exist, wait wait_time seconds and try again
        # until it's found.
        while not os.path.exists(filepath):
            print(filepath + " hasn't arrived. Waiting ")
            time.sleep(wait_time)
        # If the file exists but locked, wait wait_time seconds and check
        # again until it's no longer locked by another process.
        while is_locked(filepath):
            print(filepath + " is currently in use. Waiting ") 
            time.sleep(wait_time)

processor = "CPU"

try:
    if 'gpu' in os.environ['CONDA_DEFAULT_ENV']:
        processor = "GPU"
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        policyConfig = 'mixed_float16'
        policy = tf.keras.mixed_precision.Policy(policyConfig)
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
except Exception:
    pass

SEED = 0


env = 'cloud'
if (os.path.expanduser("~") == '/home/doktormatte'):
    env = 'local'

import_path = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/'
export_path = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/results/'
if env == 'cloud':
    import_path = '/content/data/Data/'
    export_path = '/drive/MyDrive/'
    
def reset_tensorflow_keras_backend():    
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    _ = gc.collect()
    try:
        if 'gpu' in os.environ['CONDA_DEFAULT_ENV']:
            policyConfig = 'mixed_float16'
            policy = tf.keras.mixed_precision.Policy(policyConfig)
            tf.keras.mixed_precision.set_global_policy(policy)
    except Exception:
        pass

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

    
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

set_global_determinism(seed=SEED)

def computeHCF(x, y):
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            hcf = i
    return hcf


def pause():
    input("Press the <ENTER> key to continue...")

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)


def read_data_nh(df, n_steps_in, n_steps_out, n_features, architecture):    

    
    t_win = n_steps_in*n_steps_out
    Z = df
    Z=Z.to_numpy()   
    
    if architecture in ['CNN_LSTM', 'CNN_BiLSTM', 'CNN_GRU']:
        X, y = split_sequences(Z, t_win, n_steps_out )
        X = X.reshape((X.shape[0], n_steps_in, n_steps_out, n_features))
    elif architecture == 'ConvLSTM':
        X, y = split_sequences(Z, t_win, n_steps_out )
        X = X.reshape((X.shape[0], n_steps_in, 1, n_steps_out, n_features))
    else:
        X, y = split_sequences(Z, n_steps_in, n_steps_out)   
        
    return X, y

# test_df = pd.read_csv('/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/acn_caltech/energy/train_sum.csv')
# X,_ = read_data_nh(test_df, 4, 16, 107, 'LSTM')
# sys.exit()


def read_data_h(df1, df2, n_steps_in,n_steps_out,n_features):
     
    Z = df1
    Z = Z.to_numpy()     

    X, y = split_sequences(Z[:,10:10+n_features+1], n_steps_in, n_steps_out)        
    
    n_train = int(0.7*len(X))
    Z1 = df2    
    Z1 = Z1.to_numpy()
    Z1 = Z1.transpose()
    Z2 = np.concatenate((Z1,Z1),axis=1) 
    X2 = np.zeros([len(Z),10+96],float)  
  
    for i in range(len(Z)-n_steps_in): 
     if Z[i+n_steps_in-1,-1] == 0:             
          qq = np.array(Z2[0][0:96])
          X2[i] = np.append(Z[i+n_steps_in-1][0:10],qq) 
     else:            
         qq = np.array(Z2[1][0:96])
         X2[i] = np.append(Z[i+n_steps_in-1][0:10],qq) 
    
    X2 = X2[0:len(X),]
    return X, y, X2

def read_data_plc(df, n_steps_in, n_steps_out):   
    
    Z = df
    Z=Z.to_numpy()      
    X, y = split_sequences(Z, n_steps_in, n_steps_out)   
        
    return X, y



class NHgenerator(Sequence):
    def __init__(self, df, n_steps_in, n_steps_out, n_features, architecture, batch_size,stateful=False):
        self.df = df        
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_features = n_features       
        self.architecture = architecture
        self.batch_size = batch_size        
        
        # Call your data read function to create the input data and target values
        X, y = read_data_nh(self.df, self.n_steps_in, self.n_steps_out, self.n_features, self.architecture)
        
        if stateful: 
            train_size = X.shape[0]
            train_size = train_size-(train_size % 64)            
            X = X[:train_size]            
            y = y[:train_size]           
            self.batch_size = 64
        
        self.data = X
        self.targets = y         


    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]        

        return batch_data, batch_targets
    
    def get_data(self):
        return self.targets



class Hgenerator(Sequence):
    def __init__(self, df, df2, n_steps_in, n_steps_out, n_features, batch_size,stateful=False):
        self.df = df
        self.df2 = df2
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_features = n_features        
        self.batch_size = batch_size        
        
        # Call your data read function to create the input data and target values
        X, y, X2 = read_data_h(self.df, self.df2, self.n_steps_in, self.n_steps_out, self.n_features)
        
        if stateful: 
            train_size = X.shape[0]
            train_size = train_size-(train_size % 64)            
            X = X[:train_size]
            X2 = X2[:train_size]
            y = y[:train_size]           
            self.batch_size = 64
        
        self.data = X
        self.targets = y  
        self.data2 = X2


    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data2 = self.data2[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [batch_data,batch_data2], batch_targets
    
    def get_data(self):
        return self.targets



class PLCgenerator(Sequence):
    def __init__(self, df, n_steps_in, n_steps_out, batch_size,stateful=False):
        self.df = df
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        # self.n_features = n_features        
        self.batch_size = batch_size        
        
        # Call your data read function to create the input data and target values
        X, y = read_data_plc(self.df, self.n_steps_in, self.n_steps_out)
        
        if stateful: 
            train_size = X.shape[0]
            train_size = train_size-(train_size % 64)            
            X = X[:train_size]
            y = y[:train_size]           
            self.batch_size = 64
        
        self.data = X
        self.targets = y  


    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [batch_data,batch_data], batch_targets
    
    def get_data(self):
        return self.targets





    
    
glob_res_cols = ['id','model','dataset','rmse','mae','r_squared','accuracy','precision','recall','f1', 'fp','tp']		



def non_hybrid_loads(model_prop_dict,dirname):

    
    accuracies = []
    models = []
    architectures = ['LSTM', 'GRU', 'BiLSTM', 'Stacked', 'Conv1D', 'CNN_LSTM', 'CNN_BiLSTM', 'CNN_GRU' , 'ConvLSTM']
    architectures = ['LSTM', 'GRU', 'BiLSTM', 'Stacked', 'Conv1D']
    problem_architectures = ['CNN_LSTM', 'ConvLSTM','CNN_BiLSTM', 'CNN_GRU']
    random.shuffle(dirs)
    summary_cols = ['names','layers','dataset','rmse','mae','r_squared']
    val_run_cols = ['id', 'model', 'dataset', 'performance']
    val_splits = [4]
    train_test_split = 0.7
    
    iteration = 0
    
    
    
    # sys.exit()
    
    for k in range(1):
        
        
        architecture = model_prop_dict['architecture']  
        n_features = model_prop_dict['n_features']  
        n_steps_in = model_prop_dict['n_steps_in']  
        n_steps_out= model_prop_dict['n_steps_out']  
        po_size= model_prop_dict['po_size']  
        nf_1= model_prop_dict['nf_1']  
        nf_2= model_prop_dict['nf_2']  
        ker_size= model_prop_dict['ker_size']  
        stacked= model_prop_dict['stacked']  
        stack_size= model_prop_dict['stack_size']  
        nodes_1= model_prop_dict['nodes_1']  
        nodes_2= model_prop_dict['nodes_2']  
        nodes_3= model_prop_dict['nodes_3']  
        nodes_4= model_prop_dict['nodes_4']  
        activation_rec_1= model_prop_dict['activation_rec_1']  
        activation_rec_2= model_prop_dict['activation_rec_2']  
        activation_rec_3= model_prop_dict['activation_rec_3']  
        activation_rec_4= model_prop_dict['activation_rec_4']  
        dense_1= model_prop_dict['dense_1']  
        dropout= model_prop_dict['dropout']  
        bat_size= model_prop_dict['bat_size']  
        n_epoch= model_prop_dict['n_epoch']  
        # n_epoch = 1
        activation_1= model_prop_dict['activation_1']  
        activation_2= model_prop_dict['activation_2']  
        kernel_reg_1= model_prop_dict['kernel_reg_1']  
        kernel_reg_2= model_prop_dict['kernel_reg_2']  
        kernel_reg_3= model_prop_dict['kernel_reg_3']  
        bias_reg_1= model_prop_dict['bias_reg_1']  
        bias_reg_2= model_prop_dict['bias_reg_2']  
        bias_reg_3= model_prop_dict['bias_reg_3']  
        activ_reg_1= model_prop_dict['activ_reg_1']  
        activ_reg_2= model_prop_dict['activ_reg_2']  
        activ_reg_3= model_prop_dict['activ_reg_3']  
        optimizer= model_prop_dict['optimizer']  
        grad_clip= model_prop_dict['grad_clip']  
        stateful= model_prop_dict['stateful']  
        
        include_lags = model_prop_dict['include_lags']  
        include_avgs= model_prop_dict['include_avgs']  
        # learning_rate= model_prop_dict['learning_rate']  
        
        
        wait_for_files([export_path + dirname + '_real.csv']) 
        glob_res_table = pd.read_csv(export_path + dirname + '_real.csv')
        existing = glob_res_table[glob_res_table.dataset == dirname]
        exists = False
        for existing_dict in existing['model'].values:            
            if model_prop_dict == ast.literal_eval(existing_dict):            
                exists = True
                print('Already done...')
                continue
        if exists:
            time.sleep(3)
            continue  
        
        if optimizer == 1:
            # learning_rate = random.choice([0.0001, 0.001, 0.01])
            try:
                adam = optimizers.Adam(clipnorm=grad_clip,learning_rate= model_prop_dict['learning_rate'] )
            except:
                adam = optimizers.Adam(clipnorm=grad_clip)
        if optimizer == 2:
            # learning_rate = random.choice([0.0001, 0.001, 0.01, 0.1])
            try:
                adam = optimizers.SGD(clipnorm=grad_clip,learning_rate=model_prop_dict['learning_rate'] ,momentum=0.9)
            except:
                adam = optimizers.SGD(clipnorm=grad_clip,momentum=0.9)
        if optimizer == 0:        
            # learning_rate = random.choice([0.0001, 0.001, 0.01])
            try: 
                adam =optimizers.Adam(learning_rate=model_prop_dict['learning_rate'])
            except:
                adam =optimizers.Adam()
        # stateful = random.choice([True, False])
        if architecture == 'Conv1D':
            stateful=False
        if stateful:
            bat_size = 64
        po_size = 2
        if n_steps_out == 1:
            po_size = 1
        
        
        
        model_name = uuid.uuid4().hex        
       
        try:         
       
            
            print('\n')
            print('Pre-testing non-hybrid loads architecture ' + dirname)            
            print('\n')         
                
            temp_train_df = pd.read_csv(import_path + dirname + '/energy/train_sum.csv')
            averages = pd.read_csv(import_path + dirname + '/energy/averages_norm.csv')
            
            if include_lags == 1:
                for n in range(n_steps_in):
                    temp_train_df.insert(10+n, 'lag_' + str(n+1), [0.0]*len(temp_train_df), True)
                x = temp_train_df['value']
                for n in range(n_steps_in):
                    temp_train_df['lag_' + str(n+1)] = np.roll(x,n+1)
                temp_train_df.insert(10, 'lag_1w', np.roll(x,96*7), True)
                temp_train_df.insert(10, 'lag_1d', np.roll(x,96), True)
                temp_train_df = temp_train_df[96*7:]    
    
            if include_avgs == 1:
                for m in range(96):
                    temp_train_df.insert(10+m, str(10+m), [0.0]*len(temp_train_df), True)    
                temp_train_df.loc[temp_train_df['weekend'] == 0, [str(x) for x in list(range(10,106))]] = np.array(averages.weekday.T)
                temp_train_df.loc[temp_train_df['weekend'] == 1, [str(x) for x in list(range(10,106))]] = np.array(averages.weekend.T)  
                       
            # X_train, y_train = read_data_nh(temp_train_df, n_steps_in, n_steps_out, n_features, architecture)
            
            temp_test_df = pd.read_csv(import_path + dirname + '/energy/test_sum.csv')
            averages = pd.read_csv(import_path + dirname + '/energy/averages_norm.csv')
            
            if include_lags == 1:
                for n in range(n_steps_in):
                    temp_test_df.insert(10+n, 'lag_' + str(n+1), [0.0]*len(temp_test_df), True)
                x = temp_test_df['value']
                for n in range(n_steps_in):
                    temp_test_df['lag_' + str(n+1)] = np.roll(x,n+1)
                temp_test_df.insert(10, 'lag_1w', np.roll(x,96*7), True)
                temp_test_df.insert(10, 'lag_1d', np.roll(x,96), True)
                temp_test_df = temp_test_df[96*7:]    
    
            if include_avgs == 1:
                for m in range(96):
                    temp_test_df.insert(10+m, str(10+m), [0.0]*len(temp_test_df), True)    
                temp_test_df.loc[temp_test_df['weekend'] == 0, [str(x) for x in list(range(10,106))]] = np.array(averages.weekday.T)
                temp_test_df.loc[temp_test_df['weekend'] == 1, [str(x) for x in list(range(10,106))]] = np.array(averages.weekend.T)  
                       
            # X_test, y_test = read_data_nh(temp_test_df, n_steps_in, n_steps_out, n_features, architecture)
    
        
        except Exception as e:
            print(e)
            continue
        
        # if architecture in ['LSTM', 'GRU', 'BiLSTM', 'Stacked', 'CNN_LSTM', 'ConvLSTM','CNN_BiLSTM', 'CNN_GRU']:
        #     if stateful:
        #         train_size = X_train.shape[0]
        #         train_size = train_size-(train_size % 64)
        #         test_size = X_test.shape[0]
        #         test_size = test_size-(test_size % 64)
        #         X_train = X_train[:train_size]
        #         y_train = y_train[:train_size]
        #         X_test = X_test[:test_size]   
        #         y_test = y_test[:test_size] 
        #         bat_size = 64
            
        model = keras.Sequential()
        
        if architecture == 'Conv1D':
            if stateful:
                model.add(Conv1D(kernel_regularizer=kernel_reg_1,
                                     bias_regularizer=bias_reg_1,
                                     activity_regularizer=activ_reg_1,
                                     filters=nf_1, kernel_size=ker_size,
                                     activation=activation_1,
                                     batch_input_shape=(bat_size,n_steps_in, n_features))) 
            else:
                model.add(Conv1D(kernel_regularizer=kernel_reg_1,
                                     bias_regularizer=bias_reg_1,
                                     activity_regularizer=activ_reg_1,
                                     filters=nf_1, kernel_size=ker_size,
                                     activation=activation_1,
                                     input_shape=(n_steps_in, n_features))) 
            model.add(Conv1D(kernel_regularizer=kernel_reg_2,
                                 bias_regularizer=bias_reg_2,
                                 activity_regularizer=activ_reg_2,
                                 filters=nf_2, kernel_size=ker_size, activation=activation_1)) 
            model.add(MaxPooling1D(pool_size=po_size))
            model.add(Flatten())   
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_3,
                                bias_regularizer=bias_reg_3,
                                activity_regularizer=activ_reg_3,
                                activation=activation_2))
            model.add(Dropout(dropout)) 
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
            
        if architecture == 'ConvLSTM':
            if stateful:
                model.add(ConvLSTM2D(filters=nf_1, kernel_size=(1,ker_size),activation=activation_1, stateful=stateful, batch_input_shape=(bat_size,n_steps_in, 1, n_steps_out, n_features)))
            else:
                model.add(ConvLSTM2D(filters=nf_1, kernel_size=(1,ker_size),activation=activation_1, stateful=stateful, input_shape=(n_steps_in, 1, n_steps_out, n_features)))
                  
            model.add(Flatten())
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_1,
                                bias_regularizer=bias_reg_1,
                                activity_regularizer=activ_reg_1,
                                activation=activation_1))
            model.add(Dropout(dropout))  
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
        
        if architecture == 'CNN_LSTM':
            if stateful:
                model.add(TimeDistributed(Conv1D(
                    kernel_regularizer=kernel_reg_1,
                    bias_regularizer=bias_reg_1,
                    activity_regularizer=activ_reg_1,
                    filters=nf_1, kernel_size=ker_size,padding='same',activation=activation_1), batch_input_shape=(bat_size,n_steps_in, n_steps_out, n_features)))
            else:
                model.add(TimeDistributed(Conv1D(
                    kernel_regularizer=kernel_reg_1,
                    bias_regularizer=bias_reg_1,
                    activity_regularizer=activ_reg_1,
                    filters=nf_1, kernel_size=ker_size,padding='same',activation=activation_1), input_shape=(n_steps_in, n_steps_out, n_features)))
            model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_2,
                bias_regularizer=bias_reg_2,
                activity_regularizer=activ_reg_2,filters=nf_2, kernel_size=ker_size, padding='same',activation=activation_1))) 
            model.add(TimeDistributed(MaxPooling1D(pool_size=po_size,padding='same')))
            model.add(TimeDistributed(Flatten())) 
            model.add(LSTM(nodes_1, stateful=stateful)) 
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_3,
                                bias_regularizer=bias_reg_3,
                                activity_regularizer=activ_reg_3, activation=activation_1))
            model.add(Dropout(dropout))   
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])  
    
    
        if architecture == 'CNN_BiLSTM':
            if stateful:
                model.add(TimeDistributed(Conv1D(
                    kernel_regularizer=kernel_reg_1,
                    bias_regularizer=bias_reg_1,
                    activity_regularizer=activ_reg_1,
                    filters=nf_1, kernel_size=ker_size,padding='same',activation=activation_1), batch_input_shape=(bat_size,n_steps_in, n_steps_out, n_features)))
            else:
                model.add(TimeDistributed(Conv1D(
                    kernel_regularizer=kernel_reg_1,
                    bias_regularizer=bias_reg_1,
                    activity_regularizer=activ_reg_1,
                    filters=nf_1, kernel_size=ker_size,padding='same',activation=activation_1), input_shape=(n_steps_in, n_steps_out, n_features)))
            model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_2,
                bias_regularizer=bias_reg_2,
                activity_regularizer=activ_reg_2,filters=nf_2, kernel_size=ker_size, padding='same',activation=activation_1))) 
            model.add(TimeDistributed(MaxPooling1D(pool_size=po_size,padding='same')))
            model.add(TimeDistributed(Flatten())) 
            model.add(keras.layers.Bidirectional(LSTM(nodes_1, stateful=stateful))) 
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_3,
                                bias_regularizer=bias_reg_3,
                                activity_regularizer=activ_reg_3, activation=activation_1))
            model.add(Dropout(dropout))   
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])  
    
        if architecture == 'CNN_GRU':
            if stateful:
                model.add(TimeDistributed(Conv1D(
                    kernel_regularizer=kernel_reg_1,
                    bias_regularizer=bias_reg_1,
                    activity_regularizer=activ_reg_1,
                    filters=nf_1, kernel_size=ker_size,padding='same',activation=activation_1), batch_input_shape=(bat_size,n_steps_in, n_steps_out, n_features)))
            else:
                model.add(TimeDistributed(Conv1D(
                    kernel_regularizer=kernel_reg_1,
                    bias_regularizer=bias_reg_1,
                    activity_regularizer=activ_reg_1,
                    filters=nf_1, kernel_size=ker_size,padding='same',activation=activation_1), input_shape=(n_steps_in, n_steps_out, n_features)))
            model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_2,
                bias_regularizer=bias_reg_2,
                activity_regularizer=activ_reg_2,filters=nf_2, kernel_size=ker_size, padding='same',activation=activation_1))) 
            model.add(TimeDistributed(MaxPooling1D(pool_size=po_size,padding='same')))
            model.add(TimeDistributed(Flatten())) 
            model.add(GRU(nodes_1, stateful=stateful))
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_3,
                                bias_regularizer=bias_reg_3,
                                activity_regularizer=activ_reg_3, activation=activation_1))
            model.add(Dropout(dropout))   
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])             
    
        if architecture == 'Stacked':
            if stacked == 1:
                if stateful:
                    if stack_size == 2:
                        model.add(keras.layers.LSTM(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_2,activation=activation_rec_2, stateful=stateful))
                    if stack_size == 3:
                        model.add(keras.layers.LSTM(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_3,activation=activation_rec_3, stateful=stateful))
                    if stack_size == 4:
                        model.add(keras.layers.LSTM(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_3,return_sequences=True,activation=activation_rec_3, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_4,activation=activation_rec_4, stateful=stateful))
                else:
                    if stack_size == 2:
                        model.add(keras.layers.LSTM(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_2,activation=activation_rec_2, stateful=stateful))
                    if stack_size == 3:
                        model.add(keras.layers.LSTM(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_3,activation=activation_rec_3, stateful=stateful))
                    if stack_size == 4:
                        model.add(keras.layers.LSTM(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_3,return_sequences=True,activation=activation_rec_3, stateful=stateful))
                        model.add(keras.layers.LSTM(nodes_4,activation=activation_rec_4, stateful=stateful))
                    
            if stacked == 0:
                if stateful:
                    if stack_size == 2:                    
                        model.add(keras.layers.GRU(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.GRU(nodes_2,activation=activation_rec_2, stateful=stateful))   
                    if stack_size == 3:                    
                        model.add(keras.layers.GRU(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.GRU(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful))   
                        model.add(keras.layers.GRU(nodes_3,activation=activation_rec_3, stateful=stateful))  
                    if stack_size == 4:
                        model.add(keras.layers.GRU(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.GRU(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful))   
                        model.add(keras.layers.GRU(nodes_3,return_sequences=True,activation=activation_rec_3, stateful=stateful))
                        model.add(keras.layers.GRU(nodes_4,activation=activation_rec_4, stateful=stateful))
                else:
                    if stack_size == 2:                    
                        model.add(keras.layers.GRU(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.GRU(nodes_2,activation=activation_rec_2, stateful=stateful))   
                    if stack_size == 3:                    
                        model.add(keras.layers.GRU(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.GRU(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful))   
                        model.add(keras.layers.GRU(nodes_3,activation=activation_rec_3, stateful=stateful))  
                    if stack_size == 4:
                        model.add(keras.layers.GRU(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful))
                        model.add(keras.layers.GRU(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful))   
                        model.add(keras.layers.GRU(nodes_3,return_sequences=True,activation=activation_rec_3, stateful=stateful))
                        model.add(keras.layers.GRU(nodes_4,activation=activation_rec_4, stateful=stateful))                  
                        
            if stacked == 2:
                if stateful:
                    if stack_size == 2:
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_2,activation=activation_rec_2, stateful=stateful)))
                    if stack_size == 3:
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_3,activation=activation_rec_3, stateful=stateful)))
                    if stack_size == 4:
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = nodes_1,return_sequences=True,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_3,return_sequences=True,activation=activation_rec_3, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_4,activation=activation_rec_4, stateful=stateful)))
                else:
                    if stack_size == 2:
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_2,activation=activation_rec_2, stateful=stateful)))
                    if stack_size == 3:
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_3,activation=activation_rec_3, stateful=stateful)))
                    if stack_size == 4:
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = nodes_1,return_sequences=True,input_shape=(n_steps_in,temp_train_df.shape[1]-1),activation=activation_rec_1, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_2,return_sequences=True,activation=activation_rec_2, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_3,return_sequences=True,activation=activation_rec_3, stateful=stateful)))
                        model.add(keras.layers.Bidirectional(keras.layers.LSTM(nodes_4,activation=activation_rec_4, stateful=stateful)))
                
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_1,
                                bias_regularizer=bias_reg_1,
                                activity_regularizer=activ_reg_1,activation=activation_1))
            model.add(Dropout(dropout))  
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))                
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
            
        if architecture == 'BiLSTM':
            if stateful:
                model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=nodes_1, stateful=stateful,batch_input_shape=(bat_size,n_steps_in, temp_train_df.shape[1]-1))))
            else:
                model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=nodes_1, stateful=stateful,input_shape=(n_steps_in, temp_train_df.shape[1]-1))))
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_1,
                                bias_regularizer=bias_reg_1,
                                activity_regularizer=activ_reg_1,activation=activation_1))
            model.add(Dropout(dropout))  
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))                
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
        
        if architecture == 'GRU':
            if stateful:
                model.add(keras.layers.GRU(units = nodes_1, stateful=stateful,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1)))
            else:
                model.add(keras.layers.GRU(units = nodes_1, stateful=stateful,input_shape=(n_steps_in,temp_train_df.shape[1]-1)))
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_1,
                                bias_regularizer=bias_reg_1,
                                activity_regularizer=activ_reg_1,activation=activation_1))
            model.add(Dropout(dropout))  
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))                
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
            
        if architecture == 'LSTM':
            if stateful:
                model.add(keras.layers.LSTM(units = nodes_1, stateful=stateful,batch_input_shape=(bat_size,n_steps_in,temp_train_df.shape[1]-1)))
            else:
                model.add(keras.layers.LSTM(units = nodes_1, stateful=stateful,input_shape=(n_steps_in,temp_train_df.shape[1]-1)))
            model.add(Dense(dense_1,
                                kernel_regularizer=kernel_reg_1,
                                bias_regularizer=bias_reg_1,
                                activity_regularizer=activ_reg_1,activation=activation_1))
            model.add(Dropout(dropout))  
            model.add(Dense(n_steps_out, activation=activation_2,dtype=tf.float32))                
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse']) 
        
        
        train_generator = NHgenerator(temp_train_df, n_steps_in, n_steps_out, n_features, architecture, batch_size=bat_size,stateful=stateful)
        test_generator = NHgenerator(temp_test_df, n_steps_in, n_steps_out, n_features, architecture, batch_size=bat_size,stateful=stateful)
        
        model.build(input_shape=(n_steps_in, n_steps_out, n_features))
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=n_epoch//8,restore_best_weights=True)
        dot_img_file = '/home/doktormatte/Pictures/' + str(model_name) +'.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        sys.exit()
        if stateful:
            history = model.fit(
                train_generator,
                epochs=n_epoch,
                batch_size=bat_size,
                shuffle=False,
                callbacks=[callback]
                )
            train_loss = pd.DataFrame(history.history['loss'])  
            temp = model.predict(test_generator,batch_size=bat_size)
        else:
            history = model.fit(
                train_generator,
                epochs=n_epoch,                    
                shuffle=False,
                callbacks=[callback]
                )
            train_loss = pd.DataFrame(history.history['loss'])
            temp = model.predict(test_generator)
        
        filename = export_path + 'models/' + str(model_name) + '.sav'
        pickle.dump(model, open(filename, 'wb'))
        train_loss.to_csv(export_path + 'models/' + str(model_name) + '.csv')
        
        reset_tensorflow_keras_backend()
        
        m,n=temp.shape 
        t_target = n_steps_out               
        yhat=np.zeros((m,t_target))
        # y_obs=np.array(y_test[0:m,0:t_target])
        y_obs = test_generator.get_data()
        scores= np.zeros(m)     
        
        try:
            rmses = np.zeros(m)
            maes = np.zeros(m)
            r2s = np.zeros(m)
            for j in np.arange(m):                        
                rmse = mean_squared_error(y_obs[j,:], temp[j,:], squared=False)
                mae = mean_absolute_error(y_obs[j,:], temp[j,:])
                _r_squared = r2_score(y_obs[j,:], temp[j,:])
                rmses[j] = 1.0 - rmse
                maes[j] = mae
                r2s[j] = _r_squared
            
            perf = np.mean(rmses)                  
            
            
            row = pd.DataFrame(columns=glob_res_cols)            
            row.id = [model_name]
            row.dataset = [dirname]
            row.model = [model_prop_dict]            
            row.rmse = perf
            row.mae = np.mean(maes)
            row.r_squared = r2_score(y_obs[:,0], temp[:,0])
            
            wait_for_files([export_path + dirname + '_real.csv']) 
            glob_res_table = pd.read_csv(export_path + dirname + '_real.csv')
            glob_res_table=pd.concat([glob_res_table,row])
            glob_res_table.to_csv(export_path + dirname + '_real.csv', encoding='utf-8',index=False)
        except Exception:
            pass
        
        
        
        # sys.exit()

def hybrid_loads(model_prop_dict,dirname):    


    random.shuffle(dirs)
    val_run_cols = ['id', 'model', 'dataset', 'performance']   
    
    for i in range(1):               
            
        n_features=model_prop_dict['n_features']  
        n_steps_in=model_prop_dict['n_steps_in']  
        n_steps_out=model_prop_dict['n_steps_out']  
        dropout_1=model_prop_dict['dropout_1']  
        dropout_2=model_prop_dict['dropout_2']  
        n_n_lstm_1=model_prop_dict['n_n_lstm_1']  
        n_n_lstm_2=model_prop_dict['n_n_lstm_2']  
        n_n_lstm_3=model_prop_dict['n_n_lstm_3']  
        n_n_lstm_4=model_prop_dict['n_n_lstm_4']  
        bat_size=model_prop_dict['bat_size']  
        n_epoch=model_prop_dict['n_epoch']  
        dense_1=model_prop_dict['dense_1']  
        dense_2=model_prop_dict['dense_2']  
        dense_3=model_prop_dict['dense_3']  
        dense_4=model_prop_dict['dense_4']  
        dense_5=model_prop_dict['dense_5']  
        dense_6=model_prop_dict['dense_6']  
        dense_7=model_prop_dict['dense_7']  
        filters_1=model_prop_dict['filters_1']  
        filters_2=model_prop_dict['filters_2']  
        activation_dense=model_prop_dict['activation_dense']  
        activation_conv=model_prop_dict['activation_conv']  
        activation_lstm=model_prop_dict['activation_lstm']  
        convolve=model_prop_dict['convolve']  
        stack_layers=model_prop_dict['stack_layers']  
        kernel_reg_1=model_prop_dict['kernel_reg_1']  
        kernel_reg_2=model_prop_dict['kernel_reg_2']  
        kernel_reg_3=model_prop_dict['kernel_reg_3']  
        kernel_reg_4=model_prop_dict['kernel_reg_4']  
        kernel_reg_5=model_prop_dict['kernel_reg_5']  
        kernel_reg_6=model_prop_dict['kernel_reg_6']  
        bias_reg_1=model_prop_dict['bias_reg_1']  
        bias_reg_2=model_prop_dict['bias_reg_2']  
        bias_reg_3=model_prop_dict['bias_reg_3']  
        bias_reg_4=model_prop_dict['bias_reg_4']  
        bias_reg_5=model_prop_dict['bias_reg_5']  
        bias_reg_6=model_prop_dict['bias_reg_6']  
        activ_reg_1=model_prop_dict['activ_reg_1']  
        activ_reg_2=model_prop_dict['activ_reg_2']  
        activ_reg_3=model_prop_dict['activ_reg_3']  
        activ_reg_4=model_prop_dict['activ_reg_4']  
        activ_reg_5=model_prop_dict['activ_reg_5']  
        activ_reg_6=model_prop_dict['activ_reg_6']  
        optimizer=model_prop_dict['optimizer']  
        grad_clip=model_prop_dict['grad_clip']  
        stateful=model_prop_dict['stateful']  
        include_lags  =model_prop_dict['include_lags']  
        if stateful:
            bat_size = 64
        model_name = uuid.uuid4().hex      
        
        wait_for_files([export_path + dirname + '_real.csv']) 
        glob_res_table = pd.read_csv(export_path + dirname + '_real.csv')
        existing = glob_res_table[glob_res_table.dataset == dirname]
        exists = False
        for existing_dict in existing['model'].values:            
            if model_prop_dict == ast.literal_eval(existing_dict):            
                exists = True
                print('Already done...')
                continue
        if exists:
            time.sleep(3)
            continue  
        
        if optimizer == 1:
            # learning_rate = random.choice([0.0001, 0.001, 0.01])
            try:
                adam = optimizers.Adam(clipnorm=grad_clip,learning_rate= model_prop_dict['learning_rate'] )
            except:
                adam = optimizers.Adam(clipnorm=grad_clip)
        if optimizer == 2:
            # learning_rate = random.choice([0.0001, 0.001, 0.01, 0.1])
            try:
                adam = optimizers.SGD(clipnorm=grad_clip,learning_rate=model_prop_dict['learning_rate'] ,momentum=0.9)
            except:
                adam = optimizers.SGD(clipnorm=grad_clip,momentum=0.9)
        if optimizer == 0:        
            # learning_rate = random.choice([0.0001, 0.001, 0.01])
            try: 
                adam =optimizers.Adam(learning_rate=model_prop_dict['learning_rate'])
            except:
                adam =optimizers.Adam()
        # stateful = random.choice([True, False])
        # if architecture == 'Conv1D':
        #     stateful=False
        po_size = 2
        if n_steps_out == 1:
            po_size = 1
        
        
        
        val_run_res = pd.DataFrame(columns=val_run_cols) 
        # for dirname in dirs:      
            
        dirname = random.choice(dirs)
        print('\n')
        print('Pre-testing hybrid loads  architecture ' + dirname)            
        print('\n') 
        
        
        temp_train_df = pd.read_csv(import_path + dirname + '/energy/train_sum.csv')
        if include_lags == 1:
            for n in range(n_steps_in):
                temp_train_df.insert(10+n, 'lag_' + str(n+1), [0.0]*len(temp_train_df), True)
            x = temp_train_df['value']
            for n in range(n_steps_in):
                temp_train_df['lag_' + str(n+1)] = np.roll(x,n+1)
            temp_train_df.insert(10, 'lag_1w', np.roll(x,96*7), True)
            temp_train_df.insert(10, 'lag_1d', np.roll(x,96), True)
            temp_train_df = temp_train_df[96*7:]
        
        # station_1_train = import_path + dirname + '/energy/train_sum.csv'
        station_2_train = pd.read_csv(import_path + dirname + '/energy/averages_norm.csv')
        # X_train, y_train, X2_train = read_data_h(temp_train_df,station_2_train,n_steps_in,n_steps_out,n_features)
        
        temp_test_df = pd.read_csv(import_path + dirname + '/energy/test_sum.csv')
        if include_lags == 1:
            for n in range(n_steps_in):
                temp_test_df.insert(10+n, 'lag_' + str(n+1), [0.0]*len(temp_test_df), True)
            x = temp_test_df['value']
            for n in range(n_steps_in):
                temp_test_df['lag_' + str(n+1)] = np.roll(x,n+1)
            temp_test_df.insert(10, 'lag_1w', np.roll(x,96*7), True)
            temp_test_df.insert(10, 'lag_1d', np.roll(x,96), True)
            temp_test_df = temp_test_df[96*7:]
        
        # station_1_test = import_path + dirname + '/energy/test_sum.csv'
        station_2_test = pd.read_csv(import_path + dirname + '/energy/averages_norm.csv')
        # X_test, y_test, X2_test = read_data_h(temp_test_df,station_2_test,n_steps_in,n_steps_out,n_features)

                    
        
        if convolve==1:                                           
            input2 = keras.Input(shape=(106,1))  
            meta_layer=Dense(dense_1,
                             kernel_regularizer=kernel_reg_1,
                             bias_regularizer=bias_reg_1,
                             activity_regularizer=activ_reg_1,activation=activation_dense)(input2)
            meta_layer = keras.layers.Conv1D(
                kernel_regularizer=kernel_reg_2,
                bias_regularizer=bias_reg_2,
                activity_regularizer=activ_reg_2,filters=filters_1, kernel_size=1, activation=activation_conv)(input2)
            meta_layer = keras.layers.Conv1D(
                kernel_regularizer=kernel_reg_3,
                bias_regularizer=bias_reg_3,
                activity_regularizer=activ_reg_3,
                filters=filters_2, kernel_size=1, activation=activation_conv)(meta_layer)
            meta_layer = keras.layers.MaxPool1D(pool_size=2)(meta_layer)
            meta_layer = keras.layers.Flatten()(meta_layer)
        else:
            input2 = keras.Input(shape=(106,))  
            meta_layer = keras.layers.Dense(106, activation=activation_dense)(input2)                   
        

        if stateful:                
            input1 = keras.Input(shape=(n_steps_in, n_features),batch_size=bat_size)                        
        else:
            input1 = keras.Input(shape=(n_steps_in, n_features))                        
        if stack_layers==3:
            model_LSTM = LSTM(n_n_lstm_1, return_sequences=True, activation=activation_lstm, stateful=stateful)(input1)
            model_LSTM = LSTM(n_n_lstm_2, return_sequences=True, activation=activation_lstm, stateful=stateful)(input1)                        
            model_LSTM=LSTM(n_n_lstm_3, activation=activation_lstm, stateful=stateful)(input1)
        if stack_layers==2:
            model_LSTM = LSTM(n_n_lstm_1, return_sequences=True, activation=activation_lstm, stateful=stateful)(input1)
            model_LSTM=LSTM(n_n_lstm_3, activation=activation_lstm, stateful=stateful)(input1)
        else:                        
            model_LSTM=LSTM(n_n_lstm_3, activation=activation_lstm, stateful=stateful)(input1)                       
            
            
        model_LSTM=Dropout(dropout_1)(model_LSTM)
        model_LSTM=Dense(dense_1,
                         kernel_regularizer=kernel_reg_4,
                         bias_regularizer=bias_reg_4,
                         activity_regularizer=activ_reg_4,activation=activation_dense)(model_LSTM)                    
        meta_layer = keras.layers.Dense(dense_2,
                                        kernel_regularizer=kernel_reg_5,
                                        bias_regularizer=bias_reg_5,
                                        activity_regularizer=activ_reg_5,activation=activation_dense)(meta_layer)    
        meta_layer = keras.layers.Dense(dense_3,
                                        kernel_regularizer=kernel_reg_5,
                                        bias_regularizer=bias_reg_5,
                                        activity_regularizer=activ_reg_5,activation=activation_dense)(meta_layer)
        model_merge = keras.layers.concatenate([model_LSTM, meta_layer])
        model_merge = Dense(dense_4,
                            kernel_regularizer=kernel_reg_1,
                            bias_regularizer=bias_reg_1,
                            activity_regularizer=activ_reg_1,
                            activation=activation_dense)(model_merge)
        model_merge = Dropout(dropout_2)(model_merge)                      

        
        output = Dense(n_steps_out, activation=activation_dense,dtype=tf.float32)(model_merge)
        model = Model(inputs=[input1, input2], outputs=output) 
        
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=n_epoch//8,restore_best_weights=True)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
        
        
        train_generator = Hgenerator(temp_train_df, station_2_train, n_steps_in, n_steps_out, n_features, batch_size=bat_size,stateful=stateful)
        test_generator = Hgenerator(temp_test_df, station_2_test, n_steps_in, n_steps_out, n_features, batch_size=bat_size,stateful=stateful)
        
        
        if stateful:
            history = model.fit(train_generator, epochs=n_epoch, batch_size=bat_size,shuffle=False,callbacks=[callback])  
            train_loss = pd.DataFrame(history.history['loss'])  
            temp = model.predict(test_generator,batch_size=bat_size)
        else:
            history = model.fit(train_generator, epochs=n_epoch, shuffle=False,callbacks=[callback])       
            train_loss = pd.DataFrame(history.history['loss'])  
            temp = model.predict(test_generator)
            
        filename = export_path + 'models/' + str(model_name) + '.sav'
        pickle.dump(model, open(filename, 'wb'))
        train_loss.to_csv(export_path + 'models/' + str(model_name) + '.csv')
        
        reset_tensorflow_keras_backend()
        
        m,n=temp.shape 
        t_target = n_steps_out               
        yhat=np.zeros((m,t_target))
        y_obs = test_generator.get_data()
        # y_obs=np.array(y_test[0:m,0:t_target])
        scores= np.zeros(m)     
        
        try:
            rmses = np.zeros(m)
            maes = np.zeros(m)
            r2s = np.zeros(m)
            for j in np.arange(m):                        
                rmse = mean_squared_error(y_obs[j,:], temp[j,:], squared=False)
                mae = mean_absolute_error(y_obs[j,:], temp[j,:])
                _r_squared = r2_score(y_obs[j,:], temp[j,:])
                rmses[j] = 1.0 - rmse
                maes[j] = mae
                r2s[j] = _r_squared
            
            perf = np.mean(rmses)          
            
            
            
            row = pd.DataFrame(columns=glob_res_cols)            
            row.id = [model_name]
            row.dataset = [dirname]
            row.model = [model_prop_dict]            
            row.rmse = perf
            row.mae = np.mean(maes)
            row.r_squared = r2_score(y_obs[:,0], temp[:,0])
            
            
            
            wait_for_files([export_path + dirname + '_real.csv']) 
            glob_res_table = pd.read_csv(export_path + dirname + '_real.csv')
            glob_res_table=pd.concat([glob_res_table,row])
            glob_res_table.to_csv(export_path + dirname + '_real.csv', encoding='utf-8',index=False)
        except Exception:
            pass    

            
            
            
            
   
            
            
def plc_loads(model_prop_dict,dirname):   


    random.shuffle(dirs)
    iteration = 0
    summary_cols = ['names','layers','dataset','rmse','mae','r_squared']
    val_run_cols = ['id', 'model', 'dataset', 'performance']
    val_splits = [4]   
    
    
    for i in range(1):           
            
        n_features=model_prop_dict['n_features']  
        n_steps_in=model_prop_dict['n_steps_in']  
        n_steps_out=model_prop_dict['n_steps_out']  
        n_n_lstm_1=model_prop_dict['n_n_lstm_1']  
        n_n_lstm_2=model_prop_dict['n_n_lstm_2']  
        bat_size=model_prop_dict['bat_size']  
        n_epoch=model_prop_dict['n_epoch'] 
        # n_epoch=1
        activation_dense=model_prop_dict['activation_dense']  
        activation_conv=model_prop_dict['activation_conv']  
        activation_lstm=model_prop_dict['activation_lstm']  
        filters_1=model_prop_dict['filters_1']  
        filters_2=model_prop_dict['filters_2']  
        filters_3=model_prop_dict['filters_3']  
        dropout_1=model_prop_dict['dropout_1']  
        dropout_2=model_prop_dict['dropout_2']  
        dense_1=model_prop_dict['dense_1']  
        dense_2=model_prop_dict['dense_2']  
        dense_3=model_prop_dict['dense_3']  
        dense_4=model_prop_dict['dense_4']  
        filters_1=model_prop_dict['filters_1']  
        filters_2=model_prop_dict['filters_2']  
        second_step=model_prop_dict['second_step']  
        second_LSTM=model_prop_dict['second_LSTM']  
        first_conv=model_prop_dict['first_conv']  
        second_conv=model_prop_dict['second_conv']  
        second_conv=model_prop_dict['second_conv']  
        kernel_reg_1=model_prop_dict['kernel_reg_1']  
        kernel_reg_2=model_prop_dict['kernel_reg_2']  
        kernel_reg_3=model_prop_dict['kernel_reg_3']  
        kernel_reg_4=model_prop_dict['kernel_reg_4']  
        bias_reg_1=model_prop_dict['bias_reg_1']  
        bias_reg_2=model_prop_dict['bias_reg_2']  
        bias_reg_3=model_prop_dict['bias_reg_3']  
        bias_reg_4=model_prop_dict['bias_reg_4']  
        activ_reg_1=model_prop_dict['activ_reg_1']  
        activ_reg_2=model_prop_dict['activ_reg_2']  
        activ_reg_3=model_prop_dict['activ_reg_3']  
        activ_reg_4=model_prop_dict['activ_reg_4']  
        optimizer=model_prop_dict['optimizer']  
        grad_clip=model_prop_dict['grad_clip']  
        stateful=model_prop_dict['stateful']  
        
        include_lags=model_prop_dict['include_lags']  
        include_avgs=model_prop_dict['include_avgs']  
        first_dense = model_prop_dict['first_dense']
        # learning_rate=model_prop_dict['learning_rate']
        
        wait_for_files([export_path + dirname + '_real.csv']) 
        glob_res_table = pd.read_csv(export_path + dirname + '_real.csv')
        existing = glob_res_table[glob_res_table.dataset == dirname]
        exists = False
        for existing_dict in existing['model'].values:            
            if model_prop_dict == ast.literal_eval(existing_dict):            
                exists = True
                print('Already done...')
                continue
        if exists:
            time.sleep(3)
            continue 
        
        if include_lags == 1:
            if include_avgs == 1:
                n_features = 107 + 2 + n_steps_in
            else:
                n_features = 11 + 2 + n_steps_in
        else:
            if include_avgs == 1:
                n_features = 107
            else:
                n_features = 11
        
        if optimizer == 1:
            # learning_rate = random.choice([0.0001, 0.001, 0.01])
            try:
                adam = optimizers.Adam(clipnorm=grad_clip,learning_rate= model_prop_dict['learning_rate'] )
            except:
                adam = optimizers.Adam(clipnorm=grad_clip)
        if optimizer == 2:
            # learning_rate = random.choice([0.0001, 0.001, 0.01, 0.1])
            try:
                adam = optimizers.SGD(clipnorm=grad_clip,learning_rate=model_prop_dict['learning_rate'] ,momentum=0.9)
            except:
                adam = optimizers.SGD(clipnorm=grad_clip,momentum=0.9)
        if optimizer == 0:        
            # learning_rate = random.choice([0.0001, 0.001, 0.01])
            try: 
                adam =optimizers.Adam(learning_rate=model_prop_dict['learning_rate'])
            except:
                adam =optimizers.Adam()
        # stateful = random.choice([True, False])
        # if architecture == 'Conv1D':
        #     stateful=False
        po_size = 2
        if n_steps_out == 1:
            po_size = 1
        
        
        model_name = uuid.uuid4().hex        

        
        val_run_res = pd.DataFrame(columns=val_run_cols) 
        # for dirname in dirs:      
            
        dirname = random.choice(dirs)
        
        print('\n')
        print('Pre-testing PLCnet loads architecture ' + dirname)            
        print('\n')         
            
        temp_train_df = pd.read_csv(import_path + dirname + '/energy/train_sum.csv')
        averages = pd.read_csv(import_path + dirname + '/energy/averages_norm.csv')
        
        if include_lags == 1:
            for n in range(n_steps_in):
                temp_train_df.insert(10+n, 'lag_' + str(n+1), [0.0]*len(temp_train_df), True)
            x = temp_train_df['value']
            for n in range(n_steps_in):
                temp_train_df['lag_' + str(n+1)] = np.roll(x,n+1)
            temp_train_df.insert(10, 'lag_1w', np.roll(x,96*7), True)
            temp_train_df.insert(10, 'lag_1d', np.roll(x,96), True)
            temp_train_df = temp_train_df[96*7:]    

        if include_avgs == 1:
            for m in range(96):
                temp_train_df.insert(10+m, str(10+m), [0.0]*len(temp_train_df), True)    
            temp_train_df.loc[temp_train_df['weekend'] == 0, [str(x) for x in list(range(10,106))]] = np.array(averages.weekday.T)
            temp_train_df.loc[temp_train_df['weekend'] == 1, [str(x) for x in list(range(10,106))]] = np.array(averages.weekend.T)  
                   
        X_train, y_train = read_data_plc(temp_train_df, n_steps_in, n_steps_out)
        
        temp_test_df = pd.read_csv(import_path + dirname + '/energy/test_sum.csv')
        averages = pd.read_csv(import_path + dirname + '/energy/averages_norm.csv')
        
        if include_lags == 1:
            for n in range(n_steps_in):
                temp_test_df.insert(10+n, 'lag_' + str(n+1), [0.0]*len(temp_test_df), True)
            x = temp_test_df['value']
            for n in range(n_steps_in):
                temp_test_df['lag_' + str(n+1)] = np.roll(x,n+1)
            temp_test_df.insert(10, 'lag_1w', np.roll(x,96*7), True)
            temp_test_df.insert(10, 'lag_1d', np.roll(x,96), True)
            temp_test_df = temp_test_df[96*7:]    

        if include_avgs == 1:
            for m in range(96):
                temp_test_df.insert(10+m, str(10+m), [0.0]*len(temp_test_df), True)    
            temp_test_df.loc[temp_test_df['weekend'] == 0, [str(x) for x in list(range(10,106))]] = np.array(averages.weekday.T)
            temp_test_df.loc[temp_test_df['weekend'] == 1, [str(x) for x in list(range(10,106))]] = np.array(averages.weekend.T)  
                   
        # X_test, y_test = read_data_plc(temp_test_df, n_steps_in, n_steps_out)
        
        # if stateful: 
        #     train_size = X_train.shape[0]
        #     train_size = train_size-(train_size % 64)
        #     test_size = X_test.shape[0]
        #     test_size = test_size-(test_size % 64)
        #     X_train = X_train[:train_size]
        #     y_train = y_train[:train_size]
        #     X_test = X_test[:test_size]  
        #     y_test = y_test[:test_size] 
        #     bat_size = 64
        

            
        input_conv = keras.Input(shape=(n_steps_in,n_features))  
        
        if first_conv == 1:   
            conv_layer = keras.layers.Conv1D(
                                    kernel_regularizer=kernel_reg_1,
                                    bias_regularizer=bias_reg_1,
                                    activity_regularizer=activ_reg_1,filters=filters_1, kernel_size=1, activation=activation_conv)(input_conv)
            conv_layer = keras.layers.MaxPool1D(pool_size=2)(conv_layer)
            conv_layer = keras.layers.Conv1D(
                                    kernel_regularizer=kernel_reg_2,
                                    bias_regularizer=bias_reg_2,
                                    activity_regularizer=activ_reg_2,filters=filters_2, kernel_size=1, activation=activation_conv)(conv_layer)
            conv_layer = keras.layers.Flatten()(conv_layer)
            if first_dense == 1:
                conv_layer = Dense(dense_1,
                                   kernel_regularizer=kernel_reg_3,
                                   bias_regularizer=bias_reg_3,
                                   activity_regularizer=activ_reg_3,activation=activation_dense)(conv_layer)
        else:
            conv_layer = Dense(dense_2,
                               kernel_regularizer=kernel_reg_1,
                               bias_regularizer=bias_reg_1,
                               activity_regularizer=activ_reg_1,activation=activation_dense)(input_conv)
            conv_layer = keras.layers.Flatten()(conv_layer)
            conv_layer = Dense(dense_3,
                               kernel_regularizer=kernel_reg_2,
                               bias_regularizer=bias_reg_2,
                               activity_regularizer=activ_reg_2,activation=activation_dense)(conv_layer)
            conv_layer = Dropout(dropout_1)(conv_layer)
            
        if stateful: 
            input_LSTM = keras.Input(shape=(n_steps_in, n_features),batch_size=bat_size) 
        else:
            input_LSTM = keras.Input(shape=(n_steps_in, n_features)) 
        LSTM_layer = LSTM(n_n_lstm_1, activation=activation_lstm,stateful=stateful,return_sequences=False)(input_LSTM)  
        model_merge = keras.layers.concatenate([LSTM_layer, conv_layer])
        
        reshape_dim = 0
        if first_conv == 1: 
            if first_dense == 1:
                reshape_dim = dense_1 + n_n_lstm_1   
            else:
                reshape_dim = conv_layer.shape[1] + n_n_lstm_1    
        else:
            reshape_dim = dense_3 + n_n_lstm_1    
        
        model_merge = keras.layers.Reshape((1,reshape_dim))(model_merge)
        
        if second_step == 1:
            if second_conv == 1:
                model_merge = keras.layers.Conv1D(
                    kernel_regularizer=kernel_reg_3,
                    bias_regularizer=bias_reg_3,
                    activity_regularizer=activ_reg_3,filters=filters_3, kernel_size=1, activation=activation_conv)(model_merge)
                model_merge = keras.layers.Flatten()(model_merge)    
            else:
                model_merge = LSTM(n_n_lstm_1, activation=activation_lstm)(model_merge)  
            
        model_merge = Dense(dense_4,
                            kernel_regularizer=kernel_reg_4,
                            bias_regularizer=bias_reg_4,
                            activity_regularizer=activ_reg_4,activation=activation_dense)(model_merge)
        model_merge = Dropout(dropout_2)(model_merge) 
        
        output = Dense(n_steps_out, activation=activation_dense,dtype=tf.float32)(model_merge)
        model = Model(inputs=[input_LSTM, input_conv], outputs=output) 
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=n_epoch//8,restore_best_weights=True)
        
        
        train_generator = PLCgenerator(temp_train_df, n_steps_in, n_steps_out, batch_size=bat_size,stateful=stateful)
        test_generator = PLCgenerator(temp_test_df, n_steps_in, n_steps_out, batch_size=bat_size,stateful=stateful)
        # model.fit(generator, epochs=10)
        
        if stateful:
            history = model.fit(train_generator, epochs=n_epoch,batch_size=bat_size, shuffle=False,callbacks=[callback])   
            train_loss = pd.DataFrame(history.history['loss'])
            temp = model.predict(test_generator)
            
        else:
            history = model.fit(train_generator, epochs=n_epoch, shuffle=False,callbacks=[callback])
            train_loss = pd.DataFrame(history.history['loss'])
            temp = model.predict(test_generator)           
            
        filename = export_path + 'models/' + str(model_name) + '.sav'
        pickle.dump(model, open(filename, 'wb'))
        train_loss.to_csv(export_path + 'models/' + str(model_name) + '.csv')
        temp = np.squeeze(temp)
        
        reset_tensorflow_keras_backend()
        
        m,n=temp.shape 
        t_target = n_steps_out
           
        yhat=np.zeros((m,t_target))
        y_obs = test_generator.get_data()
        # y_obs=np.array(y_test[0:m,0:t_target])
        scores= np.zeros(m)        

        try:
            rmses = np.zeros(m)
            maes = np.zeros(m)
            r2s = np.zeros(m)
            for j in np.arange(m):                        
                rmse = mean_squared_error(y_obs[j,:], temp[j,:], squared=False)
                mae = mean_absolute_error(y_obs[j,:], temp[j,:])
                _r_squared = r2_score(y_obs[j,:], temp[j,:])
                rmses[j] = 1.0 - rmse
                maes[j] = mae
                r2s[j] = _r_squared
            
            perf = np.mean(rmses)      

            
            
            
            
            row = pd.DataFrame(columns=glob_res_cols)            
            row.id = [model_name]
            row.dataset = [dirname]
            row.model = [model_prop_dict]            
            row.rmse = perf
            row.mae = np.mean(maes)
            row.r_squared = r2_score(y_obs[:,0], temp[:,0])
            
            wait_for_files([export_path + dirname + '_real.csv'])             
            glob_res_table = pd.read_csv(export_path + dirname + '_real.csv')
            glob_res_table=pd.concat([glob_res_table,row])
            glob_res_table.to_csv(export_path + dirname + '_real.csv', encoding='utf-8',index=False)
        except Exception:
            pass
        

 
for _ in range(100000):
    dirs = ['acn_caltech','acn_jpl','palo_alto','boulder']
    for dirname in dirs:

        architectures = ['LSTM', 'GRU', 'BiLSTM', 'Stacked', 'Conv1D', 'CNN_LSTM', 'CNN_BiLSTM', 'CNN_GRU' , 'ConvLSTM']
        architectures = ['CNN_LSTM', 'ConvLSTM','CNN_BiLSTM', 'CNN_GRU','Conv1D']
        problem_architectures = ['CNN_LSTM', 'ConvLSTM','CNN_BiLSTM', 'CNN_GRU','Conv1D']
        architecture = random.choice(architectures)
        
        include_lags = random.choice([0,1])    
        include_avgs = random.choice([0,1])    
        include_lags = 1
        # include_avgs = 0
        n_features = 107
        n_steps_in = random.choice([4,8,16,32,64])
        # n_steps_in = 16
        n_steps_out = random.choice([4,16,96])
        if dirname == 'acn_caltech' and n_steps_out == 16:
            architecture = 'Stacked'
            
        # n_steps_out = 96
        # if processor == 'GPU':
        #     n_steps_out = random.choice([4,16])
        # else:
        #     n_steps_out = 96
        # n_steps_out = 96
        
        
        if n_steps_in > n_steps_out:
            n_steps_in = n_steps_out  
        if n_steps_out == 96 and n_steps_in > 8:
            n_steps_in = 8        
        if include_avgs == 1:
            if n_steps_out == 96:            
                    n_steps_in = 2
            if n_steps_out == 16 and n_steps_in == 16:
                    n_steps_in = 8          
        

                
                    
        # if n_steps_out == 4:
        #     if architecture == 'ConvLSTM':
        #         if dirname in ['acn_caltech','acn_jpl']:
        #             continue
        # if n_steps_out == 16:            
        #     if architecture == 'ConvLSTM':
        #         if dirname in ['acn_caltech','palo_alto','boulder']:
        #             continue
        #     if architecture in ['CNN_BiLSTM', 'CNN_GRU','CNN_LSTM']:
        #         if dirname in ['acn_caltech','acn_jpl']:
        #             continue
        # if n_steps_out == 96:
        #     if architecture == 'Conv1D':
        #         if dirname in ['acn_caltech','acn_jpl']:
        #             continue
        #     if architecture in ['CNN_BiLSTM', 'CNN_GRU','CNN_LSTM']:
        #         if dirname in ['acn_jpl']:
        #             continue
            
        if include_lags == 1:
            if include_avgs == 1:
                n_features = 107 + 2 + n_steps_in
            else:
                n_features = 11 + 2 + n_steps_in
        else:
            if include_avgs == 1:
                n_features = 107
            else:
                n_features = 11
                
                
        po_size = 2
        if n_steps_out == 1:
            po_size = 1
        nf_1 = random.choice([2, 4, 8, 16, 32])
        nf_2 = random.choice([2, 4, 8, 16, 32])
        ker_size = 4
        
        # stacked = random.randint(0,1)
        stacked = random.choice([0,1,2])
        # stacked = 2
        stack_size = random.choice([2,3,4])
        nodes_1 = random.choice([8, 16, 32, 64, 128, 256])
        nodes_2 = random.choice([4, 8, 16, 32, 64, 128])
        nodes_3 = random.choice([4, 8, 16, 32, 64, 128])
        nodes_4 = random.choice([4, 8, 16, 32, 64, 128])
        
        activation_rec_1 = random.choice(['sigmoid', 'tanh', 'relu'])
        activation_rec_2 = random.choice(['sigmoid', 'tanh', 'relu'])
        activation_rec_3 = random.choice(['sigmoid', 'tanh', 'relu'])
        activation_rec_4 = random.choice(['sigmoid', 'tanh', 'relu'])
        
        dense_1 = random.choice([4, 8, 16, 32, 64, 128])
        activation = random.randint(0,1)
        dropout = random.randint(1,60)/100.0            
        # epochs = 2
        # batch_size = random.randint(64,256)
        # architecture = random.choice(architectures) 
        # architecture = 'Stacked'
        # architecture = 'ConvLSTM'
        # if architecture in problem_architectures:
        #     if n_steps_out == 96:
        #         n_steps_in = 2
        #     if n_steps_out == 16:
        #         if n_steps_in > n_steps_out:
        #             n_steps_in = 16
        # architecture = 'LSTM'
        bat_size = random.choice([4, 8, 16, 32, 64, 128]) 
        n_epoch = random.choice([16, 32, 64, 128])    
        # n_epoch = 1
        #n_epoch = 1
        activation_1 = random.choice(['sigmoid', 'tanh', 'relu'])
        activation_2 = random.choice(['sigmoid', 'tanh', 'relu'])
        if architecture == 'Conv1D' or architecture == 'ConvLSTM':            
            ker_size=1   
            
        kernel_reg_1 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_2 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_3 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        bias_reg_1 = random.choice([None, None, None, None, None])
        bias_reg_2 = random.choice([None, None, None, None, None])
        bias_reg_3 = random.choice([None, None, None, None, None])
        activ_reg_1 = random.choice([None, None, None, None, None])
        activ_reg_2 = random.choice([None, None, None, None, None])
        activ_reg_3 = random.choice([None, None, None, None, None])
        
        grad_clip = random.choice([1.1, 2.0, 4.0, 8.0])        
        optimizer = random.choice([0,1,2])
        # optimizer = random.choice([1,2])
        if optimizer == 1:
            learning_rate = random.choice([0.0001, 0.001, 0.01])
            adam = optimizers.Adam(clipnorm=grad_clip,learning_rate=learning_rate)
        if optimizer == 2:
            learning_rate = random.choice([0.0001, 0.001, 0.01, 0.1])
            adam = optimizers.SGD(clipnorm=grad_clip,learning_rate=learning_rate,momentum=0.9)
        if optimizer == 0:        
            learning_rate = random.choice([0.0001, 0.001, 0.01])
            adam =optimizers.Adam(learning_rate=learning_rate)
        stateful = random.choice([True, False])          
        # stateful = True
        
        
        model_prop_dict = {}
        model_prop_dict['architecture'] = architecture
        model_prop_dict['n_features'] = n_features
        model_prop_dict['n_steps_in'] = n_steps_in
        model_prop_dict['n_steps_out'] = n_steps_out
        model_prop_dict['po_size'] = po_size
        model_prop_dict['nf_1'] = nf_1
        model_prop_dict['nf_2'] = nf_2
        model_prop_dict['ker_size'] = ker_size
        model_prop_dict['stacked'] = stacked
        model_prop_dict['stack_size'] = stack_size
        model_prop_dict['nodes_1'] = nodes_1
        model_prop_dict['nodes_2'] = nodes_2
        model_prop_dict['nodes_3'] = nodes_3
        model_prop_dict['nodes_4'] = nodes_4
        model_prop_dict['activation_rec_1'] = activation_rec_1
        model_prop_dict['activation_rec_2'] = activation_rec_2
        model_prop_dict['activation_rec_3'] = activation_rec_3
        model_prop_dict['activation_rec_4'] = activation_rec_4
        model_prop_dict['dense_1'] = dense_1
        model_prop_dict['dropout'] = dropout
        model_prop_dict['bat_size'] = bat_size
        model_prop_dict['n_epoch'] = n_epoch
        model_prop_dict['activation_1'] = activation_1
        model_prop_dict['activation_2'] = activation_2
        model_prop_dict['kernel_reg_1'] = kernel_reg_1
        model_prop_dict['kernel_reg_2'] = kernel_reg_2
        model_prop_dict['kernel_reg_3'] = kernel_reg_3
        model_prop_dict['bias_reg_1'] = bias_reg_1
        model_prop_dict['bias_reg_2'] = bias_reg_2
        model_prop_dict['bias_reg_3'] = bias_reg_3
        model_prop_dict['activ_reg_1'] = activ_reg_1
        model_prop_dict['activ_reg_2'] = activ_reg_2
        model_prop_dict['activ_reg_3'] = activ_reg_3
        model_prop_dict['optimizer'] = optimizer
        model_prop_dict['grad_clip'] = grad_clip
        model_prop_dict['stateful'] = stateful
        model_prop_dict['include_lags'] = include_lags
        model_prop_dict['include_avgs'] = include_avgs
        model_prop_dict['learning_rate'] = learning_rate
        
        non_hybrid_loads(model_prop_dict,dirname)

sys.exit()
 
                
for _ in range(100):
    dirs = ['acn_caltech', 'acn_jpl', 'boulder', 'palo_alto']
    for dirname in dirs:
    
        include_lags = random.choice([0,1]) 
        include_lags = 1         
        n_steps_in = random.choice([4,8,16,32,64]) 
        
        
        
        n_steps_out = random.choice([4,16,96])
        
        if processor == 'GPU':
            n_steps_out = random.choice([4,16])
        else:
            n_steps_out = 96
        # n_steps_out = 96
        if n_steps_in > n_steps_out:
            n_steps_in = n_steps_out
            
        n_features = 1
        if include_lags == 1:
            n_features = 1 + 2 + n_steps_in        
        
        dropout_1 = random.randint(1,60)/100.0 
        dropout_2 = random.randint(1,60)/100.0 
        n_n_lstm_1 = random.choice([8, 16, 32, 64, 128])
        n_n_lstm_2 = random.choice([8, 16, 32, 64, 128])
        n_n_lstm_3 = random.choice([8, 16, 32, 64, 128])
        n_n_lstm_4 = random.choice([8, 16, 32, 64, 128])
        bat_size = random.randint(4,512)    
        bat_size = random.choice([8, 16, 32, 64, 128])
        n_epoch = random.choice([32, 64, 128, 256])
        
        
        dense_1 = random.choice([4, 8, 16, 32, 64, 128])
        dense_2 = random.choice([4, 8, 16, 32, 64, 128])
        dense_3 = random.choice([4, 8, 16, 32, 64, 128])
        dense_4 = random.choice([4, 8, 16, 32, 64, 128])
        dense_5 = random.choice([4, 8, 16, 32, 64, 128])
        dense_6 = random.choice([4, 8, 16, 32, 64, 128])
        dense_7 = random.choice([4, 8, 16, 32, 64, 128])
        model_name = uuid.uuid4().hex
        
        filters_1 = random.choice([2, 4, 8, 16, 32])
        filters_2 = random.choice([2, 4, 8, 16, 32])
        
        kernel_reg_1 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_2 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_3 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_4 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_5 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_6 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        bias_reg_1 = random.choice([None, None, None, None, None])
        bias_reg_2 = random.choice([None, None, None, None, None])
        bias_reg_3 = random.choice([None, None, None, None, None])
        bias_reg_4 = random.choice([None, None, None, None, None])
        bias_reg_5 = random.choice([None, None, None, None, None])
        bias_reg_6 = random.choice([None, None, None, None, None])
        activ_reg_1 = random.choice([None, None, None, None, None])        
        activ_reg_2 = random.choice([None, None, None, None, None])
        activ_reg_3 = random.choice([None, None, None, None, None])
        activ_reg_4 = random.choice([None, None, None, None, None])
        activ_reg_5 = random.choice([None, None, None, None, None])      
        activ_reg_6 = random.choice([None, None, None, None, None])       
        
        activation_dense = random.choice(['sigmoid', 'tanh', 'relu'])
        activation_conv = random.choice(['sigmoid', 'tanh', 'relu'])
        activation_lstm = random.choice(['sigmoid', 'tanh', 'relu'])
        convolve = random.choice([0, 1])
        stack_layers = random.choice([1, 2, 3])
        
        
        grad_clip = random.choice([1.1, 2.0, 4.0, 8.0])        
        optimizer = random.choice([0,1,2])
        # optimizer = random.choice([1,2])
        if optimizer == 1:
            learning_rate = random.choice([0.0001, 0.001, 0.01])
            adam = optimizers.Adam(clipnorm=grad_clip,learning_rate=learning_rate)
        if optimizer == 2:
            learning_rate = random.choice([0.0001, 0.001, 0.01, 0.1])
            adam = optimizers.SGD(clipnorm=grad_clip,learning_rate=learning_rate)
        if optimizer == 0:        
            learning_rate = random.choice([0.0001, 0.001, 0.01])
            adam =optimizers.Adam(learning_rate=learning_rate)
        stateful = random.choice([True, False])
        
        model_prop_dict = {}
        # model_prop_dict = dict.fromkeys(model_properties)
        model_prop_dict['n_features'] = n_features
        model_prop_dict['n_steps_in'] = n_steps_in
        model_prop_dict['n_steps_out'] = n_steps_out
        model_prop_dict['dropout_1'] = dropout_1
        model_prop_dict['dropout_2'] = dropout_2
        model_prop_dict['n_n_lstm_1'] = n_n_lstm_1
        model_prop_dict['n_n_lstm_2'] = n_n_lstm_2
        model_prop_dict['n_n_lstm_3'] = n_n_lstm_3
        model_prop_dict['n_n_lstm_4'] = n_n_lstm_4
        model_prop_dict['bat_size'] = bat_size
        model_prop_dict['n_epoch'] = 2
        model_prop_dict['dense_1'] = dense_1
        model_prop_dict['dense_2'] = dense_2
        model_prop_dict['dense_3'] = dense_3
        model_prop_dict['dense_4'] = dense_4
        model_prop_dict['dense_5'] = dense_5
        model_prop_dict['dense_6'] = dense_6
        model_prop_dict['dense_7'] = dense_7
        model_prop_dict['filters_1'] = filters_1
        model_prop_dict['filters_2'] = filters_2
        model_prop_dict['activation_dense'] = activation_dense
        model_prop_dict['activation_conv'] = activation_conv
        model_prop_dict['activation_lstm'] = activation_lstm
        model_prop_dict['convolve'] = convolve
        model_prop_dict['stack_layers'] = stack_layers
        model_prop_dict['kernel_reg_1'] = kernel_reg_1
        model_prop_dict['kernel_reg_2'] = kernel_reg_2
        model_prop_dict['kernel_reg_3'] = kernel_reg_3
        model_prop_dict['kernel_reg_4'] = kernel_reg_4
        model_prop_dict['kernel_reg_5'] = kernel_reg_5
        model_prop_dict['kernel_reg_6'] = kernel_reg_6
        model_prop_dict['bias_reg_1'] = bias_reg_1
        model_prop_dict['bias_reg_2'] = bias_reg_2
        model_prop_dict['bias_reg_3'] = bias_reg_3
        model_prop_dict['bias_reg_4'] = bias_reg_4
        model_prop_dict['bias_reg_5'] = bias_reg_5
        model_prop_dict['bias_reg_6'] = bias_reg_6
        model_prop_dict['activ_reg_1'] = activ_reg_1
        model_prop_dict['activ_reg_2'] = activ_reg_2
        model_prop_dict['activ_reg_3'] = activ_reg_3
        model_prop_dict['activ_reg_4'] = activ_reg_4
        model_prop_dict['activ_reg_5'] = activ_reg_5
        model_prop_dict['activ_reg_6'] = activ_reg_6
        model_prop_dict['optimizer'] = optimizer
        model_prop_dict['grad_clip'] = grad_clip
        model_prop_dict['stateful'] = stateful
        model_prop_dict['include_lags'] = include_lags  
        model_prop_dict['learning_rate'] = learning_rate
    
    
    
    
        hybrid_loads(model_prop_dict, dirname)


sys.exit()


PLC = True
if PLC:
    for i in range(2000):
    
        include_lags = random.choice([0,1])    
        include_avgs = random.choice([0,1])    
        include_lags = 1
        include_avgs = 1
        n_steps_in = random.choice([4,8,16,32,64])
        
        # n_steps_out = random.choice([4,16,96])
        n_steps_out = random.choice([4,16])
        
        # if processor == 'GPU':
        #     n_steps_out = random.choice([4,16])
        # else:
        #     n_steps_out = 96
        
        # n_steps_out = 96
        if n_steps_in > n_steps_out:
            n_steps_in = n_steps_out   
            
        if include_lags == 1:
            if include_avgs == 1:
                n_features = 107 + 2 + n_steps_in
            else:
                n_features = 11 + 2 + n_steps_in
        else:
            if include_avgs == 1:
                n_features = 107
            else:
                n_features = 11
        
        n_n_lstm_1 = random.choice([32, 64, 128])
        n_n_lstm_2 = random.choice([64, 128, 256])
        bat_size = random.choice([8, 16, 32, 64, 128]) 
        n_epoch = random.choice([32, 64, 128])
        n_epoch = 1
        
        activation_dense = random.choice(['sigmoid', 'tanh', 'relu'])
        activation_conv = random.choice(['sigmoid', 'tanh', 'relu'])
        activation_lstm = random.choice(['sigmoid', 'tanh', 'relu'])
        
        filters_1 = random.choice([4, 8, 16, 32, 64])
        filters_2 = random.choice([4, 8, 16, 32, 64])
        filters_3 = random.choice([4, 8, 16, 32, 64])
        
        dropout_1 = random.randint(1,60)/100.0 
        dropout_2 = random.randint(1,60)/100.0 
        
        dense_1 = random.choice([32, 64, 128])
        dense_2 = random.choice([32, 64, 128])
        dense_3 = random.choice([32, 64, 128])
        dense_4 = random.choice([32, 64, 128])
        
        kernel_reg_1 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_2 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_3 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        kernel_reg_4 = random.choice([None, None, None, None, None, 'l1', 'l2'])
        bias_reg_1 = random.choice([None, None, None, None, None])
        bias_reg_2 = random.choice([None, None, None, None, None])
        bias_reg_3 = random.choice([None, None, None, None, None])
        bias_reg_4 = random.choice([None, None, None, None, None])
        activ_reg_1 = random.choice([None, None, None, None, None])
        activ_reg_2 = random.choice([None, None, None, None, None])
        activ_reg_3 = random.choice([None, None, None, None, None])
        activ_reg_4 = random.choice([None, None, None, None, None])        
        second_step = random.randint(0,1)
        second_LSTM = random.randint(0,1)
        first_conv = random.randint(0,1)
        second_conv = random.randint(0,1)
        first_dense = random.randint(0,1)
        
        grad_clip = random.choice([1.1, 2.0, 4.0, 8.0])        
        optimizer = random.choice([0,1,2])
        if optimizer == 1:
            learning_rate = random.choice([0.0001, 0.001, 0.01])
            adam = optimizers.Adam(clipnorm=grad_clip,learning_rate=learning_rate)
        if optimizer == 2:
            learning_rate = random.choice([0.0001, 0.001, 0.01, 0.1])
            adam = optimizers.SGD(clipnorm=grad_clip,learning_rate=learning_rate)
        if optimizer == 0:        
            learning_rate = random.choice([0.0001, 0.001, 0.01])
            adam =optimizers.Adam(learning_rate=learning_rate)
        stateful = random.choice([True, False])
        stateful = True
        
        
        model_prop_dict = {}
        model_prop_dict['n_features'] = n_features
        model_prop_dict['n_steps_in'] = n_steps_in
        model_prop_dict['n_steps_out'] = n_steps_out
        model_prop_dict['n_n_lstm_1'] = n_n_lstm_1
        model_prop_dict['n_n_lstm_2'] = n_n_lstm_2
        model_prop_dict['bat_size'] = bat_size
        model_prop_dict['n_epoch'] = n_epoch
        model_prop_dict['activation_dense'] = activation_dense
        model_prop_dict['activation_conv'] = activation_conv
        model_prop_dict['activation_lstm'] = activation_lstm
        model_prop_dict['filters_1'] = filters_1
        model_prop_dict['filters_2'] = filters_2
        model_prop_dict['filters_3'] = filters_3
        model_prop_dict['dropout_1'] = dropout_1
        model_prop_dict['dropout_2'] = dropout_2
        model_prop_dict['dense_1'] = dense_1
        model_prop_dict['dense_2'] = dense_2
        model_prop_dict['dense_3'] = dense_3
        model_prop_dict['dense_4'] = dense_4
        model_prop_dict['filters_1'] = filters_1
        model_prop_dict['filters_2'] = filters_2
        model_prop_dict['second_step'] = second_step
        model_prop_dict['first_dense'] = first_dense
        model_prop_dict['second_LSTM'] = second_LSTM
        model_prop_dict['first_conv'] = first_conv
        model_prop_dict['second_conv'] = second_conv
        model_prop_dict['second_conv'] = second_conv
        model_prop_dict['kernel_reg_1'] = kernel_reg_1
        model_prop_dict['kernel_reg_2'] = kernel_reg_2
        model_prop_dict['kernel_reg_3'] = kernel_reg_3
        model_prop_dict['kernel_reg_4'] = kernel_reg_4
        model_prop_dict['bias_reg_1'] = bias_reg_1
        model_prop_dict['bias_reg_2'] = bias_reg_2
        model_prop_dict['bias_reg_3'] = bias_reg_3
        model_prop_dict['bias_reg_4'] = bias_reg_4
        model_prop_dict['activ_reg_1'] = activ_reg_1
        model_prop_dict['activ_reg_2'] = activ_reg_2
        model_prop_dict['activ_reg_3'] = activ_reg_3
        model_prop_dict['activ_reg_4'] = activ_reg_4
        model_prop_dict['optimizer'] = optimizer
        model_prop_dict['grad_clip'] = grad_clip
        model_prop_dict['stateful'] = stateful
        model_prop_dict['include_lags'] = include_lags
        model_prop_dict['include_avgs'] = include_avgs
        model_prop_dict['learning_rate'] = learning_rate
        
        for dirname in dirs:
            plc_loads(model_prop_dict, dirname)



# dirs = ['acn_caltech']               
for dirname in dirs:         
    
        
            
    plc1_done = True
    plc4_done = True
    plc24_done = True
    
    h1_done = True
    h4_done = True
    h24_done = True
    
    lstm1_done = True
    lstm4_done = True
    lstm24_done = True
    
    gru1_done = True
    gru4_done = True
    gru24_done = True
    
    bilstm1_done = True
    bilstm4_done = True
    bilstm24_done = True
    
    stacked1_done = True
    stacked4_done = True
    stacked24_done = True
    
    convlstm1_done = False
    convlstm4_done = False
    convlstm24_done = False
    
    cnnlstm1_done = False
    cnnlstm4_done = False
    cnnlstm24_done = False
    
    conv1_done = False
    conv4_done = False
    conv24_done = False    
    
    
    # plc1_done = False
    # plc4_done = False
    # plc24_done = False
    
    model_mode = None
        
    final_res_path = "/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/results_final/"
    filenames = os.listdir(final_res_path)
       
    for filename in filenames:
        if 'load' in filename:
            wait_for_files([final_res_path + filename]) 
            results = pd.read_csv(final_res_path + filename)  
            models = results.model
            for i in range(len(models)):
                try:
                    model_dict = ast.literal_eval(models[i])
                    # if model_dict['n_steps_out'] > 4:
                    #     continue
                except Exception as e:
                    print(e)
                    continue
                try:
                    a = model_dict['stateful']
                except:
                    model_dict['stateful'] = False
                try:
                    a = model_dict['include_lags']
                except:
                    model_dict['include_lags'] = 0
                try:
                    a = model_dict['include_avgs']
                except:
                    model_dict['include_avgs'] = 0
                try:
                    a = model_dict['stack_size']
                except:
                    model_dict['stack_size'] = 1
                try:
                    a = model_dict['nodes_2']  
                except:
                    model_dict['nodes_2'] = 0
                try:
                    a = model_dict['nodes_3']  
                except:
                    model_dict['nodes_3'] = 0
                try:
                    a = model_dict['nodes_4']  
                except:
                    model_dict['nodes_4'] = 0
                try:
                    a = model_dict['activation_rec_1']  
                except:
                    model_dict['activation_rec_1'] = 0
                try:
                    a = model_dict['activation_rec_2']  
                except:
                    model_dict['activation_rec_2'] = 0
                try:
                    a = model_dict['activation_rec_3']  
                except:
                    model_dict['activation_rec_3'] = 0
                try:
                    a = model_dict['activation_rec_4']  
                except:
                    model_dict['activation_rec_4'] = 0
                try:
                    a = model_dict['optimizer']  
                except:
                    model_dict['optimizer'] = 0
                try:
                    a = model_dict['grad_clip']  
                except:
                    model_dict['grad_clip'] = None
                    
                if model_dict['include_lags'] == 1:
                    if model_dict['include_avgs'] == 1:
                        model_dict['n_features'] = 107 + 2 + model_dict['n_steps_in']
                    else:
                        model_dict['n_features'] = 11 + 2 + model_dict['n_steps_in']
                else:
                    if model_dict['include_avgs'] == 1:
                        model_dict['n_features'] = 107
                    else:
                        model_dict['n_features'] = 11    
                    
                if "convolve" in model_dict:
                    
                    # hybrid_loads(model_dict,dirname)  
                    # continue
                    if model_dict['n_steps_out'] == 4:  
                        if not h1_done:
                            hybrid_loads(model_dict, dirname)       
                            h1_done = True
                    if model_dict['n_steps_out'] == 16:  
                        if not h4_done:
                            hybrid_loads(model_dict, dirname) 
                            h4_done = True
                    if model_dict['n_steps_out'] == 96:  
                        if not h24_done:
                            hybrid_loads(model_dict, dirname) 
                            h24_done = True
                if 'first_dense' in model_dict:
                    # plc_loads(model_dict,dirname)
                    # continue
                    if model_dict['n_steps_out'] == 4:
                        if not plc1_done:
                            for i in range(2):
                                model_dict['first_dense'] = i
                                plc_loads(model_dict, dirname)
                                plc1_done = True
                    if model_dict['n_steps_out'] == 16:
                        if not plc4_done:
                            for i in range(2):
                                model_dict['first_dense'] = i
                                plc_loads(model_dict, dirname)
                                plc4_done = True
                    if model_dict['n_steps_out'] == 96:
                        if not plc24_done:
                            for i in range(2):
                                model_dict['first_dense'] = i
                                plc_loads(model_dict, dirname)
                                plc24_done = True
                if 'architecture' in model_dict:
                    # non_hybrid_loads(model_dict,dirname) 
                    if model_dict['architecture'] == 'LSTM':
                        if model_dict['n_steps_out'] == 4:  
                            if not lstm1_done:
                                non_hybrid_loads(model_dict, dirname)       
                                lstm1_done = True
                        if model_dict['n_steps_out'] == 16:  
                            if not lstm4_done:
                                non_hybrid_loads(model_dict, dirname)       
                                lstm4_done = True
                        if model_dict['n_steps_out'] == 96:  
                            if not lstm24_done:
                                non_hybrid_loads(model_dict, dirname)       
                                lstm24_done = True
                    if model_dict['architecture'] == 'GRU':
                        if model_dict['n_steps_out'] == 4:  
                            if not gru1_done:
                                non_hybrid_loads(model_dict, dirname)       
                                gru1_done = True
                        if model_dict['n_steps_out'] == 16:  
                            if not gru4_done:
                                non_hybrid_loads(model_dict, dirname)       
                                gru4_done = True
                        if model_dict['n_steps_out'] == 96:  
                            if not gru24_done:
                                non_hybrid_loads(model_dict, dirname)       
                                gru24_done = True
                    if model_dict['architecture'] == 'BiLSTM':
                        if model_dict['n_steps_out'] == 4:  
                            if not bilstm1_done:
                                non_hybrid_loads(model_dict, dirname)       
                                bilstm1_done = True
                        if model_dict['n_steps_out'] == 16:  
                            if not bilstm4_done:
                                non_hybrid_loads(model_dict, dirname)       
                                bilstm4_done = True
                        if model_dict['n_steps_out'] == 96:  
                            if not bilstm24_done:
                                non_hybrid_loads(model_dict, dirname)       
                                bilstm24_done = True
                    if model_dict['architecture'] == 'Stacked':
                        if model_dict['n_steps_out'] == 4:  
                            if not stacked1_done:
                                non_hybrid_loads(model_dict, dirname)       
                                stacked1_done = True
                        if model_dict['n_steps_out'] == 16:  
                            if not stacked4_done:
                                non_hybrid_loads(model_dict, dirname)       
                                stacked4_done = True
                        if model_dict['n_steps_out'] == 96:  
                            if not stacked24_done:
                                non_hybrid_loads(model_dict, dirname)       
                                stacked24_done = True
                    if model_dict['architecture'] == 'Conv1D':
                        if model_dict['n_steps_out'] == 4:  
                            if not conv1_done:
                                non_hybrid_loads(model_dict, dirname)       
                                conv1_done = True
                        if model_dict['n_steps_out'] == 16:  
                            if not conv4_done:
                                non_hybrid_loads(model_dict, dirname)       
                                conv4_done = True
                        if model_dict['n_steps_out'] == 96:  
                            if not conv24_done:
                                non_hybrid_loads(model_dict, dirname)       
                                conv24_done = True
                    if model_dict['architecture'] in ['CNN_LSTM','CNN_BiLSTM','CNN_GRU']:
                        if model_dict['n_steps_out'] == 4:  
                            if not cnnlstm1_done:
                                non_hybrid_loads(model_dict, dirname)       
                                cnnlstm1_done = True
                        if model_dict['n_steps_out'] == 16:  
                            if not cnnlstm4_done:
                                non_hybrid_loads(model_dict, dirname)       
                                cnnlstm4_done = True
                        if model_dict['n_steps_out'] == 96:  
                            if not cnnlstm24_done:
                                non_hybrid_loads(model_dict, dirname)       
                                cnnlstm24_done = True
                    if model_dict['architecture'] == 'ConvLSTM': 
                        if model_dict['n_steps_out'] == 4:  
                            if not convlstm1_done:
                                non_hybrid_loads(model_dict, dirname)       
                                convlstm1_done = True
                        if model_dict['n_steps_out'] == 16:  
                            if not convlstm4_done:
                                non_hybrid_loads(model_dict, dirname)       
                                convlstm4_done = True
                        if model_dict['n_steps_out'] == 96:  
                            if not convlstm24_done:
                                non_hybrid_loads(model_dict, dirname)       
                                convlstm24_done = True
                    
                        
                        
           