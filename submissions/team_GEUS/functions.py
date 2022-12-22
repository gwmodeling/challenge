import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow import keras
import random
import os

def q01(s, o):
    """
    1 % percentile difference Q01obs - Q01sim 
    input:
        s: simulated
        o: observed
    output:
        q01:  1 % percentile difference 
    """
    s,o = _filter_nan(s,o)
    return (np.percentile(o,1)-np.percentile(s,1))

def q99(s, o):
    """
    99 % percentile difference Q99obs - Q99sim
    input:
        s: simulated
        o: observed
    output:
        q99:  99 % percentile difference
    """
    s,o = _filter_nan(s,o)
    return (np.percentile(o,99)-np.percentile(s,99))
   
def corr(s,o):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    s,o = _filter_nan(s,o)
    if s.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(o, s)[0,1]
        
    return corr

def _filter_nan(s,o):
    """
    this functions removes data from simulated and observed data
    whereever the observed or simulated data contains a NaN
    """
    if np.sum(~np.isnan(s*o))>=1:
        data = np.array([s.flatten(),o.flatten()])
        data = np.transpose(data)
        data = data[~np.isnan(data).any(1)]
        s = data[:,0]
        o = data[:,1]
    return s, o

def kge(s, o):
    """
    Kling-Gupta Efficiency
    input:
        s: simulated
        o: observed
    output:
        kge: Kling-Gupta Efficiency
        cc: correlation 
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    s,o = _filter_nan(s,o)
    cc = corr(s,o)
    alpha = np.std(s)/np.std(o)
    beta = np.sum(s)/np.sum(o)
    kge = 1- np.sqrt( (cc-1)**2 + (alpha-1)**2 + (beta-1)**2 )
    return kge, cc, alpha, beta

def split_sequences(data, n_steps):

    X, y = list(), list()

    for i in range(len(data)-n_steps):
        #find the end of this pattern
        end_ix = i + n_steps
        out_end_ix = end_ix-1
        # check if we are beyond the dataset
        if out_end_ix > len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_ix, 1:], data[out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
        m_1=~np.isnan(X).any(axis=(1,2))
        m_2=~np.isnan(y)
        m=m_1 & m_2
    return np.array(X)[m,:,:], np.array(y)[m]

def quantile_loss(q, y, y_p):
    
    e = y-y_p
    return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))

def train_lstm_model(n_steps, n_features, epochs, batch_size, n_cells, dropout, recurrent_dropout, learning_rate,X_train,y_train,X_test,y_test,q):
    
    #initialize
    model = Sequential()
    model.add(LSTM(n_cells,activation='tanh',recurrent_activation='sigmoid', dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_initializer='zeros', bias_initializer='zeros',  input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    if np.isfinite(q):
        model.compile(optimizer=opt, loss=lambda y, y_p: quantile_loss(q, y, y_p))
    else:
        model.compile(optimizer=opt, loss='mean_squared_error')
    
    # early stopping
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=10, restore_best_weights=True)
    
    # train
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=es, verbose=0)
    
    return model

def norm(d,avg,std):
    return np.divide(np.subtract(d.values,avg),std)

def rev_norm(d,avg,std):
    d=d*std[0]+avg[0]
    d[d<0]=0
    return d



    
    
    
