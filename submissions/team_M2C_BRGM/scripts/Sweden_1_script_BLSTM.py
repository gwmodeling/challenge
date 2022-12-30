#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 17:30:05 2022

@author: chidesiv
"""
modtype="BILSTM" #GRU, LSTM,BILSTM
Well="Sweden_1" #Germany,Netherlands, Sweden_1,Sweden_2,USA
 #8,19,57
seq_length=48
initm=30
test_size=0.2


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
from tensorflow import device
from tensorflow.random import set_seed
from numpy.random import seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
import optuna
from optuna.samplers import TPESampler
from uncertainties import unumpy





gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*6)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
    
def Scaling_data(X_train,X_valid,y_train):
    """
    Scaling function to fit on training data and transform on test data

    Parameters
    ----------
    X_train : TYPE
        Original training data of input variables.
    X_valid : TYPE
        Original testing data.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train_s : TYPE
        scaled training input variables
    X_valid_s : TYPE
        scaled test input variables
    y_train_s : TYPE
        scaled training target variable.
    X_C_scaled : TYPE
        combined scaled training and test input variables

    """
    Scaler=MinMaxScaler(feature_range=(0,1))
    X_train_s=Scaler.fit_transform(X_train)
    X_valid_s=Scaler.transform(X_valid)
    target_scaler = MinMaxScaler(feature_range=(0,1))
    target_scaler.fit(np.array(y_train).reshape(-1,1))
    y_train_s=target_scaler.transform(np.array(y_train).reshape(-1,1))
    
    X_C_scaled=np.concatenate((X_train_s,X_valid_s),axis=0)
   
    return X_train_s,X_valid_s,y_train_s,X_C_scaled
 
    
data_input =pd.read_csv(r"/home/chidesiv/Desktop/Scripts_py/Modeling_challenge/challenge-main/data/"+str(Well)+"/"+"input_data.csv")

# create a time column with datetime format
if Well=="USA":
    data_input['time'] = pd.to_datetime(data_input['Unnamed: 0'])#USA
else:
    data_input['time'] = pd.to_datetime(data_input['time'])#Germany,Netherlands, Sweden_1,Sweden_2

#import target variable 
#Please change the file path in the below lines before running this script
target=pd.read_csv(r"/home/chidesiv/Desktop/Scripts_py/Modeling_challenge/challenge-main/data/"+str(Well)+"/"+"heads.csv")

# create date column of target variables in datetimeformat
if Well=="Netherlands":
    target['Date']=pd.to_datetime(target['Unnamed: 0']) #Netherlands
else:
    target['Date']=pd.to_datetime(target['Date'])#Germany ,Sweden_1,Sweden_2,USA

# Set the split date
if Well=="Germany":
    split_date = pd.Timestamp('2017-01-01')
elif Well=="USA":
    split_date = pd.Timestamp('2017-01-01')
else:
    split_date = pd.Timestamp('2016-01-01')

# Select the data before the split date for training
train = data_input[data_input.time < split_date]

# Select the data after the split date for testing
test = data_input[data_input.time >= split_date]

#Merge the input and target variables of training period
data=target.merge(train,how='inner',left_on='Date',right_on='time')



# Now normalise the data using the scaling function defined above and  this also returns combined train and test input variables 

if Well=='USA':
    X_train_s, X_valid_s, y_train_s,X_C_scaled=Scaling_data( data[['PRCP', 'TMAX', 'TMIN', 'Stage_m', 'ET']], test[['PRCP', 'TMAX', 'TMIN', 'Stage_m', 'ET']], target["head"])
else:
    X_train_s, X_valid_s, y_train_s,X_C_scaled=Scaling_data( data[['rr', 'tg', 'tn', 'tx', 'pp', 'hu', 'fg', 'qq','et']], test[['rr', 'tg', 'tn', 'tx', 'pp', 'hu', 'fg', 'qq','et']], target["head"])

  
    




def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new

def reshape_onlyinputdata(x: np.ndarray,  seq_length: int) -> Tuple[np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        

    return x_new


# Reshape data
X_train_l,y_train_l= reshape_data(X_train_s,y_train_s,seq_length=seq_length)

X_c= reshape_onlyinputdata(X_C_scaled,seq_length=seq_length)


X_valid_l=X_c[int((len(X_train_l))):]




X_train_ls, X_valid_ls, y_train_ls, y_valid_ls  = train_test_split(X_train_l, y_train_l , test_size=0.2,random_state=1,shuffle=False)






def func_dl(trial):
   with device('/gpu:0'):    
    #     tf.config.experimental.set_memory_growth('/gpu:0', True)    
        set_seed(2)
        seed(1)
        
        
        
        optimizer_candidates={
            "adam":Adam(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True)),
            # "SGD":SGD(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True)),
            # "RMSprop":RMSprop(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True))
        }
        optimizer_name=trial.suggest_categorical("optimizer",list(optimizer_candidates))
        optimizer1=optimizer_candidates[optimizer_name]

    
        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=50,restore_best_weights = True),TFKerasPruningCallback(trial, monitor='val_loss')]
        
        epochs=trial.suggest_int('epochs', 50, 500,step=50)
        batch_size=trial.suggest_int('batch_size', 16,256,step=16)
        #weight=trial.suggest_float("weight", 1, 5)
        n_layers = trial.suggest_int('n_layers', 1, 6)
        model = Sequential()
        for i in range(n_layers):
            num_hidden = trial.suggest_int("n_units_l{}".format(i), 10, 100,step=10)
            return_sequences = True
            if i == n_layers-1:
                return_sequences = False
            # Activation function for the hidden layer
            if modtype == "GRU":
                model.add(GRU(num_hidden,input_shape=(X_train_ls.shape[1],X_train_ls.shape[2]),return_sequences=return_sequences))
            elif modtype == "LSTM":
                model.add(LSTM(num_hidden,input_shape=(X_train_ls.shape[1],X_train_ls.shape[2]),return_sequences=return_sequences))
            elif modtype == "BILSTM":
                model.add(Bidirectional(LSTM(num_hidden,input_shape=(X_train_ls.shape[1],X_train_ls.shape[2]),return_sequences=return_sequences)))
            model.add(Dropout(trial.suggest_float("dropout_l{}".format(i), 0.2, 0.2), name = "dropout_l{}".format(i)))
            #model.add(Dense(units = 1, name="dense_2", kernel_initializer=trial.suggest_categorical("kernel_initializer",['uniform', 'lecun_uniform']),  activation = 'Relu'))
        #model.add(BatchNormalization())study_blstm_
        model.add(Dense(1))
        model.compile(optimizer=optimizer1,loss="mse",metrics=['mse'])
        ##model.summary()


        model.fit(X_train_l,y_train_l,validation_data = (X_valid_ls,y_valid_ls ),shuffle = False,batch_size =batch_size,epochs=epochs,callbacks=callbacks
                  ,verbose = False)
  
        score=model.evaluate(X_valid_ls,y_valid_ls)

       

        return score[1]


Study_DL= optuna.create_study(direction='minimize',sampler=TPESampler(seed=10),study_name='study_'+str(modtype)+str(Well))
Study_DL.optimize(func_dl,n_trials=100)


import pickle
pickle.dump(Study_DL,open('./'+'study_'+str(modtype)+str(Well)+'.pkl', 'wb'))



par = Study_DL.best_params
par_names = list(par.keys())

def optimizer_1(learning_rate,optimizer):
        tf.random.set_seed(init+11111)
        if optimizer==Adam:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer==SGD:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        return opt

#@jit(nopython=True)
def gwlmodel(init, params):
        with tf.device('/gpu:0'):
            seed(init+99999)
            tf.random.set_seed(init+11111)
            par= params.best_params
            # seq_length = par.get(par_names[0])
            learning_rate=par.get(par_names[0])
            optimizer = par.get(par_names[1])
            epochs = par.get(par_names[2])
            batch_size = par.get(par_names[3])
            n_layers = par.get(par_names[4])
            # X_train, X_valid, y_train, y_valid=Scaling_data( X_tr'+ str(file_c['code_bss'][k])], X_va'+ str(file_c['code_bss'][k])], y_t'+ str(file_c['code_bss'][k])], y_v'+ str(file_c['code_bss'][k])])
            # X_train_l, y_train_l = reshape_data(X_train, y_train,seq_length=seq_length)
            model = Sequential()
            # i = 1
            for i in range(n_layers):
                return_sequences = True
                if i == n_layers-1:
                    return_sequences = False
                if modtype == "GRU":
                    model.add(GRU(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences))
                elif modtype == "LSTM":
                    model.add(LSTM(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences))
                elif modtype == "BILSTM":
                    model.add(Bidirectional(LSTM(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences)))
                model.add(Dropout(par["dropout_l{}".format(i)]))
            model.add(Dense(1))
            opt = optimizer_1(learning_rate,optimizer)
            model.compile(optimizer = opt, loss="mse",metrics = ['mse'])
            callbacks = [EarlyStopping(monitor = 'val_loss', mode ='min', verbose = 1, patience=50,restore_best_weights = True), tf.keras.callbacks.ModelCheckpoint(filepath='best_model'+str(modtype)+str(init)+str(Well)+'.h5', monitor='val_loss', save_best_only = True, mode = 'min')]
            model.fit(X_train_l, y_train_l,validation_split = 0.2, batch_size=batch_size, epochs=epochs,callbacks=callbacks)
        return model
    






target_scaler = MinMaxScaler(feature_range=(0,1))
target_scaler.fit(np.array(target["head"]).reshape(-1,1))
sim_init= np.zeros((len(X_valid_l), initm))
sim_init[:]=np.nan
sim_tr= np.zeros((len(X_train_l), initm))
sim_tr[:]=np.nan


for init in range(initm):
    globals()["model"+str(init)]= gwlmodel(init,Study_DL)
    # globals()["model"+str(init)]=tf.keras.models.load_model(r"/home/chidesiv/Desktop/Scripts_py/selectedstations/data/Conservative/BC_PTWT/"+'best_model'+str(modtype)+str(init)+str(ID)+str(wavelet)+'.h5')
    # globals()["model"+str(init)]=tf.keras.models.load_model('best_model'+str(modtype)+str(init)+str(Well)+'.h5')
    globals()["y_pred_valid"+str(init)] = globals()["model"+str(init)].predict(X_valid_l)
    globals()["sim_test"+str(init)]  = target_scaler.inverse_transform(globals()["y_pred_valid"+str(init)])
    globals()["y_pred_train"+str(init)] = globals()["model"+str(init)].predict(X_train_l)
    globals()["sim_train"+str(init)] = target_scaler.inverse_transform(globals()["y_pred_train"+str(init)])


Train_index=data['Date']
Test_index=test.time   
for init in range(initm):
    sim_init[:,init]=globals()["sim_test"+str(init)][:,0]
    sim_tr[:,init]=globals()["sim_train"+str(init)][:,0]
    # sim_tc[:,init]=globals()["sim_c"+str(init)][:,0]
    
    sim_f=pd.DataFrame(sim_init)
    sim_t=pd.DataFrame(sim_tr)
    # sim_co=pd.DataFrame(sim_tc)
    sim_mean=sim_f.mean(axis=1) 
    sim_tr_mean=sim_t.mean(axis=1) 
    # sim_tc_mean=sim_co.mean(axis=1) 
    sim_init_uncertainty = unumpy.uarray(sim_f.mean(axis=1),1.96*sim_f.std(axis=1))
    sim_tr_uncertainty = unumpy.uarray(sim_t.mean(axis=1),1.96*sim_t.std(axis=1))
    
    sim=np.asarray(sim_mean).reshape(-1,1)
    sim_train=np.asarray(sim_tr_mean).reshape(-1,1)
    
    obs_tr = np.asarray(target_scaler.inverse_transform(y_train_l).reshape(-1,1))
    
    y_err = unumpy.std_devs(sim_init_uncertainty)
    y_err_tr = unumpy.std_devs(sim_tr_uncertainty)
    
    
    submission_tr=pd.DataFrame({'Date':Train_index[seq_length-1:],'Simulated Head':sim_train.reshape(-1,),'95% Lower Bound':sim_train.reshape(-1,) - y_err_tr,'95% Upper Bound':sim_train.reshape(-1,) + y_err_tr})
    submission_te=pd.DataFrame({'Date':Test_index,'Simulated Head':sim.reshape(-1,),'95% Lower Bound':sim.reshape(-1,) -y_err,'95% Upper Bound':sim.reshape(-1,) + y_err})
    submission_f=pd.concat([submission_tr,submission_te])
    submission_final=submission_f.reset_index(drop=True)

submission_final.to_csv('submission_form_'+str(Well)+'.csv')



from matplotlib import pyplot
pyplot.figure(figsize=(20,6))
Train_index=data['Date']
Test_index=test.time


       
pyplot.plot( Train_index,data["head"], 'k', label ="observed train", linewidth=1.5,alpha=0.9)

pyplot.fill_between(Train_index[seq_length-1:],sim_train.reshape(-1,) - y_err_tr,
                sim_train.reshape(-1,) + y_err_tr, facecolor = (1.0, 0.8549, 0.72549),
                label ='95% confidence training',linewidth = 1,
                edgecolor = (1.0, 0.62745, 0.47843))    
pyplot.plot( Train_index[seq_length-1:],sim_train, 'r', label ="simulated  median train", linewidth = 0.9)
pyplot.plot( Test_index,sim, 'r', label ="simulated mean", linewidth = 0.9)

        

pyplot.fill_between( Test_index,sim.reshape(-1,) - y_err,
                sim.reshape(-1,) + y_err, facecolor = (1,0.7,0,0.4),
                label ='95% confidence',linewidth = 1,
                edgecolor = (1,0.7,0,0.7))    


pyplot.vlines(x=Test_index.iloc[0], ymin=[data["head"].min()*0.999], ymax=[data["head"].max()*1.001], colors='teal', ls='--', lw=2, label='Start of Testing data')
pyplot.title((str(modtype)+' '+ "ALL"), size=24,fontweight = 'bold')
pyplot.xticks(fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.ylabel('GWL [m asl]', size=24)
pyplot.xlabel('Date',size=24)
pyplot.tight_layout()

pyplot.savefig('./'+'Well_ID'+str(Well)+str(modtype)+str(initm)+'PT_full.png', dpi=600, bbox_inches = 'tight', pad_inches = 0.1)



