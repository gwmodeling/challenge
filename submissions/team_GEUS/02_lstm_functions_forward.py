import pandas as pd
import numpy as np
import functions as fn
from matplotlib import pyplot as plt

wells=["Germany","Netherlands","Sweden_1","Sweden_2","USA"]

f=0
# optimized parameters: n_steps	batchsize	n_cells	dropout	rec_dropout	learning_rate	rr_acc	rrnet_acc	snow
lstm_par={
    "Germany": [70,128,16,0.1235991,0.005899819,0.003489363,1,1,0],
    "Netherlands": [66,118,79,0.02927205,0.307829,0.001,2,2,1],
    "Sweden_1": [75,77,101,0.06206156,0.4,0.001,2,0,0],
    "Sweden_2": [7,81,62,0.167958,0.29374,0.001,2,1,0],
    "USA": [49,74,49,0.246498,0.2360094,0.002731655,2,4,0]
    }

# which climate variables to use
var_dict_climate={
    "Germany": ['time','rr','tg','et'],
    "Netherlands": ['time','rr','tg','et'],
    "Sweden_1": ['time','rr', 'tg','et'],
    "Sweden_2": ['time','rr', 'tg','et'],
    "USA": ['time','PRCP','Stage_m','ET']
    }

# start and end date for each of the well submissions
dates={    
    "Germany": ['2002-05-01',' 2021-12-31'],
    "Netherlands": ['2000-01-01',' 2020-11-27'],
    "Sweden_1": ['2001-01-02',' 2020-12-29'],
    "Sweden_2": ['2001-01-02',' 2020-12-29'],
    "USA": ['2002-03-01',' 2021-12-31']
    }

for well in wells:
    print(well)
    
    # control of rr accumulated
    if lstm_par[well][6]==0:
        var_rr_acc=[]
    if lstm_par[well][6]==1:
        var_rr_acc=['rr_6m'] 
    if lstm_par[well][6]==2:
        var_rr_acc=['rr_6m','rr_12m'] 
    if lstm_par[well][6]==3:
        var_rr_acc=['rr_6m','rr_12m','rr_24m'] 
    if lstm_par[well][6]==4:
        var_rr_acc=['rr_6m','rr_12m','rr_24m','rr_36m',]
    
    # control of rr_net accumulated
    if lstm_par[well][7]==0:
        var_rrnet_acc=[]
    if lstm_par[well][7]==1:
        var_rrnet_acc=['rrnet_6m'] 
    if lstm_par[well][7]==2:
        var_rrnet_acc=['rrnet_6m','rrnet_12m'] 
    if lstm_par[well][7]==3:
        var_rrnet_acc=['rrnet_6m','rrnet_12m','rrnet_24m'] 
    if lstm_par[well][7]==4:
        var_rrnet_acc=['rrnet_6m','rrnet_12m','rrnet_24m','rrnet_36m']
    
    # control of snow
    if lstm_par[well][8]==0:
        var_snow=[]  
    if lstm_par[well][8]==1:
        var_snow=['sn_stor','sn_melt','sn_melt_3m']    
    
    #define parameters
    n_steps = int(lstm_par[well][0]) # input sequence length
    epochs=100 # number trainign epoches
    batch_size=int(lstm_par[well][1]) # bacth size for training
    n_cells=int(lstm_par[well][2]) # number of cells in lstm network
    dropout=lstm_par[well][3] # dropout rate
    recurrent_dropout=lstm_par[well][4] # recurrent dropout rate
    learning_rate=lstm_par[well][5] # learnign rate
    n_seed=25 # number of seeds/training --> mean used for final prediction
    y_test=4 # number of years from start used fo testing and early stopping
    
    #read obs head
    dat_head=pd.read_csv("./data/"+well+"/heads.csv")
    dat_head=dat_head.set_index('Date')
    dat_head.index = pd.to_datetime(dat_head.index)
    
    #read covariate data
    dat_input=pd.read_csv("./data/"+well+"/input_data.csv") # standard input variables
    dat_input_accu=pd.read_csv("./data/"+well+"/input_data_agg.csv") # accumulated variables
    dat_input_snow=pd.read_csv("./data/"+well+"/input_data_snow.csv") # snow variables
    # join all input variables
    dat_input=dat_input[var_dict_climate[well]].join(dat_input_accu[var_rr_acc])
    dat_input=dat_input.join(dat_input_accu[var_rrnet_acc])
    dat_input=dat_input.join(dat_input_snow[var_snow])
    dat_input=dat_input.set_index('time')
    dat_input.index = pd.to_datetime(dat_input.index)
    
    #make copy for forward run after training
    dat_input_long=dat_input.copy()
    
    # only use temporal subset where h data is available
    dat_input=dat_input[dat_input.index <= dat_head.index.max()]
    dat_input=dat_input[dat_input.index >= np.datetime64(dat_head.index.min())-np.timedelta64(n_steps-1,'D')]
    # make one datframe with obs head and covariates
    dat=pd.concat([dat_head,dat_input], ignore_index=False, axis=1)
    dat.index = pd.to_datetime(dat.index)
    # make one datframe with obs head and covariates for long timeserien. obs head is just a dummy variable here to make the subsequent functions work
    dat_long=pd.concat([dat_input_long[var_dict_climate[well][1]],dat_input_long], ignore_index=False, axis=1)
    dat_long.index = pd.to_datetime(dat_long.index)
    # slit into train and test
    split_date=np.datetime64(dat_head.index.min())+np.timedelta64(y_test*365,'D')
    train=dat[dat.index >= split_date-np.timedelta64(n_steps+1,'D')]
    test=dat[dat.index < split_date]
    
    n_features=dat_input.shape[1]
    
    #normalize data with mean and std from train period
    avg=np.nanmean(train.values,axis=0)
    std=np.nanstd(train.values,axis=0)
    train_norm= fn.norm(train,avg,std)
    test_norm= fn.norm(test,avg,std)
    predict_norm= fn.norm(dat_long,avg,std)
    
    #generate input sequences for lstm model
    X_train, y_train = fn.split_sequences(train_norm, n_steps)
    X_test, y_test = fn.split_sequences(test_norm, n_steps)
    X_predict, y_predict = fn.split_sequences(predict_norm, n_steps)
    
    yhat_lstm_predict=np.empty((len(y_predict),n_seed))
    yhat_lstm_predict_upper=np.empty((len(y_predict),n_seed))
    yhat_lstm_predict_lower=np.empty((len(y_predict),n_seed))
    # train and predcit on test data
    for i in range(0,n_seed):
        print("lstm seed "+str(int(i)))
        gw_model=fn.train_lstm_model(n_steps, n_features, epochs, batch_size, n_cells, dropout, recurrent_dropout, learning_rate,X_train,y_train,X_test,y_test,np.nan)   
        yhat_lstm_predict[:,i]=fn.rev_norm(gw_model.predict(X_predict, verbose=0),avg,std)[:,0]
    
        gw_model=fn.train_lstm_model(n_steps, n_features, epochs, batch_size, n_cells, dropout, recurrent_dropout, learning_rate,X_train,y_train,X_test,y_test,1-0.025)   
        yhat_lstm_predict_upper[:,i]=fn.rev_norm(gw_model.predict(X_predict, verbose=0),avg,std)[:,0]
    
        gw_model=fn.train_lstm_model(n_steps, n_features, epochs, batch_size, n_cells, dropout, recurrent_dropout, learning_rate,X_train,y_train,X_test,y_test,0.025)   
        yhat_lstm_predict_lower[:,i]=fn.rev_norm(gw_model.predict(X_predict, verbose=0),avg,std)[:,0]
    
    
    out = pd.DataFrame(index=dat_long.index,columns=['Simulated Head','95% Lower Bound','95% Upper Bound'],dtype='float')
    out.index.names = ['Date']
    out.iloc[n_steps:,0]=(np.mean(yhat_lstm_predict,axis=1)).astype(float)
    out.iloc[n_steps:,1]=np.mean(yhat_lstm_predict_lower,axis=1)
    out.iloc[n_steps:,2]=np.mean(yhat_lstm_predict_upper,axis=1)
    
    out=out[out.index >= dates[well][0]]
    out=out[out.index <= dates[well][1]]
    
    plt.figure(f)
    dat_head["head"].plot(label='Observed Head',color='k',lw=1)
    out['Simulated Head'].plot(color='r',lw=1)
    out['95% Lower Bound'].plot(color='gray',lw=1)
    out['95% Upper Bound'].plot(color='gray',lw=1)
    
    plt.legend()
    plt.ylabel('head [m]')
    plt.tight_layout()
    plt.savefig(well+'_long.png',dpi=400)
    
    out.to_csv('teamGEUS_submission_'+well+'.csv')
    
    f=f+1
