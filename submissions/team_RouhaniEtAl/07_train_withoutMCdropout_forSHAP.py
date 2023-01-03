# reproducability
from numpy.random import seed

seed(1 + 347823)
import tensorflow as tf

tf.random.set_seed(1 + 63493)

import numpy as np
import os
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
import warnings
warnings.filterwarnings("ignore")

def load_GW_and_HYRAS_Data(Well_ID):
    pathGW = "./challenge-main/data"
    pathHYRAS = "./challenge-main/data"
    pathconnect = "/"

    GWData = pd.read_csv(pathGW + pathconnect + Well_ID + pathconnect + 'heads.csv',
                         parse_dates=['Date'], index_col=0, dayfirst=True,
                         decimal='.', sep=',')
    HYRASData = pd.read_csv(pathHYRAS + pathconnect + Well_ID + pathconnect + 'input_data.csv',
                            parse_dates=['time'], index_col=0, dayfirst=True,
                            decimal='.', sep=',')
    data = pd.merge(GWData, HYRASData, how='inner', left_index=True, right_index=True)

    return data


def split_data(data, GLOBAL_SETTINGS):
    dataset = data[(data.index < GLOBAL_SETTINGS["simulation_start"])]  # Testdaten abtrennen

    TrainingData = dataset[0:round(0.8 * len(dataset))]
    StopData = dataset[round(0.8 * len(dataset)) + 1:]
    StopData_ext = dataset[round(0.8 * len(dataset)) + 1 - GLOBAL_SETTINGS[
        "seq_length"]:]  # extend data according to delays/sequence length

    return TrainingData, StopData, StopData_ext


def to_supervised(data, GLOBAL_SETTINGS):
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + GLOBAL_SETTINGS["seq_length"]
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_idx, 1:], data[end_idx, 0]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)


def gwmodel(ini, GLOBAL_SETTINGS, X_train, Y_train, X_stop, Y_stop):
    # define models
    seed(ini + 872527)
    tf.random.set_seed(ini + 87747)

    inp = tf.keras.Input(shape=(GLOBAL_SETTINGS["seq_length"], X_train.shape[2]))
    cnn = tf.keras.layers.Conv1D(filters=GLOBAL_SETTINGS["filters"],
                                 kernel_size=GLOBAL_SETTINGS["kernel_size"],
                                 activation='relu',
                                 padding='same')(inp)
    cnn = tf.keras.layers.MaxPool1D(padding='same')(cnn)
    cnn = tf.keras.layers.Dropout(0.5)(cnn)
    cnn = tf.keras.layers.Flatten()(cnn)
    cnn = tf.keras.layers.Dense(GLOBAL_SETTINGS["dense_size"], activation='relu')(cnn)
    output1 = tf.keras.layers.Dense(1, activation='linear')(cnn)

    # tie together
    model = tf.keras.Model(inputs=inp, outputs=output1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=GLOBAL_SETTINGS["learning_rate"],
                                         epsilon=10E-3, clipnorm=GLOBAL_SETTINGS["clip_norm"])

    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                                          patience=15, restore_best_weights=True)

    # fit network
    history = model.fit(X_train, Y_train, validation_data=(X_stop, Y_stop),
                        epochs=GLOBAL_SETTINGS["epochs"], verbose=0,
                        batch_size=GLOBAL_SETTINGS["batch_size"], callbacks=[es])

    return model, history


def train_and_save_model(Well_ID, densesize_int, seqlength_int, batchsize_int, filters_int):
    GLOBAL_SETTINGS = {
        'pp': Well_ID,
        'batch_size': batchsize_int,  # 16-128
        'kernel_size': 3,  # ungerade
        'dense_size': densesize_int,
        'filters': filters_int,
        'seq_length': seqlength_int,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 100,
        'learning_rate': 1e-3,
        'simulation_start': pd.to_datetime('16092013', format='%d%m%Y'),
    }

    ## load data
    data = load_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])

    # scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data['head']))
    data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    # split data
    TrainingData, StopData, StopData_ext = split_data(data, GLOBAL_SETTINGS)
    TrainingData_n, StopData_n, StopData_ext_n = split_data(data_n, GLOBAL_SETTINGS)

    # sequence data
    X_train, Y_train = to_supervised(TrainingData_n.values, GLOBAL_SETTINGS)
    X_stop, Y_stop = to_supervised(StopData_ext_n.values, GLOBAL_SETTINGS)

    # build and train model with different initializations
    inimax = 10

    # define model path
    path = './' + Well_ID;
    if os.path.isdir(path) == False:
        os.mkdir(path)

    f = open(path + 'traininghistory_CNN_' + Well_ID + '.txt', "w")

    for ini in range(inimax):
        if os.path.isdir(path + "/ini" + str(ini)) == False:

            print(str(pp) + ": " + Well_ID + "_ini" + str(ini))
            model, history = gwmodel(ini, GLOBAL_SETTINGS, X_train, Y_train, X_stop, Y_stop)

            model.save(path + "/ini" + str(ini))

            loss = np.zeros((1, 100))
            loss[:, :] = np.nan
            loss[0, 0:np.shape(history.history['loss'])[0]] = history.history['loss']
            val_loss = np.zeros((1, 100))
            val_loss[:, :] = np.nan
            val_loss[0, 0:np.shape(history.history['val_loss'])[0]] = history.history['val_loss']
            print('loss', file=f)
            print(loss.tolist(), file=f)
            print('val_loss', file=f)
            print(val_loss.tolist(), file=f)

        else:
            print(Well_ID + "_ini" + str(ini) + " - already exists")

    f.close()
    return GLOBAL_SETTINGS


# =============================================================================
#### start
# =============================================================================
with tf.device("/gpu:0"):
    time1 = datetime.datetime.now()
    basedir = 'F:/pythonProject/groundWaterModellingChalleng'
    os.chdir(basedir)

    well_list = pd.read_csv("./well_list")

    # =============================================================================
    #### training loop
    # =============================================================================

    for pp in range(well_list.shape[0]):
        time_single = datetime.datetime.now()
        seed(1)
        tf.random.set_seed(1)

        Well_ID = well_list.piezometer[pp]

        bestkonfig = pd.read_csv('./log_summary_CNN_' + Well_ID + '.txt', delimiter='=', skiprows=(10), nrows=(7),
                                 header=None, encoding= 'unicode_escape')
        bestkonfig.columns = ['hp', 'value']
        filters_int = int(bestkonfig.value[0])
        densesize_int = int(bestkonfig.value[1])
        seqlength_int = int(bestkonfig.value[2])
        batchsize_int = int(bestkonfig.value[3])

        train_and_save_model(Well_ID, densesize_int, seqlength_int, batchsize_int, filters_int)
