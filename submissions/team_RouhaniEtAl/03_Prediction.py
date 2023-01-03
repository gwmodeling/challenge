# reproducability
from numpy.random import seed

seed(1 + 347823)
import tensorflow as tf

tf.random.set_seed(1 + 63493)

import numpy as np
import os
import pandas as pd
import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from uncertainties import unumpy

gpus = tf.config.experimental.list_physical_devices('GPU')
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
#### Functions
# =============================================================================

def load_GW_and_HYRAS_Data(Well_ID):
    # define where to find the data
    pathGW = "./challenge-main/data"
    pathHYRAS = "./challenge-main/data"
    pathconnect = "/"

    # load and merge the data
    GWData = pd.read_csv(pathGW + pathconnect + Well_ID + pathconnect + 'heads.csv',
                         parse_dates=['Date'], index_col=0, dayfirst=True,
                         decimal='.', sep=',')
    HYRASData = pd.read_csv(pathHYRAS + pathconnect + Well_ID + pathconnect + 'input_data.csv',
                            parse_dates=['time'], index_col=0, dayfirst=True,
                            decimal='.', sep=',')
    data = pd.merge(GWData, HYRASData, how='inner', left_index=True, right_index=True)

    return data


def split_data(data, GLOBAL_SETTINGS):
    dataset = data[(data.index < GLOBAL_SETTINGS["'simulation_start'"])]

    TrainingData = dataset[0:round(0.8 * len(dataset))]
    StopData = dataset[round(0.8 * len(dataset)) + 1:]
    StopData_ext = dataset[round(0.8 * len(dataset)) + 1 - GLOBAL_SETTINGS[
        "seq_length"]:]  # extend data according to delays/sequence length

    return TrainingData, StopData, StopData_ext


def to_supervised(data, GLOBAL_SETTINGS):
    # make the data sequential
    # modified after Jason Brownlee and machinelearningmastery.com

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


def to_supervised_onlyX(data, GLOBAL_SETTINGS):
    # make the data sequential
    # modified after Jason Brownlee and machinelearningmastery.com

    X = list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + GLOBAL_SETTINGS["seq_length"]
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x = data[i:end_idx, :]
        X.append(seq_x)
    return np.array(X)


def generate_scalers(pp):
    # load training data again, to use same scalers
    data_orig = load_GW_and_HYRAS_Data(pp)

    # scale data
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data_orig['head']))
    data_orig.drop(columns='head', inplace=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_orig)

    return scaler, scaler_gwl


class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def predict_distribution(X, model, n):
    preds = [model(X) for _ in range(n)]
    return np.hstack(preds)


def load_proj_data(proj_name, Well_ID, GLOBAL_SETTINGS):
    data1 = load_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])
    data1 = data1.drop(columns='head')

    path = "./" + Well_ID + "input_data.csv" + proj_name + ".csv"
    proj_data = pd.read_csv(path, parse_dates=['time'], index_col=0, dayfirst=True,
                            decimal='.', sep=',')
    proj_data = proj_data[(proj_data.index >= GLOBAL_SETTINGS["'simulation_start'"])]
    proj_data_ext = pd.concat([data1.iloc[-GLOBAL_SETTINGS["seq_length"]:], proj_data], axis=0)

    return proj_data, proj_data_ext


def applymodels_to_extreme_historicaldata(Well_ID, densesize_int, seqlength_int, batchsize_int, filters_int):
    GLOBAL_SETTINGS = {
        'pp': Well_ID,
        'batch_size': batchsize_int,
        'kernel_size': 3,
        'dense_size': densesize_int,
        'filters': filters_int,
        'seq_length': seqlength_int,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 100,
        'learning_rate': 1e-3,
        'simulation_start': pd.to_datetime('16092013', format='%d%m%Y')
    }

    pathHYRAS = "./challenge-main/data"
    pathconnect = "/"
    data = pd.read_csv(pathHYRAS + pathconnect + Well_ID + 'input_data.csv',
                       parse_dates=['time'], index_col=0, dayfirst=True,
                       decimal='.', sep=',')

    # make extreme data
    # data2 = data
    # data2['precipitation[kg m-2]'] = data2['precipitation[kg m-2]'] * 4
    # data2['P3'] = data2['P3'] * 4
    # data2['P6'] = data2['P6'] * 4
    # data2['P12'] = data2['P12'] * 4
    # data2['P18'] = data2['P18'] * 4
    # data2['P24'] = data2['P24'] * 4
    # data2['P36'] = data2['P36'] * 4
    # data2['temperature[degree_Celsius]'] = data2['temperature[degree_Celsius]'] + 5

    data = pd.read_csv(pathHYRAS + pathconnect + Well_ID + 'input_data.csv',
                       parse_dates=['time'], index_col=0, dayfirst=True,
                       decimal='.', sep=',')

    # scale data
    # scaler, scaler_gwl = generate_scalers(GLOBAL_SETTINGS["pp"])
    data_n = pd.DataFrame(scaler.transform(data2), index=data2.index, columns=data2.columns)

    # sequence data
    X_test = to_supervised_onlyX(data_n.values, GLOBAL_SETTINGS)

    # define where the models can be found
    path = './' + Well_ID

    inimax = 10
    sim_members = np.zeros((len(X_test), inimax))
    sim_members[:] = np.nan

    sim_std = np.zeros((len(X_test), inimax))
    sim_std[:] = np.nan

    for ini in range(inimax):
        loaded_model = tf.keras.models.load_model(path + "/ini" + str(ini))
        y_pred_distribution = predict_distribution(X_test, loaded_model, 100)
        sim = scaler_gwl.inverse_transform(y_pred_distribution)

        sim_members[:, ini], sim_std[:, ini] = sim.mean(axis=1), sim.std(axis=1)

    sim_members_uncertainty = unumpy.uarray(sim_members,
                                            1.96 * sim_std)  # 1.96 because of sigma rule for 95% confidence
    sim_mean = np.nanmedian(sim_members, axis=1)
    sim_mean_uncertainty = np.sum(sim_members_uncertainty, axis=1) / inimax

    return sim_mean, sim_members, sim_mean_uncertainty, sim_members_uncertainty, data, inimax


def applymodels_to_hyras(Well_ID, densesize_int, seqlength_int, batchsize_int, filters_int):
    GLOBAL_SETTINGS = {
        'pp': Well_ID,
        'batch_size': batchsize_int,
        'kernel_size': 3,
        'dense_size': densesize_int,
        'filters': filters_int,
        'seq_length': seqlength_int,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 100,
        'learning_rate': 1e-3,
        'simulation_start': pd.to_datetime('16092013', format='%d%m%Y')
    }

    pathHYRAS = "./challenge-main/data"
    pathconnect = "/"
    data = pd.read_csv(pathHYRAS + pathconnect + Well_ID + '/input_data.csv',
                       parse_dates=['time'], index_col=0, dayfirst=True,
                       decimal='.', sep=',')
    data_ext = data

    # scale data
    scaler, scaler_gwl = generate_scalers(GLOBAL_SETTINGS["pp"])
    data_ext_n = pd.DataFrame(scaler.transform(data_ext), index=data_ext.index, columns=data_ext.columns)

    # sequence data
    X_test = to_supervised_onlyX(data_ext_n.values, GLOBAL_SETTINGS)

    # define where the models can be found
    path = './' + Well_ID

    inimax = 10
    sim_members = np.zeros((len(X_test), inimax))
    sim_members[:] = np.nan

    sim_std = np.zeros((len(X_test), inimax))
    sim_std[:] = np.nan

    for ini in range(inimax):
        loaded_model = tf.keras.models.load_model(path + "/ini" + str(ini))
        y_pred_distribution = predict_distribution(X_test, loaded_model, 100)
        sim = scaler_gwl.inverse_transform(y_pred_distribution)
        sim_members[:, ini], sim_std[:, ini] = sim.mean(axis=1), sim.std(axis=1)



    sim_members_uncertainty = unumpy.uarray(sim_members,
                                            1.96 * sim_std)  # 1.96 because of sigma rule for 95% confidence
    sim_mean = np.nanmedian(sim_members, axis=1)
    sim_mean_uncertainty = np.sum(sim_members_uncertainty, axis=1) / inimax

    return sim_mean, sim_members, sim_mean_uncertainty, sim_members_uncertainty, data, inimax


# =============================================================================
#### start
# =============================================================================

with tf.device("/gpu:0"):
    time1 = datetime.datetime.now()
    basedir = './'
    os.chdir(basedir)

    well_list = pd.read_csv("./well_list")

    # =============================================================================
    #### loop
    # =============================================================================

    for pp in range(2, 4): # well_list.shape[0]
        Well_ID = well_list.piezometer[pp]
        print(str(pp) + ": " + Well_ID)

        if not os.path.exists(Well_ID):
            os.makedirs(Well_ID)

        # read optimized hyperparameters
        bestkonfig = pd.read_csv('./log_summary_CNN_' + Well_ID + '.txt',
                                 delimiter='=', skiprows=(10), nrows=(7), header=None, encoding='unicode_escape')
        bestkonfig.columns = ['hp', 'value']
        filters_int = int(bestkonfig.value[0])
        densesize_int = int(bestkonfig.value[1])
        seqlength_int = int(bestkonfig.value[2])
        batchsize_int = int(bestkonfig.value[3])

        pathGW = "./challenge-main/data/" # challenge-main/data/Germany/heads.csv
        GWData = pd.read_csv(pathGW + Well_ID + '/heads.csv', parse_dates=['Date'], index_col=0, dayfirst=True,
                             decimal='.', sep=',')
        GWData = GWData[(GWData.index <= pd.to_datetime('01012016', format='%d%m%Y'))]

        sim_mean, sim_members, sim_mean_uncertainty, sim_members_uncertainty, data, inimax = applymodels_to_hyras(
            Well_ID, densesize_int, seqlength_int, batchsize_int, filters_int)

        # sim_mean2, sim_members2, sim_mean_uncertainty2, sim_members_uncertainty2, data2, inimax2 = applymodels_to_extreme_historicaldata(
        #     Well_ID, densesize_int, seqlength_int, batchsize_int, filters_int)

        # =============================================================================
        #### plot Test-Section
        # =============================================================================

        y_err = unumpy.std_devs(sim_mean_uncertainty)
        # y_err2 = unumpy.std_devs(sim_mean_uncertainty2)

        pyplot.figure(figsize=(20, 6))

        pyplot.fill_between(data.index[seqlength_int:], sim_mean.reshape(-1, ) - y_err,
                            sim_mean.reshape(-1, ) + y_err, facecolor=(0.25, 0.65, 1, 0.5),
                            label='95% confidence', linewidth=1,
                            edgecolor=(0.25, 0.65, 1, 0.6))
        pyplot.plot(data.index[seqlength_int:], sim_mean, 'b', label="simulated median\n(observed climate\ndata)",
                    linewidth=1)

        # pyplot.fill_between(data2.index[seqlength_int:], sim_mean2.reshape(-1, ) - y_err2,
        #                     sim_mean2.reshape(-1, ) + y_err2, facecolor=(1, 0.7, 0, 0.5),
        #                     label='95% confidence', linewidth=1,
        #                     edgecolor=(1, 0.7, 0, 0.6))
        # pyplot.plot(data2.index[seqlength_int:], sim_mean2, 'r', label="simulated median\n(extreme climate\nscenario)",
        #             linewidth=1)

        pyplot.plot(GWData.index, GWData['head'], 'k', label="observed GWL", alpha=0.9)
        pyplot.title(Well_ID, size=17, fontweight='bold')
        pyplot.ylabel('Head [m]', size=15)
        pyplot.xlabel('Date', size=15)
        pyplot.legend(fontsize=14, bbox_to_anchor=(1, 1), fancybox=False, framealpha=1, edgecolor='k')
        pyplot.tight_layout()
        pyplot.grid(b=True, which='major', color='#666666', alpha=0.3, linestyle='-')
        pyplot.xticks(fontsize=14)
        pyplot.yticks(fontsize=14)

        pyplot.savefig('./' + Well_ID + '/Climproj_' + Well_ID + "_" + 'extremehist' + '_CNN_PT_.png', dpi=600)
        pyplot.savefig('./Climproj_' + Well_ID + "_" + 'extremehist' + '_CNN_PT_.png', dpi=600)
        # pyplot.show()

        # print sim data
        printdf = pd.DataFrame(data=sim_members, index=data.index[seqlength_int:])
        printdf.to_csv('./' + Well_ID + '/ensemble_member_values_CNN_' + Well_ID + '_' + 'hyras(extremehist)' + '.txt',
                       sep=';')

        printdf = pd.DataFrame(data=sim_members_uncertainty, index=data.index[seqlength_int:])
        printdf.to_csv('./' + Well_ID + '/ensemble_member_errors_CNN_' + Well_ID + '_' + 'hyras(extremehist)' + '.txt',
                       sep=';')

        printdf = pd.DataFrame(data=np.c_[sim_mean, y_err], index=data.index[seqlength_int:])
        printdf = printdf.rename(columns={0: 'Sim', 1: 'Sim_Error'})
        printdf.to_csv('./' + Well_ID + '/ensemble_mean_values_CNN_' + Well_ID + '_' + 'hyras(extremehist)' + '.txt',
                       sep=';', float_format='%.6f')
        ###
        # printdf = pd.DataFrame(data=sim_members2, index=data2.index[seqlength_int:])
        # printdf.to_csv('./' + Well_ID + '/ensemble_member_values_CNN_' + Well_ID + '_' + 'extremehist' + '.txt',
        #                sep=';')
        #
        # printdf = pd.DataFrame(data=sim_members_uncertainty2, index=data2.index[seqlength_int:])
        # printdf.to_csv('./' + Well_ID + '/ensemble_member_errors_CNN_' + Well_ID + '_' + 'extremehist' + '.txt',
        #                sep=';')
        #
        # printdf = pd.DataFrame(data=np.c_[sim_mean2, y_err2], index=data2.index[seqlength_int:])
        # printdf = printdf.rename(columns={0: 'Sim', 1: 'Sim_Error'})
        # printdf.to_csv('./' + Well_ID + '/ensemble_mean_values_CNN_' + Well_ID + '_' + 'extremehist' + '.txt', sep=';',
        #                float_format='%.6f')