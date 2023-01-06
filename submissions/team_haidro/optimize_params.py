import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from dateutil.relativedelta import relativedelta
import gc
import os


# Important note:
#   This code could use some refactoring but that has not been done (yet)


def opt_params(location, scale_q=True, aggregation_days=5, hist_years=10):
    """
    Function to get the optimal hyperparameters for a certain location
    :param location: location to do the optimization for
    :param scale_q: Use a standard scaler for the GW levels or not
    :param aggregation_days: The aggregation of the days in the most recent period
    :param hist_years: Lookback period (in years) to take into account
    :return: tuple wiht optimal hidden units and optimal epoch
    """

    # Create output folder
    o_folder = f'LSTM/{location}'

    if not os.path.exists(o_folder):
        os.makedirs(o_folder)

    # Load data
    df = pd.read_csv(f'../data/{location}/heads.csv', index_col=0, parse_dates=True)
    df_meteo = pd.read_csv(f'../data/{location}/input_data.csv', index_col=0, parse_dates=True)

    # Truncate to period where we can make predictions
    first_available_date = df_meteo.first_valid_index() + relativedelta(days=60) + relativedelta(days=12*hist_years*30)
    df = df.truncate(before=first_available_date)

    # USA has a slightly different naming
    if location == 'USA':
        rain_c = 'PRCP'
        ep_c = 'ET'
    else:
        rain_c = 'rr'
        ep_c = 'et'

    # Add recharge as explicit input
    df_meteo['voeding'] = df_meteo[rain_c] - df_meteo[ep_c]

    # Scale the data
    scaled_data = df.copy()

    if scale_q:
        scaler_q = StandardScaler()
        scaled_data['head'] = scaler_q.fit_transform(np.reshape(df['head'].values, (-1, 1)))
    else:
        scaled_data['head'] = df['head'] - df['head'].mean()

    scaled_data['head'] = df['head'] - df['head'].mean()

    scaler_10 = {}
    scaler_30 = {}

    for c in df_meteo.columns:
        # Not 100% fair. Should still apply the scaler only to the calibration period
        # Note scaler is not yet applied, only fitted. Will be applied in the next transform step
        scaler_10[c] = StandardScaler()
        scaler_10[c].fit(np.reshape(df_meteo.resample(f'{aggregation_days}D').sum()[c].dropna().values, (-1, 1)))
        scaler_30[c] = StandardScaler()
        scaler_30[c].fit(np.reshape(df_meteo.resample('30D').sum()[c].dropna().values, (-1, 1)))

    aantal_y_hist = 12 * hist_years + 60 / aggregation_days

    def transform_data(subset='cal'):
        """
        Function to transform the data to what is needed for
        :param subset: string indicating if we need to prepare the calibration or the validation set
        :return: X (3D matrix) and Y (2D matrix) that can be fed into the model
        """

        if subset == 'cal':
            df_ = scaled_data.iloc[:int(df.shape[0] * 0.75), :]
        else:
            df_ = scaled_data.iloc[int(df.shape[0] * 0.75):, :]

        x_l = np.empty((df_.shape[0], int(aantal_y_hist), df_meteo.shape[1]+1))

        for ii, i in enumerate(df_.index):

            istartd = i - relativedelta(days=60)
            istartm = istartd - relativedelta(days=12*hist_years*30)
            df_c_d = df_meteo.truncate(after=i, before=istartd + relativedelta(days=1)).resample(f'{aggregation_days}D').sum()
            df_c_m = df_meteo.truncate(after=istartd, before=istartm + relativedelta(days=1)).resample('30D').sum()

            for c in df_c_d:
                df_c_d[c] = scaler_10[c].transform(np.reshape(df_c_d[c].values, (-1, 1)))
                df_c_m[c] = scaler_30[c].transform(np.reshape(df_c_m[c].values, (-1, 1)))

            # dummy indicates temporal frequency
            df_c_m['dummy'] = 0
            df_c_d['dummy'] = 1
            df_c_ = pd.concat([df_c_m, df_c_d])

            x_l[ii, :, :] = df_c_.values

        y_l = df_.values

        return x_l, y_l

    # Prepare data for calibration and validation
    x_c_f, y_c_f = transform_data()
    x_v_f, y_v_f = transform_data(subset='val')

    # Start with some fixed hyperparameters. Note, we are not fine tuning all possible hyperparameters here.
    d = 0.2
    lr = 1e-3
    min_loss = 99999
    opt_hidden_units = 0
    max_epoch = 40

    for h in [10, 20, 40, 60, 80, 100, 120]:
        print(f'Checking hidden unit size of {h}')
        val_loss_list = []
        # We use an ensemble of 10 models
        for it in range(10):
            # Define the model
            opt = Adam(learning_rate=lr)
            model = keras.Sequential(name='HydroML')
            model.add(layers.LSTM(h, input_shape=(x_c_f.shape[1], x_c_f.shape[-1])))
            model.add(layers.Dropout(d))
            model.add(layers.Dense(1))
            model.add(layers.Activation("linear"))
            model.compile(loss='mse', optimizer=opt)

            history = model.fit(x_c_f, y_c_f, verbose=0, epochs=max_epoch, batch_size=1024, validation_data=(x_v_f, y_v_f))

            val_loss_list.append(np.expand_dims(np.array(history.history['val_loss']), -1))

            # Necessary to avoid memory issues
            del model
            del opt
            tf.keras.backend.clear_session()
            gc.collect()

        # Final validation loss is the mean of the ensembles
        val_loss_fin = np.mean(np.concatenate(val_loss_list, axis=1), axis=1)
        if np.min(val_loss_fin) < min_loss:
            min_loss = np.min(val_loss_fin)
            opt_hidden_units = h
            opt_epoch = np.argmin(val_loss_fin) + 1

    return opt_hidden_units, opt_epoch


def opt_aggregation(location, epochs, hidden_units, aggregation_days, scale_q=True, hist_years=10):
    """
    Function to fit and evaluate the model for a certain aggregation period
    :param location: location to do the optimization for
    :param scale_q: Use a standard scaler for the GW levels or not
    :param hidden_units: Number of hidden units of the network
    :param aggregation_days: Aggregation period (days) to test
    :param hist_years: Number of years in history
    :return: validation loss
    """

    o_folder = f'LSTM/{location}'

    # Create output folder
    if not os.path.exists(o_folder):
        os.makedirs(o_folder)

    # Load data
    df = pd.read_csv(f'../data/{location}/heads.csv', index_col=0, parse_dates=True)
    df_meteo = pd.read_csv(f'../data/{location}/input_data.csv', index_col=0, parse_dates=True)

    # Truncate to period where we can make predictions
    first_available_date = df_meteo.first_valid_index() + relativedelta(days=60) + relativedelta(days=12 * hist_years * 30)
    df = df.truncate(before=first_available_date)

    # USA has a slightly different naming
    if location == 'USA':
        rain_c = 'PRCP'
        ep_c = 'ET'
    else:
        rain_c = 'rr'
        ep_c = 'et'

    # Add recharge as explicit input
    df_meteo['voeding'] = df_meteo[rain_c] - df_meteo[ep_c]

    # Scale the data
    scaled_data = df.copy()

    if scale_q:
        scaler_q = StandardScaler()
        scaled_data['head'] = scaler_q.fit_transform(np.reshape(df['head'].values, (-1, 1)))
    else:
        scaled_data['head'] = df['head'] - df['head'].mean()

    scaler_10 = {}
    scaler_30 = {}

    for c in df_meteo.columns:
        # Not 100% fair. Should still apply the scaler only to the calibration period
        # Note scaler is not yet applied, only fitted. Will be applied in the next transform step
        scaler_10[c] = StandardScaler()
        scaler_10[c].fit(np.reshape(df_meteo.resample(f'{aggregation_days}D').sum()[c].dropna().values, (-1, 1)))
        scaler_30[c] = StandardScaler()
        scaler_30[c].fit(np.reshape(df_meteo.resample('30D').sum()[c].dropna().values, (-1, 1)))

    # this function fetches the needed data to input in the LSTM

    aantal_y_hist = 12 * hist_years + (60 / aggregation_days)

    def transform_data(subset='cal'):
        """
        Function to transform the data to what is needed for
        :param subset: string indicating if we need to prepare the calibration or the validation set
        :return: X (3D matrix) and Y (2D matrix) that can be fed into the model
        """
        if subset == 'cal':
            df_ = scaled_data.iloc[:int(df.shape[0] * 0.75), :]
        else:
            df_ = scaled_data.iloc[int(df.shape[0] * 0.75):, :]

        x_l = np.empty((df_.shape[0], int(aantal_y_hist), df_meteo.shape[1] + 1))

        for ii, i in enumerate(df_.index):

            istartd = i - relativedelta(days=60)
            istartm = istartd - relativedelta(days=12 * hist_years * 30)
            df_c_d = df_meteo.truncate(after=i, before=istartd + relativedelta(days=1)).resample(f'{aggregation_days}D').sum()
            df_c_m = df_meteo.truncate(after=istartd, before=istartm + relativedelta(days=1)).resample('30D').sum()
            for c in df_c_d:
                df_c_d[c] = scaler_10[c].transform(np.reshape(df_c_d[c].values, (-1, 1)))
                df_c_m[c] = scaler_30[c].transform(np.reshape(df_c_m[c].values, (-1, 1)))

            # dummy indicates temporal frequency
            df_c_m['dummy'] = 0
            df_c_d['dummy'] = 1
            df_c_ = pd.concat([df_c_m, df_c_d])

            x_l[ii, :, :] = df_c_.values

        y_l = df_.values

        return x_l, y_l

    # Prepare data for calibration and validation
    x_c_f, y_c_f = transform_data()
    x_v_f, y_v_f = transform_data(subset='val')

    d = 0.2
    lr = 1e-3

    val_loss_list = []
    # We use an ensemble of 10 models
    for it in range(10):
        # Define the model
        opt = Adam(learning_rate=lr)
        model = keras.Sequential(name='HydroML')
        model.add(layers.LSTM(hidden_units, input_shape=(x_c_f.shape[1], x_c_f.shape[-1])))
        model.add(layers.Dropout(d))
        model.add(layers.Dense(1))
        model.add(layers.Activation("linear"))
        model.compile(loss='mse', optimizer=opt)

        history = model.fit(x_c_f, y_c_f, verbose=0, epochs=epochs, batch_size=1024,
                            validation_data=(x_v_f, y_v_f))

        val_loss_list.append(np.expand_dims(np.array(history.history['val_loss']), -1))

        # Necessary to avoid memory issues
        del model
        del opt
        tf.keras.backend.clear_session()
        gc.collect()

    val_loss_fin = np.mean(np.concatenate(val_loss_list, axis=1), axis=1)

    return val_loss_fin


def optimize_params():
    """
    Function to optimize the hyperprarmeters of the LSTM model.
    Hyperparameters to be optimized are:
        * Number of epochs
        * Number of hidden units
        * Optimal aggregation of the most recent data
    Parameters are written to csv file so they can be used to build the final models
    Optimisation is done on a validation period (last 25% of the data)
    To limit calibration time we used a simple consecutive grid searches
    """

    df_l = []
    for c in ['Sweden_1', 'Sweden_2', 'Netherlands', 'Germany', 'USA']:
        # First get opt hidden units and epochs
        opt_hidden_units, opt_epoch = opt_params(c, hist_years=5)
        print(f'For {c} the optimal hidden units is {opt_hidden_units} and the number of epochs is {opt_epoch}')
        # Now fine tune the aggregation days
        min_loss = 9999
        opt_agg = 0
        for ag in [10, 6, 5, 3, 2, 1]:
            loss_list = opt_aggregation(c, opt_epoch, opt_hidden_units, ag, hist_years=5)
            if np.min(loss_list) < min_loss:
                min_loss = np.min(loss_list)
                opt_agg = ag
        print(f'For {c} the optimal aggregation is {opt_agg}')
        df_l.append({'location': c,
                     'hidden units': opt_hidden_units,
                     'epochs': opt_epoch,
                     'aggregation': opt_agg})

    # Write to csv file
    df = pd.DataFrame(df_l)
    df.to_csv('hyperparameters_v3.csv')


# First we optimize the hyperparameters
optimize_params()
