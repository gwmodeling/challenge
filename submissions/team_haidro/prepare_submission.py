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


class MonteCarloDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def fit_model(location, epochs=20, hidden_units=60, aggregation_days=10, naam='first_iteration',
              hist_years=10, aggregation_hist=30):
    """
    Fit the model on all data. Save the resulting model and
    :param location: Location of the well
    :param epochs: number of epochs to use in training
    :param hidden_units: number of hidden units of the LSTM model
    :param aggregation_days: Number of days for aggregating the data
    :param naam: name of the model
    :param hist_years: Lookback period (in years)
    :param aggregation_hist: Number of days for aggregating in the further past
    :return: Scaler to be used in inference
    """

    # Create output folder
    o_folder = f'LSTM/{location}'

    if not os.path.exists(o_folder):
        os.makedirs(o_folder)

    # Load data
    df = pd.read_csv(f'../../data/{location}/heads.csv', index_col=0, parse_dates=True)
    df_meteo = pd.read_csv(f'../../data/{location}/input_data.csv', index_col=0, parse_dates=True)

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

    scaler_q = StandardScaler()
    scaled_data['head'] = scaler_q.fit_transform(np.reshape(df['head'].values, (-1, 1)))

    scaler_10 = {}
    scaler_30 = {}

    for c in df_meteo.columns:
        scaler_10[c] = StandardScaler()
        scaler_10[c].fit(np.reshape(df_meteo.resample(f'{aggregation_days}D').sum()[c].dropna().values, (-1, 1)))
        scaler_30[c] = StandardScaler()
        scaler_30[c].fit(np.reshape(df_meteo.resample('30D').sum()[c].dropna().values, (-1, 1)))

    aantal_y_hist = 12*hist_years * (30/aggregation_hist) + (60 / aggregation_days)

    def transform_data(subset='cal'):
        """
        Function to transform the data to what is needed for
        :param subset: string indicating if we need to prepare the calibration or the validation set
        :return: X (3D matrix) and Y (2D matrix) that can be fed into the model
        """
        if subset == 'cal':
            df_ = scaled_data.iloc[:int(df.shape[0] * 0.75), :]
        elif subset == 'all':
            df_ = scaled_data
        else:
            df_ = scaled_data.iloc[int(df.shape[0] * 0.75):, :]

        x_l = np.empty((df_.shape[0], int(aantal_y_hist), df_meteo.shape[1]+1))

        for ii, i in enumerate(df_.index):

            istartd = i - relativedelta(days=60)
            istartm = istartd - relativedelta(days=12*hist_years*30)
            df_c_d = df_meteo.truncate(after=i, before=istartd + relativedelta(days=1)).resample(f'{aggregation_days}D').sum()
            df_c_m = df_meteo.truncate(after=istartd, before=istartm + relativedelta(days=1)).resample(f'{aggregation_hist}D').sum()
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
    x_c_f, y_c_f = transform_data(subset='all')

    d = 0.2
    lr = 1e-3

    # We use an ensemble of 10 models
    for it in range(10):
        # Define the model
        opt = Adam(learning_rate=lr)
        model = keras.Sequential(name='HydroML')
        model.add(layers.LSTM(hidden_units, input_shape=(x_c_f.shape[1], x_c_f.shape[-1])))
        model.add(MonteCarloDropout(d))
        model.add(layers.Dense(1))
        model.add(layers.Activation("linear"))
        model.compile(loss='mse', optimizer=opt)

        history = model.fit(x_c_f, y_c_f, epochs=epochs, batch_size=1024)

        # Save the model
        model.save(os.path.join(o_folder, f'{naam}_{it}.h5'))

        # Necessary to avoid memory issues
        del model
        del opt
        tf.keras.backend.clear_session()
        gc.collect()

    return scaler_q


def forecast_gw(location, scaler_q, aggregation_days=10, naam='first_iteration', hist_years=10,
                aggregation_hist=30):
    """
    Function to get GW for submission
    :param location: location to do the optimization for
    :param scaler_q: Standard scaler necessary to inverse transform the data
    :param aggregation_days: Aggregation period (days)
    :param naam: nmae of the model
    :param hist_years: Number of years in history
    :param aggregation_hist: Aggregation period (days) of the further past
    """

    # Create output folder
    o_folder = f'LSTM/{location}'

    # Get the dates for submission
    df = pd.read_csv(f'submission_form_{location}.csv', index_col=0, parse_dates=True)

    # Read input data
    df_meteo = pd.read_csv(f'../../data/{location}/input_data.csv', index_col=0, parse_dates=True)

    # USA has a slightly different naming
    if location == 'USA':
        rain_c = 'PRCP'
        ep_c = 'ET'
    else:
        rain_c = 'rr'
        ep_c = 'et'

    # Add recharge as explicit input
    df_meteo['voeding'] = df_meteo[rain_c] - df_meteo[ep_c]

    scaled_data = df.copy()

    scaler_10 = {}
    scaler_30 = {}

    for c in df_meteo.columns:
        scaler_10[c] = StandardScaler()
        scaler_10[c].fit(np.reshape(df_meteo.resample(f'{aggregation_days}D').sum()[c].dropna().values, (-1, 1)))
        scaler_30[c] = StandardScaler()
        scaler_30[c].fit(np.reshape(df_meteo.resample('30D').sum()[c].dropna().values, (-1, 1)))

    aantal_y_hist = 12 * hist_years * (30 / aggregation_hist) + 60 / aggregation_days

    def transform_data():
        """
        Function to transform the data to what is needed for
        :return: X (3D matrix) that can be fed into the model
        """
        df_ = scaled_data

        x_l = np.empty((df_.shape[0], int(aantal_y_hist), df_meteo.shape[1] + 1))

        for ii, i in enumerate(df_.index):
            istartd = i - relativedelta(days=60)
            istartm = istartd - relativedelta(days=12 * hist_years * 30)
            df_c_d = df_meteo.truncate(after=i, before=istartd + relativedelta(days=1)).resample(f'{aggregation_days}D').sum()
            df_c_m = df_meteo.truncate(after=istartd, before=istartm + relativedelta(days=1)).resample(f'{aggregation_hist}D').sum()
            for c in df_c_d:
                df_c_d[c] = scaler_10[c].transform(np.reshape(df_c_d[c].values, (-1, 1)))
                df_c_m[c] = scaler_30[c].transform(np.reshape(df_c_m[c].values, (-1, 1)))
            df_c_m['dummy'] = 0
            df_c_d['dummy'] = 1
            df_c_ = pd.concat([df_c_m, df_c_d])

            # dummy indicates temporal frequency
            x_l[ii, :, :] = df_c_.values

        return x_l

    x_c_f = transform_data()

    ycp = []

    for it in range(10):
        # Load the model
        model = tf.keras.models.load_model(os.path.join(o_folder, f'{naam}_{it}.h5'),
                                           custom_objects={"MonteCarloDropout": MonteCarloDropout})
        for mc in range(250):
            # Fit 250 iterations for each model
            ycp.append(model.predict(x_c_f))

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Get the mean and quantiles of the 10x250 values
    scaled_data['Simulated Head'] = np.mean(np.concatenate(ycp, axis=1), axis=1)
    scaled_data['95% Lower Bound'] = np.quantile(np.concatenate(ycp, axis=1), 0.05, axis=1)
    scaled_data['95% Upper Bound'] = np.quantile(np.concatenate(ycp, axis=1), 0.95, axis=1)

    # Inverse transform to get proper inputs
    for c in scaled_data:
        scaled_data[c] = scaler_q.inverse_transform(np.reshape(scaled_data[c].values, (-1, 1)))

    # Write to output
    scaled_data.to_csv(f'submission_form_{location}.csv')


# This function trains the model and prepares the files for submission

# Load the optimized parameters
df_param = pd.read_csv('hyperparameters_v3.csv', index_col='location')
naam = 'opt_v3_MC'

df_l = []
# Iterate over all locations
for c in ['Netherlands', 'Germany', 'Sweden_1', 'Sweden_2', 'USA']:
    # Fit the model on all data
    scaler_q = fit_model(c, epochs=df_param.loc[c, 'epochs'], hidden_units=df_param.loc[c, 'hidden units'],
                         aggregation_days=df_param.loc[c, 'aggregation'], naam=naam, hist_years=5)

    # Prepare the submission
    forecast_gw(c, scaler_q, naam=naam, aggregation_days=df_param.loc[c, 'aggregation'],
                hist_years=5)
