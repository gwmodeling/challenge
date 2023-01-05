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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


class MonteCarloDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def fit_model(location, epochs=20, hidden_units=60, aggregation_days=10, naam='first_iteration',
              hist_years=10, aggregation_hist=30):

    o_folder = f'LSTM/{location}'

    if not os.path.exists(o_folder):
        os.makedirs(o_folder)

    df = pd.read_csv(f'../data/{location}/heads.csv', index_col=0, parse_dates=True)
    df_meteo = pd.read_csv(f'../data/{location}/input_data.csv', index_col=0, parse_dates=True)

    first_available_date = df_meteo.first_valid_index() + relativedelta(days=60) + relativedelta(days=12*hist_years*30)
    df = df.truncate(before=first_available_date)

    if location == 'USA':
        rain_c = 'PRCP'
        ep_c = 'ET'
    else:
        rain_c = 'rr'
        ep_c = 'et'

    df_meteo['voeding'] = df_meteo[rain_c] - df_meteo[ep_c]

    scaled_data = df.copy()


    scaler_q = StandardScaler()
    scaled_data['head'] = scaler_q.fit_transform(np.reshape(df['head'].values, (-1, 1)))


    scaler_10 = {}
    scaler_30 = {}

    for c in df_meteo.columns:
        # Still split between calibration and validation
        scaler_10[c] = StandardScaler()
        scaler_10[c].fit(np.reshape(df_meteo.resample(f'{aggregation_days}D').sum()[c].dropna().values, (-1, 1)))
        scaler_30[c] = StandardScaler()
        scaler_30[c].fit(np.reshape(df_meteo.resample('30D').sum()[c].dropna().values, (-1, 1)))

    # this function fetches the needed data to input in the LSTM

    aantal_y_hist = 12*hist_years * (30/aggregation_hist) + (60 / aggregation_days)

    def transform_data(subset='cal'):
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
            df_c_m['dummy'] = 0
            df_c_d['dummy'] = 1
            df_c_ = pd.concat([df_c_m, df_c_d])

            x_l[ii, :, :] = df_c_.values

        y_l = df_.values

        return x_l, y_l

    x_c_f, y_c_f = transform_data(subset='all')

    d = 0.2
    lr = 1e-3

    for it in range(10):
        opt = Adam(learning_rate=lr)
        model = keras.Sequential(name='HydroML')
        model.add(layers.LSTM(hidden_units, input_shape=(x_c_f.shape[1], x_c_f.shape[-1])))
        model.add(MonteCarloDropout(d))
        model.add(layers.Dense(1))
        model.add(layers.Activation("linear"))
        model.compile(loss='mse', optimizer=opt)

        history = model.fit(x_c_f, y_c_f, epochs=epochs, batch_size=1024)

        model.save(os.path.join(o_folder, f'{naam}_{it}.h5'))

        del model
        del opt

        tf.keras.backend.clear_session()

        gc.collect()

    return scaler_q


def forecast_gw(location, scaler_q, aggregation_days=10, naam='first_iteration', hist_years=10,
                aggregation_hist=30):

    o_folder = f'LSTM/{location}'

    df = pd.read_csv(f'../submissions/team_haidro/submission_form_{location}.csv', index_col=0, parse_dates=True)

    df_meteo = pd.read_csv(f'../data/{location}/input_data.csv', index_col=0, parse_dates=True)

    if location == 'USA':
        rain_c = 'PRCP'
        ep_c = 'ET'
    else:
        rain_c = 'rr'
        ep_c = 'et'

    df_meteo['voeding'] = df_meteo[rain_c] - df_meteo[ep_c]

    scaled_data = df.copy()

    scaler_10 = {}
    scaler_30 = {}

    for c in df_meteo.columns:
        # Still split between calibration and validation
        scaler_10[c] = StandardScaler()
        scaler_10[c].fit(np.reshape(df_meteo.resample(f'{aggregation_days}D').sum()[c].dropna().values, (-1, 1)))
        scaler_30[c] = StandardScaler()
        scaler_30[c].fit(np.reshape(df_meteo.resample('30D').sum()[c].dropna().values, (-1, 1)))

    aantal_y_hist = 12 * hist_years * (30 / aggregation_hist) + 60 / aggregation_days

    def transform_data():

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

            x_l[ii, :, :] = df_c_.values

        return x_l

    x_c_f = transform_data()

    ycp = []

    for it in range(10):
        model = tf.keras.models.load_model(os.path.join(o_folder, f'{naam}_{it}.h5'),
                                           custom_objects={"MonteCarloDropout": MonteCarloDropout})
        for mc in range(100):
            ycp.append(model.predict(x_c_f))

        del model

        tf.keras.backend.clear_session()

        gc.collect()

    scaled_data['Simulated Head'] = np.mean(np.concatenate(ycp, axis=1), axis=1)
    scaled_data['95% Lower Bound'] = np.quantile(np.concatenate(ycp, axis=1), 0.05, axis=1)
    scaled_data['95% Upper Bound'] = np.quantile(np.concatenate(ycp, axis=1), 0.95, axis=1)

    for c in scaled_data:
        scaled_data[c] = scaler_q.inverse_transform(np.reshape(scaled_data[c].values, (-1, 1)))

    scaled_data.to_csv(f'../submissions/team_haidro/submission_form_{location}.csv')


# This function evaluates the models on the validation period. This should give an idea on the expected model
# performance

# Load the optimized parameters
df_param = pd.read_csv('hyperparameters_v3.csv', index_col='location')
naam = 'opt_v3_MC'

df_l = []
# Iterate over all locations
for c in ['Netherlands', 'Germany', 'Sweden_1', 'Sweden_2', 'USA']:
    # Fit the model
    scaler_q = fit_model(c, epochs=df_param.loc[c, 'epochs'], hidden_units=df_param.loc[c, 'hidden units'],
                         aggregation_days=df_param.loc[c, 'aggregation'], naam=naam, hist_years=5)

    # Prepare the submission
    forecast_gw(c, scaler_q, naam=naam, aggregation_days=df_param.loc[c, 'aggregation'],
                hist_years=5)
