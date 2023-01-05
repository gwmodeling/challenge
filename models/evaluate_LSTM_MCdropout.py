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
              scale_q=False, hist_years=10, aggregation_hist=30):

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

    if scale_q:
        scaler_q = StandardScaler()
        scaled_data['head'] = scaler_q.fit_transform(np.reshape(df['head'].values, (-1, 1)))
    else:
        scaled_data['head'] = df['head'] - df['head'].mean()

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

    x_c_f, y_c_f = transform_data()
    x_v_f, y_v_f = transform_data(subset='val')

    d = 0.2
    lr = 1e-3

    for it in range(10):
        opt = Adam(learning_rate=lr)
        model = keras.Sequential(name='HydroML')
        model.add(layers.LSTM(hidden_units, input_shape=(x_c_f.shape[1], x_c_f.shape[-1])))
        model.add(MonteCarloDropout(d))
        # model.add(layers.Dropout(d))
        model.add(layers.Dense(1))
        model.add(layers.Activation("linear"))
        model.compile(loss='mse', optimizer=opt)

        history = model.fit(x_c_f, y_c_f, epochs=epochs, batch_size=1024, validation_data=(x_v_f, y_v_f))

        fig, ax = plt.subplots()
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        plt.savefig(os.path.join(o_folder, f'loss_ens_{it}.png'))
        plt.close()

        model.save(os.path.join(o_folder, f'{naam}_{it}.h5'))

        del model
        del opt

        tf.keras.backend.clear_session()

        gc.collect()


def evaluate_model(location, plot=False, aggregation_days=10, naam='first_iteration', scale_q=False, hist_years=10,
                   aggregation_hist=30):

    o_folder = f'LSTM/{location}'

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

    if scale_q:
        scaler_q = StandardScaler()
        scaled_data['head'] = scaler_q.fit_transform(np.reshape(df['head'].values, (-1, 1)))
    else:
        av_head = df['head'].mean()
        scaled_data['head'] = df['head'] - df['head'].mean()

    scaler_10 = {}
    scaler_30 = {}

    for c in df_meteo.columns:
        # Still split between calibration and validation
        scaler_10[c] = StandardScaler()
        scaler_10[c].fit(np.reshape(df_meteo.resample(f'{aggregation_days}D').sum()[c].dropna().values, (-1, 1)))
        scaler_30[c] = StandardScaler()
        scaler_30[c].fit(np.reshape(df_meteo.resample('30D').sum()[c].dropna().values, (-1, 1)))

    aantal_y_hist = 12 * hist_years * (30 / aggregation_hist) + 60 / aggregation_days

    def transform_data(subset='cal'):
        if subset == 'cal':
            df_ = scaled_data.iloc[:int(df.shape[0] * 0.75), :]
        else:
            df_ = scaled_data.iloc[int(df.shape[0] * 0.75):, :]

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

        y_l = df_.values

        return x_l, y_l

    x_c_f, y_c_f = transform_data()
    x_v_f, y_v_f = transform_data(subset='val')

    ycp = []
    yvp = []

    for it in range(10):
        model = tf.keras.models.load_model(os.path.join(o_folder, f'{naam}_{it}.h5'),
                                           custom_objects={"MonteCarloDropout": MonteCarloDropout})
        for mc in range(250):
            ycp.append(model.predict(x_c_f))
            yvp.append(model.predict(x_v_f))

        del model

        tf.keras.backend.clear_session()

        gc.collect()

    scaled_data['pred'] = np.nan
    scaled_data.iloc[:int(df.shape[0] * 0.75), -1] = np.mean(np.concatenate(ycp, axis=1), axis=1)
    scaled_data.iloc[int(df.shape[0] * 0.75):, -1] = np.mean(np.concatenate(yvp, axis=1), axis=1)

    scaled_data['pred_q05'] = np.nan
    scaled_data.iloc[:int(df.shape[0] * 0.75), -1] = np.quantile(np.concatenate(ycp, axis=1), 0.05, axis=1)
    scaled_data.iloc[int(df.shape[0] * 0.75):, -1] = np.quantile(np.concatenate(yvp, axis=1),  0.05, axis=1)

    scaled_data['pred_q95'] = np.nan
    scaled_data.iloc[:int(df.shape[0] * 0.75), -1] = np.quantile(np.concatenate(ycp, axis=1), 0.95, axis=1)
    scaled_data.iloc[int(df.shape[0] * 0.75):, -1] = np.quantile(np.concatenate(yvp, axis=1),  0.95, axis=1)

    for c in scaled_data:
        if scale_q:
            scaled_data[c] = scaler_q.inverse_transform(np.reshape(scaled_data[c].values, (-1, 1)))
        else:
            scaled_data[c] = scaled_data[c] + av_head

    df_c = scaled_data.iloc[:int(df.shape[0] * 0.75), :]
    df_v = scaled_data.iloc[int(df.shape[0] * 0.75):, :]

    rmse_c = np.sqrt(mean_squared_error(df_c['head'].values, df_c['pred'].values))
    rmse_v = np.sqrt(mean_squared_error(df_v['head'].values, df_v['pred'].values))

    print(f'RMSE Calibration: {rmse_c}, RMSE Validation: {rmse_v}')

    if plot:
        fig, ax = plt.subplots(figsize=(13,5))
        ax.plot(df_c.index, df_c['head'].values, color='black')
        ax.plot(df_v.index, df_v['head'].values, color='black')
        ax.plot(df_c.index, df_c['pred'].values, color='steelblue')
        ax.plot(df_v.index, df_v['pred'].values, color='steelblue')
        ax.fill_between(df_v.index, df_v['pred_q05'].values, df_v['pred_q95'], color='steelblue', alpha=0.5)
        ax.fill_between(df_c.index, df_c['pred_q05'].values, df_c['pred_q95'], color='steelblue', alpha=0.5)
        ax.axvline(df_v.first_valid_index(), color='black', linestyle=':')
        ax.set_title(f'{location}: RMSE Calibration: {round(rmse_c, 2)}, RMSE Validation: {round(rmse_v, 2)}')
        plt.tight_layout()
        plt.savefig(os.path.join(o_folder, f'{naam}.png'))
        plt.show()
        plt.close()

    return rmse_c, rmse_v


# This function evaluates the models on the validation period. This should give an idea on the expected model
# performance

# Load the optimized parameters
df_param = pd.read_csv('hyperparameters_v3.csv', index_col='location')
naam = 'opt_v3_MC'

df_l = []
# Iterate over all locations
for c in ['Netherlands', 'Germany', 'Sweden_1', 'Sweden_2', 'USA']:
    # Fit the model
    fit_model(c, epochs=df_param.loc[c, 'epochs'], hidden_units=df_param.loc[c, 'hidden units'],
              aggregation_days=df_param.loc[c, 'aggregation'], naam=naam, scale_q=True, hist_years=5)

    # Evaluate the model
    rmse_c, rmse_v = evaluate_model(c, naam=naam, aggregation_days=df_param.loc[c, 'aggregation'], scale_q=True,
                                    plot=True, hist_years=5)

    df_l.append({'location': c,
                 'RMSE calibration': rmse_c,
                 'RMSE validation': rmse_v})

# Write results to csv file
df = pd.DataFrame(df_l)
df.to_csv(f'{naam}.csv')
