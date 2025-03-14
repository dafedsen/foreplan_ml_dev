import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from cuml.preprocessing import MinMaxScaler
from cuml.metrics import r2_score, mean_squared_error

import cudf as cd
import cupy as cp
import numpy as np
import math
from datetime import timedelta
import time

import logging

logger = logging.getLogger(__name__)

from scripts.connection import *
from scripts.functions import *

def run_model(dbase, dbset):
    logger.info("CNN LSTM forecast running.")
    id_cust = get_id_cust_from_id_prj(dbase['id_prj'][0])
    id_prj = dbase['id_prj'][0]
    id_version = extract_number(dbase['version_name'][0])

    update_process_status(id_prj, id_version, 'RUNNING')

    t_forecast = get_forecast_time(dbase, dbset)    

    start_time = time.time()
    pred, err = run_cnn_lstm(dbase, t_forecast, dbset)
    end_time = time.time()

    logger.info("Sending CNN LSTM forecast result.")
    send_process_result(pred, id_cust)

    logger.info("Sending CNN LSTM forecast evaluation.")
    send_process_evaluation(err, id_cust)

    # update_process_status(id_prj, id_version, 'SUCCESS')
    print(str(timedelta(seconds=end_time - start_time)))

    return str(timedelta(seconds=end_time - start_time))

def create_sequences(data, n_lookback, n_forecast):
    X, Y = [], []
    for i in range(len(data) - n_lookback - n_forecast + 1):
        X.append(data[i:i + n_lookback, :])
        Y.append(data[i + n_lookback:i + n_lookback + n_forecast, 0])
    return cp.array(X), cp.array(Y)

def run_cnn_lstm(dbase, t_forecast, dbset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project = dbase['id_prj'][0]
    version = dbase['version_name'][0]

    level1_list = dbase['level1'].unique().to_arrow().to_pylist()
    level1_list = sorted(level1_list)

    level2_list = dbase['level2'].unique().to_arrow().to_pylist()
    level2_list = sorted(level2_list)

    prc_settings = dbset[dbset['model_name'] == 'CNN LSTM']
    prc_settings = prc_settings[['adj_include', 'level1', 'level2', 'model_name', 'out_std_dev', 'ad_smooth_method']]

    dbase_N = dbase.copy()
    #dbase_N['flag_adj'] = 'N'

    dbase_Y = dbase.copy()
    #dbase_Y['flag_adj'] = 'Y'

    for (level1, level2), group_data in dbase.groupby(['level1', 'level2']):
        # Select data for the current group
        prc_mask_y = (prc_settings['level1'] == level1) & (prc_settings['level2'] == level2) & (prc_settings['adj_include'] == 'Yes')
        prc_mask_n = (prc_settings['level1'] == level1) & (prc_settings['level2'] == level2) & (prc_settings['adj_include'] == 'No')
        
        prc_y = prc_settings.loc[prc_mask_y]
        prc_n = prc_settings.loc[prc_mask_n]

        ADJUSTMENT_Y = prc_y['ad_smooth_method'].iloc[0]

        SIGMA_Y = prc_y['out_std_dev'].iloc[0]
        SIGMA_N = prc_n['out_std_dev'].iloc[0]

        mask = (dbase['level1'] == level1) & (dbase['level2'] == level2)
        group_df = dbase.loc[mask]

        # Preprocessing steps on 'new_value'
        group_df_y = group_df.copy()  # Avoid SettingWithCopyWarning
        group_df_n = group_df.copy()

        adjusting_data(group_df_y, ADJUSTMENT_Y, SIGMA_Y)
        cleansing_outliers(group_df_y, SIGMA_Y)
        cleansing_outliers(group_df_n, SIGMA_N)

        # Update 'new_value' in the original DataFrame
        dbase_Y.loc[mask, 'hist_value'] = group_df_y['hist_value']
        dbase_N.loc[mask, 'hist_value'] = group_df_n['hist_value']

    dbase_Y['flag_adj'] = 'Y'
    dbase_N['flag_adj'] = 'N'

    df_train = pd.concat([dbase_N, dbase_Y], ignore_index=True)
    pivot_data = df_train.pivot_table(
        index=['hist_date'],
        columns=['level1', 'level2', 'flag_adj'],
        values='hist_value'
    ).reset_index()

    print(pivot_data)
    print(type(pivot_data))
    pivot_data = pivot_data.to_pandas()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(pivot_data.iloc[:, 1:])

    print(scaled_data.shape)
    print(scaled_data)
    print(type(scaled_data))

    n_lookback = 30
    n_forecast = 1

    train_size = int(len(scaled_data) * 0.9)
    train_data = scaled_data[:train_size, :]
    test_data = scaled_data[train_size - n_lookback:, :]

    # Prepare training sequences
    X_train, Y_train = [], []
    for i in range(n_lookback, train_size - n_forecast + 1):
        X_train.append(train_data[i - n_lookback:i, :])
        Y_train.append(train_data[i:i + n_forecast, :])

    X_train = np.array(X_train).get()
    Y_train = np.array(Y_train).get()

    # Prepare test sequences
    X_test = []
    for i in range(n_lookback, len(test_data) - n_forecast + 1):
        X_test.append(test_data[i - n_lookback:i, :])

    X_test = np.array(X_test)
    actual_data = test_data[n_lookback:, :]

    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(Y_train.shape[2]))

    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    forecasts = []
    errors = []

    for col in range(scaled_data.shape[1]):  # Iterate over columns
        x_input = scaled_data[-n_lookback:, :].reshape(1, n_lookback, scaled_data.shape[1])  # Use the last lookback sequence
        future_forecast = []

        for _ in range(365):  # Predict for 365 days
            if isinstance(x_input, cp.ndarray):
                x_input = x_input.get()
                
            y_pred = model.predict(x_input, verbose=0)
            future_forecast.append(y_pred.flatten()[col])

            y_pred_new = np.zeros((1, 1, scaled_data.shape[1]))
            y_pred_new[0, 0, col] = y_pred.flatten()[col]  # Update only the target column
            x_input = np.append(x_input[:, 1:, :], y_pred_new, axis=1)

        # Recreate full feature set for inverse transform
        future_forecast = np.array(future_forecast)
        dummy_features = np.zeros((365, scaled_data.shape[1]))
        dummy_features[:, 0] = future_forecast

        # Inverse scale the forecast
        dummy_features = dummy_features.get()
        future_forecast_full = scaler.inverse_transform(dummy_features)
        future_forecast = future_forecast_full[:, col]

        print(dummy_features.shape)
        print('CHECK')
        print(test_data.shape)

        # Get actual values for comparison
        actual_values = scaler.inverse_transform(test_data)[:, col][-365:]

        n_actual = min(len(actual_values), len(future_forecast))
        actual_values = actual_values[:n_actual]
        future_forecast_test = future_forecast[:n_actual]

        # Calculate errors
        
        print(type(actual_values))
        print(type(future_forecast_test))
        
        rmse = cp.sqrt(mean_squared_error(actual_values, future_forecast_test))
        r2 = r2_score(actual_values, future_forecast_test)
        bias = (cp.sum(future_forecast_test - actual_values) / cp.sum(actual_values)) * 100
        
        actual_values = cp.asarray(actual_values)
        future_forecast_test = cp.asarray(future_forecast_test)
        print(type(actual_values))
        print(type(future_forecast_test))
        
        mape = cp.mean(cp.abs((future_forecast_test - actual_values) / actual_values)) * 100

        print(pivot_data.columns)
        col_name = pivot_data.columns[col + 1]  # Get column name (exclude date)
        print(col_name)
        level1, level2, flag_adj = col_name

        # Create a DataFrame for the forecast
        print(type(pivot_data['hist_date']))
        last_date = pivot_data['hist_date'].iloc[-1]
        forecast_dates = pandas.date_range(start=last_date + pandas.Timedelta(days=1), periods=365)
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'level1': level1,
            'level2': level2,
            'flag_adj': flag_adj,
            'forecast': future_forecast
        })
        forecasts.append(forecast_df)

        errors.append({
            'level1': level1,
            'level2': level2,
            'flag_adj': flag_adj,
            'rmse': rmse,
            'r2': r2,
            'bias': bias,
            'mape': mape
        })

    # Combine all forecasts
    final_forecasts = pd.concat(forecasts, ignore_index=True)
    error_metrics = pandas.DataFrame(errors)
    print(final_forecasts)
    print(error_metrics)
    #return final_forecasts, error_metrics