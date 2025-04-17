import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from cuml.preprocessing import MinMaxScaler, LabelEncoder
from cuml.metrics import r2_score, mean_squared_error

import cudf as cd
import cupy as cp
import numpy as np
import pandas as pd
import math
from datetime import timedelta
import time

import logging

logger = logging.getLogger(__name__)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from scripts.connection import *
from scripts.functions import *

def run_model(dbase, dbset):
    try:
        logger.info("CNN LSTM forecast running.")
        id_cust = get_id_cust_from_id_prj(dbase['id_prj'][0])
        id_prj = dbase['id_prj'][0]
        id_version = extract_number(dbase['version_name'][0])

        
        t_forecast = get_forecast_time(dbase, dbset)    

        start_time = time.time()
        pred, err = run_cnn_lstm(dbase, t_forecast, dbset)
        end_time = time.time()

        update_model_finished(id_prj, id_version, 1)
        update_process_status_progress(id_prj, id_version)

        logger.info("Sending CNN LSTM forecast result.")
        send_process_result(pred, id_cust)

        logger.info("Sending CNN LSTM forecast evaluation.")
        send_process_evaluation(err, id_cust)

        print(str(timedelta(seconds=end_time - start_time)))
        status = check_update_process_status_success(id_prj, id_version)

        if status:
            update_end_date(id_prj, id_version)

        return str(timedelta(seconds=end_time - start_time))
    
    except Exception as e:
        logger.error(f"Error in cnn_lstm_pytorch.run_model : {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def create_sequences(data, n_lookback, n_forecast):
    X, Y = [], []
    for i in range(len(data) - n_lookback - n_forecast + 1):
        X.append(data[i:i + n_lookback, :])
        Y.append(data[i + n_lookback:i + n_lookback + n_forecast, 0])
    return cp.array(X), cp.array(Y)

def run_cnn_lstm(dbase, t_forecast, dbset):

    project = dbase['id_prj'][0]
    id_version = extract_number(dbset['version_name'][0])
    id_cust = get_id_cust_from_id_prj(project)

    prc_settings = dbset[dbset['model_name'] == 'CNN LSTM']
    prc_settings = prc_settings[['adj_include', 'id_prj_prc', 'level1', 'level2', 'model_name', 'out_std_dev', 'ad_smooth_method']]

    id_prj_prc_y = prc_settings[prc_settings['adj_include'] == 'Yes']['id_prj_prc'].iloc[0]
    id_prj_prc_n = prc_settings[prc_settings['adj_include'] == 'No']['id_prj_prc'].iloc[0]

    df = dbase.drop(columns=['id_prj', 'version_name'])
    df['hist_date'] = cd.to_datetime(df['hist_date'])

    dbase_N = dbase.copy()
    dbase_Y = dbase.copy()

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

    df_base = cd.concat([dbase_Y, dbase_N], ignore_index=True)
    df = df_base[['level1', 'level2', 'flag_adj', 'hist_date', 'hist_value']]
    df = df.sort_values(by=['level1', 'level2', 'flag_adj', 'hist_date'])

    scaler = MinMaxScaler()
    df['hist_value'] = scaler.fit_transform(df[['hist_value']])

    encoders = {}
    for col in ['level1', 'level2', 'flag_adj']:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

    n_lookback = 30
    n_forecast = t_forecast.shape[0]

    X, Y = create_sequences(df.drop(columns=['hist_date']).to_cupy(), n_lookback, n_forecast)
    X = cp.asnumpy(X)
    Y = cp.asnumpy(Y)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    train_dataset = TimeSeriesDataset(X_train, Y_train)
    test_dataset = TimeSeriesDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X_train.shape[2]
    hidden_size = 128
    num_layers = 2
    output_size = n_forecast
    dropout = 0.1
    learning_rate = 0.001
    num_epochs = 20
    try:
        model = AdvancedLSTM(input_size, hidden_size, num_layers, output_size, dropout, bidirectional=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_batch)
                loss.backward()
                optimizer.step()
            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    except Exception as e:
        print('ERROR EXCEPTION TRAINING MODEL: ', e)

    try:
        model.to(DEVICE)
        model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)  # Move input to the same device as model
                Y_batch = Y_batch.to(DEVICE)
                y_pred = model(X_batch).squeeze()
                predictions.extend(y_pred.cpu().numpy())
                actuals.extend(Y_batch.cpu().numpy())

        # Convert to NumPy arrays
        predictions = cp.array(predictions).reshape(-1, 1)
        actuals = cp.array(actuals).reshape(-1, 1)

        # Mismatch fix
        dummy_features = cp.zeros((predictions.shape[0], scaler.n_features_in_ - 1))
        predictions_expanded = cp.hstack([dummy_features, predictions])

        dummy_features = cp.zeros((actuals.shape[0], scaler.n_features_in_ - 1))
        actuals_expanded = cp.hstack([dummy_features, actuals])

        # Inverse transform to original scale
        predictions = scaler.inverse_transform(predictions_expanded)
        predictions = predictions[:, -1]

        actuals = scaler.inverse_transform(actuals_expanded)
        actuals = actuals[:, -1]

        # Compute Metrics
        rmse = cp.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        bias = (cp.sum(predictions - actuals) / cp.sum(actuals)) * 100
        mape = cp.mean(cp.abs((predictions - actuals) / actuals)) * 100
    
    except Exception as e:
        print('ERROR EXCEPTION EVALUATING MODEL: ', e)
        rmse = 999999999
        r2 = 999999999
        bias = 999999999
        mape = 999999999

    try:
        forecast_results = []
        evaluation_results = []

        st = dbset.to_pandas()
        time_type = st['fcast_type'][0]
        time_freq = get_time_freq_by_type(time_type)

        for (level1, level2, flag_adj), group_df in df.groupby(['level1', 'level2', 'flag_adj']):
            print(f"Forecasting for {level1} - {level2} - {flag_adj}")

            # Get last LOOKBACK days for the group
            last_data = group_df[['hist_value', 'level1', 'level2', 'flag_adj']].values[-n_lookback:]
            forecast_input = torch.tensor(last_data, dtype=torch.float32).to(DEVICE).unsqueeze(0)

            # Generate Forecast
            forecast_result = []
            for _ in range(t_forecast.shape[0]):
                y_pred = model(forecast_input).cpu().detach().numpy().flatten()
                forecast_result.append(y_pred[-1])

                # Update input for next step
                new_input = np.array([[y_pred[-1], level1, level2, flag_adj]])
                forecast_input = torch.cat((forecast_input[:, 1:, :], torch.tensor(new_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)), dim=1)

            # Convert forecast back to original scale
            forecast_result = scaler.inverse_transform(np.array(forecast_result).reshape(-1, 1)).flatten()

            # Store Forecast
            future_dates = pd.date_range(start=group_df['hist_date'].max(), periods=n_forecast+1, freq=time_freq)[1:]
            forecast_df = pd.DataFrame({'date': future_dates, 'level1': encoders['level1'].inverse_transform([level1])[0], 
                                        'level2': encoders['level2'].inverse_transform([level2])[0], 
                                        'flag_adj': encoders['flag_adj'].inverse_transform([flag_adj])[0], 
                                        'forecast': forecast_result})
            forecast_results.append(forecast_df)

            evaluation_df = pd.DataFrame({
                'rmse': [float(rmse)], 'r2': [float(r2)], 'bias': [float(bias)], 'mape': [float(mape)],
                'level1': encoders['level1'].inverse_transform([level1])[0], 
                'level2': encoders['level2'].inverse_transform([level2])[0],
                'flag_adj': encoders['flag_adj'].inverse_transform([flag_adj])[0]
            })
            evaluation_results.append(evaluation_df)

        # Combine Forecast Results
        final_forecast = pd.concat(forecast_results)
        final_forecast['id_prj_prc'] = final_forecast['flag_adj'].map({
            'Y' : id_prj_prc_y,
            'N' : id_prj_prc_n
        })
        final_forecast['id_model'] = 4
        final_forecast['partition_cust_id'] = id_cust
        final_forecast.rename(columns={'date': 'fcast_date', 'forecast': 'fcast_value'}, inplace=True)
        final_forecast = final_forecast[['id_prj_prc', 'id_model', 'partition_cust_id', 'level1', 'level2', 'fcast_date', 'fcast_value']]
        final_forecast['fcast_value'] = final_forecast['fcast_value'].astype(float).round(3)
        final_forecast = final_forecast.dropna()
        pred = cd.from_pandas(final_forecast)

        final_evaluation = pd.concat(evaluation_results)
        final_evaluation['id_prj_prc'] = final_evaluation['flag_adj'].map({
            'Y' : id_prj_prc_y,
            'N' : id_prj_prc_n
        })
        final_evaluation = final_evaluation.drop(columns=['flag_adj'])
        final_evaluation = final_evaluation.melt(
            id_vars=['id_prj_prc', 'level1', 'level2'],
            value_vars=['rmse', 'r2', 'bias', 'mape'],
            var_name='err_method',
            value_name='err_value'
        )

        final_evaluation['id_model'] = 4
        final_evaluation['partition_cust_id'] = id_cust
        final_evaluation = final_evaluation.drop_duplicates()
        final_evaluation.reset_index(drop=True, inplace=True)
        err_method_mapping = {
            'bias' : '1',
            'mape' : '2',
            'r2' : '3',
            'rmse' : '4',
        }
        final_evaluation['id_err_method'] = final_evaluation['err_method'].replace(err_method_mapping)
        final_evaluation = final_evaluation[['id_prj_prc', 'id_err_method', 'id_model', 'level1', 'level2', 'err_value', 'partition_cust_id']]
        final_evaluation = final_evaluation.dropna()
        final_evaluation['err_value'] = final_evaluation['err_value'].astype(float).round(3)
        final_evaluation['err_value'] = final_evaluation['err_value'].apply(lambda x: round(x, 3))
        err = cd.from_pandas(final_evaluation)

    except Exception as e:
        print('ERROR EXCEPTION FORECASTING MODEL: ', e)

    return pred, err

def get_time_freq_by_type(time_type):
    time_type = str(time_type)

    if time_type == 'Daily':
       time_freq = 'D'
    elif time_type == 'Weekly':
       time_freq = 'W'
    elif time_type == 'Monthly':
       time_freq = 'M'
    elif time_type == 'Yearly':
       time_freq = 'Y'
    else:
       return 'Error Forecsat Time Setting'

    return time_freq

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
# class AdvancedLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3, bidirectional=True):
#         super(AdvancedLSTM, self).__init__()

#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         num_directions = 2 if bidirectional else 1

#         # LSTM Layer (Bidirectional + Multiple Layers)
#         self.lstm = nn.LSTM(
#             input_size,
#             hidden_size,
#             num_layers,
#             batch_first=True,
#             dropout=dropout,
#             bidirectional=bidirectional
#         )

#         # Batch Normalization Layer
#         self.batch_norm = nn.BatchNorm1d(hidden_size * num_directions)

#         # Fully Connected Layers with Activation & Dropout
#         self.fc1 = nn.Linear(hidden_size * num_directions, hidden_size)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         lstm_out, (hn, cn) = self.lstm(x)  # LSTM output & hidden states

#         # Get last hidden state (concatenate if bidirectional)
#         if self.bidirectional:
#             last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)  # Merge both directions
#         else:
#             last_hidden = hn[-1]

#         # Apply Batch Normalization
#         norm_hidden = self.batch_norm(last_hidden)

#         # Fully Connected Layers with Activation & Dropout
#         out = self.fc1(norm_hidden)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)

#         return out

class AdvancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3, bidirectional=True):
        super(AdvancedLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        # LSTM Layer (Bidirectional + Multiple Layers)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # Batch Normalization Layer
        self.batch_norm = nn.BatchNorm1d(hidden_size * num_directions)

        # Fully Connected Layers with Activation & Dropout
        self.fc1 = nn.Linear(hidden_size * num_directions, hidden_size * 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)

        if self.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]

        norm_hidden = self.batch_norm(last_hidden)

        out = self.fc1(norm_hidden)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)

        return out