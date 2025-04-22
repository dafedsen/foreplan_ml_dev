import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from scripts.connection import *
from scripts.functions import *

import math
import cudf as cd
import numpy as np
from datetime import timedelta
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from cuml.preprocessing import MinMaxScaler, LabelEncoder
from cuml.metrics import r2_score, mean_squared_error
from cuml.model_selection import train_test_split

logger = logging.getLogger(__name__)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def run_model(dbase, dbset):
    try:
        logger.info("CNN LSTM Single forecast running.")
        id_cust = get_id_cust_from_id_prj(dbase['id_prj'][0])
        id_prj = dbase['id_prj'][0]
        id_version = extract_number(dbase['version_name'][0])

        t_forecast = get_forecast_time(dbase, dbset)    

        start_time = time.time()
        pred, err = run_cnn_lstm(dbase, t_forecast, dbset)
        end_time = time.time()

        logger.info("Sending CNN LSTM Single forecast result.")
        send_process_result(pred, id_cust)

        logger.info("Sending CNN LSTM Single forecast evaluation.")
        send_process_evaluation(err, id_cust)

        print(str(timedelta(seconds=end_time - start_time)))
        status = check_update_process_status_success(id_prj, id_version)

        if status:
            update_end_date(id_prj, id_version)

        logger.info("CNN LSTM Single Finished.")

        return str(timedelta(seconds=end_time - start_time))
    
    except Exception as e:
        logger.error(f"Error in cnn_lstm_pytorch_single.run_model : {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def predict_model(df, st, t_forecast):

    try:
        pred = cd.DataFrame()
        
        # Preparing Parameter
        level1 = df['level1'][0]
        level2 = df['level2'][0]

        PROCESS = st['id_prj_prc'][0]
        
        ADJUSTMENT = st['adj_include'][0]

        SIGMA = st['out_std_dev'][0]

        SMOOTHING = st['ad_smooth_method'][0]

        # Data Cleansing          
        if ADJUSTMENT == 'Yes':
            adjusting_data(df, SMOOTHING, SIGMA)

        cleansing_outliers(df, SIGMA)

        # Data Preparation
        df = df.sort_values(by='hist_date')

        scaler = MinMaxScaler()
        df['hist_value'] = scaler.fit_transform(df[['hist_value']])
        df_t = df.copy()
        df_t = df_t[['hist_date', 'hist_value']]
        
        n_lookback = 30
        n_forecast = t_forecast.shape[0]

        X, Y = create_sequences(df_t.drop(columns=['hist_date']).to_cupy(), n_lookback, n_forecast)
        X = cp.asnumpy(X)
        Y = cp.asnumpy(Y)
        
        # Data Splitting
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

        train_dataset = TimeSeriesDataset(X_train, Y_train)
        test_dataset = TimeSeriesDataset(X_test, Y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initiate Model and Train
        input_size = X_train.shape[2]
        hidden_size = 128
        num_layers = 1
        output_size = n_forecast
        dropout = 0.1
        learning_rate = 0.001
        num_epochs = 50

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
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        except Exception as e:
            print('ERROR EXCEPTION TRAINING MODEL: ', e)

        # Model Predict
        try:
            model.to(DEVICE)
            model.eval()
            last_data = df_t[['hist_value']].values[-n_lookback:]
            forecast_input = torch.tensor(last_data, dtype=torch.float32).to(DEVICE).unsqueeze(0)

            forecast_result = []
            for _ in range(t_forecast.shape[0]):
                y_pred = model(forecast_input).cpu().detach().numpy().flatten()
                forecast_result.append(y_pred[-1])

                # Update input for next step
                new_input = np.array([[y_pred[-1]]])
                forecast_input = torch.cat((forecast_input[:, 1:, :], torch.tensor(new_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)), dim=1)

            forecast_result = scaler.inverse_transform(np.array(forecast_result).reshape(-1, 1)).flatten()
            print(forecast_result)
            
            # Prediction Data Process
            pred['date'] = t_forecast['date']
            pred[level2] = forecast_result
            pred['level1'] = level1
            pred['adj_include'] = ADJUSTMENT
            pred['id_prj_prc'] = PROCESS
            pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]

        except Exception as e:
            print('ERROR EXCEPTION PREDICTING MODEL: ', e)
            pred['date'] = t_forecast['date']
            pred[level2] = 0
            pred['level1'] = level1
            pred['adj_include'] = ADJUSTMENT
            pred['id_prj_prc'] = PROCESS
            pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]

    except Exception as e:
        print('ERROR EXCEPTION in predict_model', e)
        pred['date'] = t_forecast['date']
        pred[level2] = 0
        pred['level1'] = level1
        pred['adj_include'] = ADJUSTMENT
        pred['id_prj_prc'] = PROCESS
        pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]

    except Exception as e:
        print(f"An unexpected error occurred in predict_model: {e}")

    
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

        if math.isinf(mape) == True:
            mape = 999999999

        if math.isinf(r2) == True:
            r2 = 999999999

        # Evaluation Data Process
        err = cd.DataFrame({
            'rmse': [float(rmse)], 
            'r2': [float(r2)], 
            'bias': [float(bias)], 
            'mape': [float(mape)]
        })
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT
        err['id_prj_prc'] = PROCESS

    except Exception as e:
        print('ERROR EXCEPTION in evaluate_model', e)
        rmse = 999999999
        r2 = 999999999
        bias = 999999999
        mape = 999999999

        err = cd.DataFrame({
            'rmse': [float(rmse)], 
            'r2': [float(r2)], 
            'bias': [float(bias)], 
            'mape': [float(mape)]
        })
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT
        err['id_prj_prc'] = PROCESS

    except Exception as e:
        print(f"An unexpected error occurred in evaluate_model: {e}")

    return pred, err


def run_cnn_lstm(dbase, t_forecast, dbset):

    project = dbase['id_prj'][0]
    id_version = extract_number(dbset['version_name'][0])
    id_cust = get_id_cust_from_id_prj(project)

    level1_list = dbase['level1'].unique().to_arrow().to_pylist()
    level1_list = sorted(level1_list)

    level2_list = dbase['level2'].unique().to_arrow().to_pylist()
    level2_list = sorted(level2_list)
    
    total_loop = len(level1_list) * len(level2_list)
    current_loop = 0

    lr_settings = dbset[dbset['model_name'] == 'CNN LSTM']
    lr_settings = lr_settings[['adj_include', 'id_prj_prc', 'level1', 'level2', 'model_name', 'out_std_dev', 'ad_smooth_method']]

    id_prj_prc_y = lr_settings[lr_settings['adj_include'] == 'Yes']['id_prj_prc'].iloc[0]
    id_prj_prc_n = lr_settings[lr_settings['adj_include'] == 'No']['id_prj_prc'].iloc[0]

    forecast_result = cd.DataFrame()
    error_result = cd.DataFrame()

    # Looping level 1
    for level1 in level1_list:

        level1_forecast = cd.DataFrame(t_forecast['date'])
        level1_forecast['level1'] = level1
        level1_forecast['adj_include'] = 'Yes'
        level1_forecast['id_prj_prc'] = id_prj_prc_y

        level1_forecast_n = cd.DataFrame(t_forecast['date'])
        level1_forecast_n['level1'] = level1
        level1_forecast_n['adj_include'] = 'No'
        level1_forecast_n['id_prj_prc'] = id_prj_prc_n

        level1_error = cd.DataFrame()
        level1_error['level1'] = level1
        level1_error['adj_include'] = 'Yes'
        level1_error['id_prj_prc'] = id_prj_prc_y

        level1_error_n = cd.DataFrame()
        level1_error_n['level1'] = level1
        level1_error_n['adj_include'] = 'No'
        level1_error_n['id_prj_prc'] = id_prj_prc_n

        # Looping level 2
        for level2 in level2_list:
            if pd.isna(level2):
                update_model_finished(project, id_version, 1/total_loop)
                update_process_status_progress(project, id_version)
                continue

            df = dbase[(dbase['level1'] == level1) & (dbase['level2'] == level2)]
            if df.empty:
                print(f"Skipping {level1}-{level2}, no data found")
                update_model_finished(project, id_version, 1/total_loop)
                update_process_status_progress(project, id_version)
                continue
            df.reset_index(inplace=True, drop=True)

            st = lr_settings[
                (lr_settings['level1'] == level1) & 
                (lr_settings['level2'] == level2)
                ]
            st_y_adj = st[st['adj_include'] == 'Yes']
            st_n_adj = st[st['adj_include'] == 'No']

            st_y_adj.reset_index(inplace=True, drop=True)
            st_n_adj.reset_index(inplace=True, drop=True)

            # Run Adjusted Data
            y_pred, y_err = predict_model(df, st_y_adj, t_forecast)
            level1_forecast = cd.merge(level1_forecast, y_pred, on=['date', 'level1', 'adj_include', 'id_prj_prc'], how='left')
            level1_error = cd.concat([level1_error, y_err], ignore_index=True)

            # Run Unadjusted Data
            n_pred, n_err = predict_model(df, st_n_adj, t_forecast)
            level1_forecast_n = cd.merge(level1_forecast_n, n_pred, on=['date', 'level1', 'adj_include', 'id_prj_prc'], how='left')
            level1_error_n = cd.concat([level1_error_n, n_err], ignore_index=True)

            update_model_finished(project, id_version, 1/total_loop)
            update_process_status_progress(project, id_version)

        # Append Adjusted and Unadjusted
        forecast_result = cd.concat([forecast_result, level1_forecast])
        forecast_result = cd.concat([forecast_result, level1_forecast_n])

        error_result = cd.concat([error_result, level1_error])
        error_result = cd.concat([error_result, level1_error_n])

        forecast_result['id_prj'] = project
        forecast_result['id_version'] = id_version

        error_result['id_prj'] = project
        error_result['id_version'] = id_version

    forecast_result = forecast_result.melt(
        id_vars=['date', 'id_prj', 'id_version', 'level1', 'adj_include', 'id_prj_prc'], 
        var_name='level2', 
        value_name='hist_value'
        )
    forecast_result['id_model'] = 4

    forecast_result['date'] = cd.to_datetime(forecast_result['date'])
    forecast_result = forecast_result.groupby(['level1', 'level2', 'adj_include', 'id_prj_prc']).apply(
        lambda group: group.sort_values('date').head(t_forecast.shape[0])
    ).reset_index(drop=True)
    forecast_result = forecast_result[['date', 'id_prj_prc', 'level1', 'level2', 'hist_value', 'id_model']]
    forecast_result.rename(columns={'date': 'fcast_date', 'hist_value': 'fcast_value'}, inplace=True)
    forecast_result['partition_cust_id'] = id_cust
    forecast_result = forecast_result.dropna()
    forecast_result['fcast_value'] = forecast_result['fcast_value'].astype(float).round(3)

    error_result = error_result.melt(
        id_vars=['level1', 'adj_include', 'id_prj_prc', 'level2', 'id_prj', 'id_version'],
        value_vars=['rmse', 'r2', 'mape', 'bias'],
        var_name='err_method',
        value_name='err_value'
    )
    error_result['id_model'] = 4
    error_result['partition_cust_id'] = id_cust
    error_result = error_result.drop_duplicates()
    error_result.reset_index(drop=True, inplace=True)

    err_method_mapping = {
        'bias' : '1',
        'mape' : '2',
        'r2' : '3',
        'rmse' : '4',
    }
    error_result['id_err_method'] = error_result['err_method'].replace(err_method_mapping)
    error_result = error_result[['id_prj_prc', 'id_err_method', 'id_model', 'level1', 'level2', 'err_value', 'partition_cust_id']]
    error_result = error_result.dropna()
    error_result['err_value'] = error_result['err_value'].astype(float).round(3)
    # error_result['err_value'] = error_result['err_value'].map(lambda x: f"{x:.3f}")
    # error_result['err_value'] = error_result['err_value'].apply(lambda x: round(x, 3))

    return forecast_result, error_result

def create_sequences(data, n_lookback, n_forecast):
    X, Y = [], []
    for i in range(len(data) - n_lookback - n_forecast + 1):
        X.append(data[i:i + n_lookback, :])
        Y.append(data[i + n_lookback:i + n_lookback + n_forecast, 0])
    return cp.array(X), cp.array(Y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
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
        self.relu2 = nn.LeakyReLU()
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