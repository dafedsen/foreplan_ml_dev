import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from scripts.connection import *
from scripts.functions import *
from scripts.logger_ml import logging_ml

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

def run_model(id_user, dbase, dbset):
    try:
        logger.info("LSTNet Single forecast running.")
        id_prj = int(dbase['id_prj'].iloc[0].item())
        version_name = dbase['version_name'].iloc[0]

        id_cust = get_id_cust_from_id_prj(id_prj)
        id_version = extract_number(version_name)

        logging_ml(id_user, id_prj, id_version, id_cust, "LSTNet", "RUNNING", "Model is running", "lstnet_pytorch_single.py : run_model")

        t_forecast = get_forecast_time(dbase, dbset)    

        start_time = time.time()
        pred, err = run_cnn_lstm(dbase, t_forecast, dbset)
        end_time = time.time()

        logger.info("Sending LSTNet Single forecast result.")
        send_process_result(pred, id_cust)

        logger.info("Sending LSTNet Single forecast evaluation.")
        send_process_evaluation(err, id_cust)

        print(str(timedelta(seconds=end_time - start_time)))
        status = check_update_process_status_success(id_prj, id_version)

        if status:
            update_end_date(id_prj, id_version)

        logging_ml(id_user, id_prj, id_version, id_cust, "LSTNet", "FINISHED", "Finished running model", "lstnet_pytorch_single.py : run_model")

        return str(timedelta(seconds=end_time - start_time))
    
    except Exception as e:
        logger.error(f"Error in cnn_lstm_pytorch_single.run_model : {str(e)}")
        logging_ml(id_user, id_prj, id_version, id_cust, "LSTNet", "ERROR", "Error in running model", "lstnet_pytorch_single.py : run_model : " + str(e))
        update_process_status(id_prj, id_version, 'ERROR')

def predict_model(df, st, t_forecast):
    try:
        pred = cd.DataFrame()
        
        # Preparing Parameter
        level1 = df['level1'].iloc[0]
        level2 = df['level2'].iloc[0]
        PROCESS = int(st['id_prj_prc'].iloc[0].item())
        
        ADJUSTMENT = st['adj_include'].iloc[0]

        logger.info(f"Predicting {level1} - {level2} - {PROCESS} - {ADJUSTMENT}")
        # SIGMA = st['out_std_dev'][0]

        # SMOOTHING = st['ad_smooth_method'][0]

        # # Data Cleansing          
        # if ADJUSTMENT == 'Yes':
        #     adjusting_data(df, SMOOTHING, SIGMA)

        # cleansing_outliers(df, SIGMA)

        # Data Preparation
        df = df.sort_values(by='hist_date')
        # logger.info("Scaling data for LSTNet")
        scaler = MinMaxScaler()
        df['hist_value'] = scaler.fit_transform(df[['hist_value']])

        df_t = df[['hist_date', 'hist_value']].dropna()
        
        n_forecast = t_forecast.shape[0]
        n_lookback = 5

        total_n_loopback_required = n_lookback + n_forecast

        if len(df) < total_n_loopback_required:
            n_lookback = max(1, len(df) - n_forecast)
            logger.warning(f"[{level1} - {level2} - {PROCESS} - {ADJUSTMENT}] Adjusted n_lookback to {n_lookback} due to limited data")

        data_array = df_t.drop(columns=['hist_date']).to_cupy()

        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        # logger.info(f"Creating sequences for {level1} - {level2}")
        # X, Y = create_sequences(df_t.drop(columns=['hist_date']).to_cupy(), n_lookback, n_forecast)
        X, Y = create_sequences(data_array, n_lookback, n_forecast)
        # if X.shape[0] == 0 or Y.shape[0] == 0:
        #     logger.warning(f"[{level1} - {level2}] No sequences could be created. Skipping.")
        #     return cd.DataFrame(), cd.DataFrame()

        X = cp.asnumpy(X)
        Y = cp.asnumpy(Y)

        # Data Splitting
        # logger.info(f"Splitting data for {level1} - {level2}")
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

        logger.info(f"X_train shape: {X_train.shape}")
        train_dataset = TimeSeriesDataset(X_train, Y_train)
        test_dataset = TimeSeriesDataset(X_test, Y_test)

        logger.info(f"train_dataset length: {len(train_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initiate Model and Train
        input_size = X_train.shape[2]
        hidden_size = 128
        cnn_kernel_size = min(n_lookback, 3)
        output_size = n_forecast
        dropout = 0.1
        learning_rate = 0.001
        num_epochs = 50

        try:
            logger.info(f"Init Training model for {level1} - {level2} - {PROCESS} - {ADJUSTMENT}")
            model = LSTNet(
                input_size=input_size,
                hidden_size=hidden_size,
                cnn_kernel_size=cnn_kernel_size,
                skip_size=0,
                output_size=output_size,
                dropout=dropout,
                highway_window=5
            )
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model.to(DEVICE)
            model.train()
            for epoch in range(num_epochs):
                for X_batch, Y_batch in train_loader:
                    X_batch = X_batch.to(DEVICE)
                    Y_batch = Y_batch.to(DEVICE)
                    optimizer.zero_grad()
                    Y_pred = model(X_batch)

                    if torch.isnan(Y_pred).any():
                        raise ValueError("NaN detected in model output Y_pred")
                    if torch.isinf(Y_pred).any():
                        raise ValueError("Inf detected in model output Y_pred")
                    
                    loss = criterion(Y_pred, Y_batch)

                    if torch.isnan(loss):
                        raise ValueError("NaN detected in loss computation")

                    loss.backward()
                    optimizer.step()
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        except Exception as e:
            logger.error(f'ERROR EXCEPTION TRAINING MODEL LSTNet: {e}')

        # Model Predict
        try:
            logger.info(f"Predicting model for {level1} - {level2} - {PROCESS} - {ADJUSTMENT}")
            model.to(DEVICE)
            model.eval()
            last_data = df_t[['hist_value']].values[-n_lookback:]
            forecast_input = torch.tensor(last_data, dtype=torch.float32).to(DEVICE).unsqueeze(0)
            forecast_result = []

            for _ in range(t_forecast.shape[0]):
                y_pred = model(forecast_input).cpu().detach().numpy().flatten()
                if np.isnan(y_pred).any():
                    raise ValueError(f"NaN detected in model output during prediction at step {_}")
                forecast_result.append(y_pred[-1])

                # Update input for next step
                new_input = np.array([[y_pred[-1]]])
                forecast_input = torch.cat((forecast_input[:, 1:, :], torch.tensor(new_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)), dim=1)

            forecast_result = scaler.inverse_transform(np.array(forecast_result).reshape(-1, 1)).flatten()
            
            # logger.info(f"t_forecast length: {len(t_forecast)}, forecast_result length: {len(forecast_result)}")
            # Prediction Data Process
            pred['date'] = t_forecast['date']
            pred[level2] = forecast_result
            pred['level1'] = level1
            pred['adj_include'] = ADJUSTMENT
            pred['id_prj_prc'] = PROCESS
            pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]
            logger.info("\n%s", pred[["date", level2]])
        except Exception as e:
            logger.error(f'ERROR EXCEPTION PREDICTING MODEL LSTNet: {e}')
            pred['date'] = t_forecast['date']
            pred[level2] = 0
            pred['level1'] = level1
            pred['adj_include'] = ADJUSTMENT
            pred['id_prj_prc'] = PROCESS
            pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]

    except Exception as e:
        logger.error(f'ERROR EXCEPTION in predict_model LSTNet single {level1} - {level2} - {PROCESS} - {ADJUSTMENT}: {e}')
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
                X_batch = X_batch.to(DEVICE)
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
            mape = 999999999.0

        if math.isinf(r2) == True:
            r2 = 999999999.0

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
        logger.error(f'ERROR EXCEPTION in evaluate_model LSTNet single : {e}')
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

    logger.info("\n%s", pred[["date", level2]])
    logger.info("\n%s", err)

    return pred, err


def run_cnn_lstm(dbase, t_forecast, dbset):

    project = int(dbase['id_prj'].iloc[0].item())
    id_version = extract_number(dbset['version_name'].iloc[0])
    id_cust = get_id_cust_from_id_prj(project)

    level1_list = dbase['level1'].unique().to_arrow().to_pylist()
    level1_list = sorted(level1_list)

    level2_list = dbase['level2'].unique().to_arrow().to_pylist()
    level2_list = sorted(level2_list)
    
    total_loop = len(level1_list) * len(level2_list)
    current_loop = 0

    lr_settings = dbset[dbset['model_name'] == 'LSTNet']
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

            df_y_adj = df[df['flag_adj'] == 1]
            df_n_adj = df[df['flag_adj'] == 0]

            # Run Adjusted Data
            y_pred, y_err = predict_model(df_y_adj, st_y_adj, t_forecast)
            level1_forecast = cd.merge(level1_forecast, y_pred, on=['date', 'level1', 'adj_include', 'id_prj_prc'], how='left')
            level1_error = cd.concat([level1_error, y_err], ignore_index=True)

            # Run Unadjusted Data
            n_pred, n_err = predict_model(df_n_adj, st_n_adj, t_forecast)
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
    forecast_result['id_model'] = 10

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
    error_result['id_model'] = 10
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

def validate_dataframe(df, required_columns):
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        if df[col].isnull().any():
            raise ValueError(f"Column '{col}' must not contain nulls.")
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

        if torch.isnan(self.X).any():
            logger.warning("WARNING: NaN detected in X inside TimeSeriesDataset")
        if torch.isnan(self.Y).any():
            logger.warning("WARNING: NaN detected in Y inside TimeSeriesDataset")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class LSTNet(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_kernel_size, skip_size, output_size, dropout=0.2, highway_window=0):
        super(LSTNet, self).__init__()
        self.P = 30  # Time window (lookback)
        self.m = input_size  # Number of input features (variables)
        self.hidR = hidden_size
        self.hidC = hidden_size
        self.Ck = cnn_kernel_size
        self.skip = skip_size
        self.output_size = output_size
        self.highway_window = highway_window

        # CNN component
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))

        # RNN component
        self.gru = nn.GRU(self.hidC, self.hidR)

        # Skip-RNN component
        if self.skip > 0:
            self.pt = int((self.P - self.Ck) / self.skip)
            self.skip_gru = nn.GRU(self.hidC, self.hidR)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidR, self.output_size)
        else:
            self.linear1 = nn.Linear(self.hidR, self.output_size)

        # Highway component
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window * self.m, self.output_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (B, 1, P, m)
        c = torch.relu(self.conv1(x))  # (B, hidC, P-Ck+1, 1)
        c = self.dropout(c)
        c = c.squeeze(3)  # (B, hidC, P-Ck+1)
        c = c.permute(2, 0, 1)  # (P-Ck+1, B, hidC)

        # GRU
        r, _ = self.gru(c)  # (P-Ck+1, B, hidR)
        r = r[-1, :, :]  # (B, hidR)

        if self.skip > 0:
            s = c.view((int((c.size(0) - 1) / self.skip), self.skip, c.size(1), c.size(2)))
            s = s.permute(1, 2, 0, 3).contiguous().view(self.skip, s.size(2), -1)
            s, _ = self.skip_gru(s)
            s = s[-1, :, :]
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        if self.highway_window > 0:
            z = x.squeeze(1)  # (B, P, m)
            z = z[:, -self.highway_window:, :].contiguous().view(batch_size, -1)
            highway = self.highway(z)
            res += highway

        return res
