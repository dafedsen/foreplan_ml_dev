import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dask.distributed import Client
import logging

from prophet import Prophet
import math
from cuml.metrics import r2_score, mean_squared_error

import cudf as cd
import cupy as cp

logger = logging.getLogger(__name__)

# try:
#     client = Client("tcp://127.0.0.1:8790")
#     logger.info("Connected to Dask scheduler successfully.")
# except Exception as e:
#     logger.error("Failed to connect to Dask scheduler. Continuing without it.")
#     client = None

from scripts.connection import *
from scripts.functions import *

import time
from datetime import timedelta
# from dask import delayed

# @delayed
def run_model(dbase, dbset):
    logger.info("Linear Regression forecast running.")
    id_cust = get_id_cust_from_id_prj(dbase['id_prj'][0])
    id_prj = dbase['id_prj'][0]
    id_version = extract_number(dbase['version_name'][0])

    update_process_status(id_prj, id_version, 'RUNNING')

    t_forecast = get_forecast_time(dbase, dbset)    

    start_time = time.time()
    pred, err = run_prophet_forecast(dbase, t_forecast, dbset)
    end_time = time.time()

    logger.info("Sending Linear Regression forecast result.")
    send_process_result(pred, id_cust)

    logger.info("Sending Linear Regression forecast evaluation.")
    send_process_evaluation(err, id_cust)

    print(str(timedelta(seconds=end_time - start_time)))

    update_process_status(id_prj, id_version, 'SUCCESS')

    return str(timedelta(seconds=end_time - start_time))

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

        df_t = df.copy()
        df_t.rename(columns={'hist_date': 'ds', 'hist_value': 'y'}, inplace=True)
        df_t = df_t[['ds', 'y']]
        
        # Data Splitting
        train_size = 0.9
        df_train = df_t.sample(frac=train_size)
        df_test = df_t.drop(df_train.index)
        
        # Convert to pandas
        df_train = df_train.to_pandas()
        df_test = df_test.to_pandas()
        
        # Initiate Model and Train
        model = Prophet(yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=True,
                        growth='linear')
        
        model.fit(df_train)
        
        # Model Predict
        y_pred = t_forecast.copy()
        y_pred.rename(columns={'date': 'ds'}, inplace=True)
        y_pred = y_pred.to_pandas()
        y_pred = model.predict(y_pred)

        # Prediction Data Process
        pred['date'] = t_forecast['date']
        pred[level2] = y_pred['yhat']
        pred['level1'] = level1
        pred['adj_include'] = ADJUSTMENT
        pred['id_prj_prc'] = PROCESS
        pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]
        
    except Exception as e:
        print('ERROR EXCEPTION', e)
        pred['date'] = t_forecast['date']
        pred[level2] = 0
        pred['level1'] = level1
        pred['adj_include'] = ADJUSTMENT
        pred['id_prj_prc'] = PROCESS
        pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    try:
        # Evaluating
        y_pred_test = model.predict(df_test)

        eval = cd.merge(y_pred_test, df_test, on='ds')
        eval['error'] = eval['yhat'] - eval['y']

        rmse = cp.sqrt(mean_squared_error(eval['y'], eval['yhat']))
        r2 = r2_score(eval['y'], eval['yhat'])
        bias = cp.mean(eval['error'])
        mape = mean_absolute_percentage_error(eval['y'], eval['yhat'])

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
        print(f"An unexpected error occurred: {e}")

    return pred, err


def run_prophet_forecast(dbase, t_forecast, dbset):

    project = dbase['id_prj'][0]
    id_version = extract_number(dbset['version_name'][0])
    id_cust = get_id_cust_from_id_prj(project)

    # reset_model_finished(project, id_version)

    level1_list = dbase['level1'].unique().to_arrow().to_pylist()
    level1_list = sorted(level1_list)

    level2_list = dbase['level2'].unique().to_arrow().to_pylist()
    level2_list = sorted(level2_list)
    
    total_loop = len(level1_list) * len(level2_list)
    current_loop = 0

    lr_settings = dbset[dbset['model_name'] == 'Prophet Forecast']
    lr_settings = lr_settings[['adj_include', 'id_prj_prc','level1', 'level2', 'model_name', 'out_std_dev', 'ad_smooth_method']]

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
            df = dbase[(dbase['level1'] == level1) & (dbase['level2'] == level2)]
            df.reset_index(inplace=True, drop=True)

            st = lr_settings[(lr_settings['level1'] == level1) & (lr_settings['level2'] == level2)]
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
            
            current_loop = current_loop + 1
            progress = (current_loop / total_loop)
            # update_running_model_process(project, id_version, progress)

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
    forecast_result['id_model'] = 2

    forecast_result['date'] = cd.to_datetime(forecast_result['date'])
    forecast_result = forecast_result.groupby(['level1', 'level2', 'adj_include', 'id_prj_prc']).apply(
        lambda group: group.sort_values('date').head(t_forecast.shape[0])
    ).reset_index(drop=True)
    forecast_result = forecast_result[['date', 'id_prj_prc', 'level1', 'level2', 'hist_value', 'id_model']]
    forecast_result.rename(columns={'date': 'fcast_date', 'hist_value': 'fcast_value'}, inplace=True)
    forecast_result['partition_cust_id'] = id_cust

    error_result = error_result.melt(
        id_vars=['level1', 'adj_include','id_prj_prc', 'level2', 'id_prj', 'id_version'],
        value_vars=['rmse', 'r2', 'mape', 'bias'],
        var_name='err_method',
        value_name='err_value'
    )
    error_result['id_model'] = 2
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

    print(forecast_result)
    print(error_result)

    return forecast_result, error_result