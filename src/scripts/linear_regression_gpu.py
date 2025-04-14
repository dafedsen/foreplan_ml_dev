import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from cuml.linear_model import LinearRegression
from cuml.model_selection import train_test_split
from cuml.metrics import r2_score
import math
import cudf as cd
# from dask import delayed

from datetime import timedelta
import time

# from dask.distributed import Client
import logging

logger = logging.getLogger(__name__)

from scripts.connection import *
from scripts.functions import *

# @delayed
def run_model(dbase, dbset):
    try:
        logger.info("Linear Regression forecast running.")
        id_cust = get_id_cust_from_id_prj(dbase['id_prj'][0])
        id_prj = dbase['id_prj'][0]
        id_version = extract_number(dbase['version_name'][0])

        t_forecast = get_forecast_time(dbase, dbset)    

        start_time = time.time()
        pred, err = run_linear_regression(dbase, t_forecast, dbset)
        end_time = time.time()

        logger.info("Sending Linear Regression forecast result.")
        send_process_result(pred, id_cust)

        logger.info("Sending Linear Regression forecast evaluation.")
        send_process_evaluation(err, id_cust)

        print(str(timedelta(seconds=end_time - start_time)))
        status = check_update_process_status_success(id_prj, id_version)

        if status:
            update_end_date(id_prj, id_version)

        return str(timedelta(seconds=end_time - start_time))
    
    except Exception as e:
        logger.error(f"Error in linear_regression_gpu.run_model : {str(e)}")
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
       
        df['hist_date'] = cd.to_datetime(df['hist_date'])
        df['hist_date'] = (df['hist_date'] - df['hist_date'].min()).dt.days
        
        X = df['hist_date'].values.reshape(-1, 1).get()
        y = df['hist_value'].values.get()
        
        # Data Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Initiate Model and Train
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Model Predict
        X_pred = get_weeks_by_number(t_forecast.shape[0] + X_test.shape[0])
        y_pred = model.predict(X_pred)

        # Prediction Data Process
        pred['date'] = t_forecast['date']
        pred[level2] = y_pred[X_test.shape[0]:]
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
        y_pred_test = model.predict(X_test)

        rmse = get_rmse(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        bias = get_bias(y_test, y_pred_test)
        mape = mean_absolute_percentage_error(y_test, y_pred_test)

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

def run_linear_regression(dbase, t_forecast, dbset):

    project = dbase['id_prj'][0]
    id_version = extract_number(dbset['version_name'][0])
    id_cust = get_id_cust_from_id_prj(project)

    level1_list = dbase['level1'].unique().to_arrow().to_pylist()
    level1_list = sorted(level1_list)

    level2_list = dbase['level2'].unique().to_arrow().to_pylist()
    level2_list = sorted(level2_list)
    
    total_loop = len(level1_list) * len(level2_list)

    lr_settings = dbset[dbset['model_name'] == 'Linear Regression']
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
    forecast_result['id_model'] = 3

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
        id_vars=['level1', 'adj_include','id_prj_prc', 'level2', 'id_prj', 'id_version'],
        value_vars=['rmse', 'r2', 'mape', 'bias'],
        var_name='err_method',
        value_name='err_value'
    )
    error_result['id_model'] = 3
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

    print(forecast_result)
    print(error_result)

    return forecast_result, error_result