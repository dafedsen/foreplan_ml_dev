from cuml.linear_model import LinearRegression
from cuml.model_selection import train_test_split
from cuml.metrics import r2_score
import math
import cudf as cd

from datetime import timedelta
import time
import os

from scripts.connection import get_dataset, get_setting, get_forecast_time
from scripts.functions import *

base = os.getcwd()
data = base + r'/scripts/db_dummy/data-scenario1.csv'
sett = base + r'/scripts/db_dummy/sett-scenario1.csv'


def run_model(id_prj, version_name):
    

    dbase = get_dataset(data)
    dbset = get_setting(sett)
    t_forecast = get_forecast_time(data, sett)

    start_time = time.time()
    pred, err = predict_model(dbase, dbset, t_forecast)
    end_time = time.time()

    print('\n')
    print(pred)

    print('\n')
    print(err)

    return str(timedelta(seconds=end_time - start_time))

def predict_model(df, st, t_forecast):
    print(st)
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
        #df.sort_values(by = ['hist_date'], inplace = True)
        
        df['hist_date'] = cd.to_datetime(df['hist_date'])
        df['hist_date'] = (df['hist_date'] - df['hist_date'].min()).dt.days
        
        #.get() to array
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
        pred = pred[['adj_include', 'date', 'level1', level2]]

    except Exception as e:
        print('ERROR EXCEPTION', e)
        pred['date'] = t_forecast['date']
        pred[level2] = 0
        pred['level1'] = level1
        pred['adj_include'] = ADJUSTMENT
        pred = pred[['adj_include', 'date', 'level1', level2]]

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

        # Evaluation Data Process
        err = cd.DataFrame({'rmse': [rmse], 'r2': [r2], 'bias': [bias], 'mape': [mape]})
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT
    
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

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return pred, err

def run_linear_regression(dbase, t_forecast, dbset):

    project = dbase['id_prj'][0]
    version = dbase['version_name'][0]
    
    # .unique().to_list()
    level1_list = dbase['level1'].unique().to_arrow().to_pylist()
    level1_list = sorted(level1_list)

    level2_list = dbase['level2'].unique().to_arrow().to_pylist()
    level2_list = sorted(level2_list)
    
    total_loop = len(level1_list) * len(level2_list)
    current_loop = 0

    lr_settings = dbset[dbset['model_name'] == 'Linear Regression']
    lr_settings = lr_settings[['adj_include', 'id_prj_prc', 'level1', 'level2', 'model_name', 'out_std_dev', 'ad_smooth_method']]

    forecast_result = cd.DataFrame()
    error_result = cd.DataFrame()

    # Looping level 1
    for level1 in level1_list:

        level1_forecast = cd.DataFrame(t_forecast['date'])
        level1_forecast['level1'] = level1
        level1_forecast['adj_include'] = 'Yes'

        level1_forecast_n = cd.DataFrame(t_forecast['date'])
        level1_forecast_n['level1'] = level1
        level1_forecast_n['adj_include'] = 'No'

        level1_error = cd.DataFrame()
        level1_error['level1'] = level1
        level1_error['adj_include'] = 'Yes'

        level1_error_n = cd.DataFrame()
        level1_error_n['level1'] = level1
        level1_error_n['adj_include'] = 'No'

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
            level1_forecast = cd.merge(level1_forecast, y_pred, on=['date', 'level1', 'adj_include'], how='left')

            level1_error = cd.concat([level1_error, y_err], ignore_index=True)
            #level1_error = level1_error._append({
                #'level1' : y_err['level1'][0],
                #'level2' : y_err['level2'][0],
                #'adj_include' : y_err['adj_include'][0],
                #'rmse' : y_err['rmse'][0],
                #'r2' : y_err['r2'][0],
                #'mape' : y_err['mape'][0],
                #'bias' : y_err['bias'][0],
            #}, ignore_index=True)

            # Run Unadjusted Data
            n_pred, n_err = predict_model(df, st_n_adj, t_forecast)
            level1_forecast_n = cd.merge(level1_forecast_n, n_pred, on=['date', 'level1', 'adj_include'], how='left')

            level1_error_n = cd.concat([level1_error_n, n_err], ignore_index=True)
            #level1_error_n = level1_error_n._append({
                #'level1' : n_err['level1'][0],
                #'level2' : n_err['level2'][0],
                #'adj_include' : n_err['adj_include'][0],
                #'rmse' : n_err['rmse'][0],
                #'r2' : n_err['r2'][0],
                #'mape' : n_err['mape'][0],
                #'bias' : n_err['bias'][0],
            #}, ignore_index=True)
            current_loop = current_loop + 1
            progress = (current_loop / total_loop) * 100
            print_process_percentage(progress)
            

        # Append Adjusted and Unadjusted
        forecast_result = cd.concat([forecast_result, level1_forecast])
        forecast_result = cd.concat([forecast_result, level1_forecast_n])
        #forecast_result = forecast_result._append(level1_forecast)
        #forecast_result = forecast_result._append(level1_forecast_n)

        error_result = cd.concat([error_result, level1_error])
        error_result = cd.concat([error_result, level1_error_n])
        #error_result = error_result._append(level1_error)
        #error_result = error_result._append(level1_error_n)

        forecast_result['id_prj'] = project
        forecast_result['version_name'] = version

        error_result['id_prj'] = project
        error_result['version_name'] = version
    
    forecast_result = forecast_result.melt(
        id_vars=['date', 'id_prj', 'version_name', 'level1', 'adj_include'], 
        var_name='level2', 
        value_name='hist_value'
        )
    forecast_result['model_name'] = 'Linear Regression'

    forecast_result['date'] = cd.to_datetime(forecast_result['date'])
    forecast_result = forecast_result.groupby(['level1', 'level2', 'adj_include']).apply(
        lambda group: group.sort_values('date').head(t_forecast.shape[0])
    ).reset_index(drop=True)
    forecast_result = forecast_result[['date', 'id_prj', 'version_name', 'level1', 'adj_include', 'level2', 'hist_value', 'model_name']]

    error_result = error_result.melt(
        id_vars=['level1', 'adj_include', 'level2', 'id_prj', 'version_name'],
        value_vars=['rmse', 'r2', 'mape', 'bias'],
        var_name='err_method',
        value_name='err_value'
    )
    error_result['id_model'] = 3
    error_result = error_result.drop_duplicates()
    error_result.reset_index(drop=True, inplace=True)

    print(forecast_result)
    print(error_result)

    return forecast_result, error_result