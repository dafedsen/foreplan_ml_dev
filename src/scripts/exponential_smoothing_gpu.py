import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from cuml import ExponentialSmoothing
from cuml.metrics import r2_score
from cuml.model_selection import train_test_split

import math
import cudf as cd
from datetime import timedelta
import time
import logging

logger = logging.getLogger(__name__)

from scripts.connection import *
from scripts.functions import *

def run_model(dbase, dbset):
    logger.info("Linear Regression forecast running.")
    id_cust = get_id_cust_from_id_prj(dbase['id_prj'][0])
    id_prj = dbase['id_prj'][0]
    id_version = extract_number(dbase['version_name'][0])

    update_process_status(id_prj, id_version, 'RUNNING')

    t_forecast = get_forecast_time(dbase, dbset)    

    start_time = time.time()
    pred, err = run_exponential_smoothing(dbase, t_forecast, dbset)
    end_time = time.time()

    logger.info("Sending Linear Regression forecast result.")
    send_process_result(pred, id_cust)

    logger.info("Sending Linear Regression forecast evaluation.")
    send_process_evaluation(err, id_cust)

    # update_process_status(id_prj, id_version, 'SUCCESS')
    print(str(timedelta(seconds=end_time - start_time)))

    return str(timedelta(seconds=end_time - start_time))

def create_lagged_features(df, target_col, lags):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def run_model(df, st, t_forecast):

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
        
        df['hist_date'] = pd.to_datetime(df['hist_date'])
        
        df_train = df[['hist_date', 'hist_value']]
        data_lagged = create_lagged_features(df_train, target_col="hist_value", lags=10)
        
        X = data_lagged.drop(columns=["hist_date", "hist_value"])
        y = data_lagged["hist_value"]
        
        # Data Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

        # Initiate Model and Train
        model = ExponentialSmoothing()
        model.fit(X_train, y_train)
        
        # Model Predict
        #X_pred = get_weeks_by_number(t_forecast.shape[0] + X_test.shape[0])
        #y_pred = model.predict(X_pred)
        
        n_forecast = t_forecast.shape[0] + X_test.shape[0]
        last_features = X.iloc[-1].values.reshape(1, -1)
        forecast = []
        
        for _ in range(n_forecast):
            # Predict the next value
            next_value = model.predict(last_features)[0]
            forecast.append(next_value)
        
            # Update features by shifting and adding the new value
            last_features = np.roll(last_features, shift=-1)
            last_features[0, -1] = next_value
        
        #y_pred = encoder.inverse_transform(forecast)
        
        forecast = forecast[-t_forecast.shape[0]:]
        
        print(t_forecast.shape)
        print(len(forecast))
        
        # Prediction Data Process
        pred['date'] = t_forecast['date']
        pred[level2] = forecast
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
        y_test = y_test.to_numpy()
        print(type(y_test))
        print(y_test)

        rmse = get_rmse(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        bias = get_bias(y_test, y_pred_test)
        mape = mean_absolute_percentage_error(y_test, y_pred_test)

        if math.isinf(mape) == True:
            mape = 999999999

        # Evaluation Data Process
        err = pd.DataFrame({'rmse': [rmse], 'r2': [r2], 'bias': [bias], 'mape': [mape]})
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT
    
    except Exception as e:
        rmse = 999999999
        r2 = 999999999
        bias = 999999999
        mape = 999999999

        err = pd.DataFrame({'rmse': [rmse], 'r2': [r2], 'bias': [bias], 'mape': [mape]})
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return pred, err