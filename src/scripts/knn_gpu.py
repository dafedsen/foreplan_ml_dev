import cudf as cd
import cupy as cp
from cuml.neighbors import KNeighborsRegressor
from cuml.preprocessing import MinMaxScaler
from cuml.model_selection import train_test_split
from cuml.metrics import mean_squared_error
from cuml.metrics import r2_score
import math

from datetime import timedelta
import time
import logging

logger = logging.getLogger(__name__)

from scripts.connection import *
from scripts.functions import *

def run_model(dbase, dbset):
    try:
        logger.info("K-Nearest Neighbors forecast running.")
        id_cust = get_id_cust_from_id_prj(dbase['id_prj'][0])
        id_prj = dbase['id_prj'][0]
        id_version = extract_number(dbase['version_name'][0])

        t_forecast = get_forecast_time(dbase, dbset)    

        start_time = time.time()
        pred, err = run_knn(dbase, t_forecast, dbset)
        end_time = time.time()

        logger.info("Sending K-Nearest Neighbors forecast result.")
        send_process_result(pred, id_cust)

        logger.info("Sending K-Nearest Neighbors forecast evaluation.")
        send_process_evaluation(err, id_cust)

        print(str(timedelta(seconds=end_time - start_time)))
        status = check_update_process_status_success(id_prj, id_version)

        if status:
            update_end_date(id_prj, id_version)

        return str(timedelta(seconds=end_time - start_time))
    
    except Exception as e:
        logger.error(f"Error in knn_gpu.run_model : {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def predict_model(df, st, t_forecast):

    try:
        pred = cd.DataFrame()

        level1 = df['level1'][0]
        level2 = df['level2'][0]

        PROCESS = st['id_prj_prc'][0]

        ADJUSTMENT = st['adj_include'][0]

        SIGMA = st['out_std_dev'][0]

        SMOOTHING = st['ad_smooth_method'][0]

        if ADJUSTMENT == 'Yes':
            adjusting_data(df, SMOOTHING, SIGMA)

        cleansing_outliers(df, SIGMA)

        scaler = MinMaxScaler()

        df = df.sort_values(by='hist_date')
        df['hist_value'] = df.hist_value.astype('float32')
        df["hist_value"] = scaler.fit_transform(df[["hist_value"]])

        lookback = 7
        X, y = [], []

        for i in range(lookback, len(df)):
            X.append(df.iloc[i - lookback:i]["hist_value"].to_numpy())
            y.append(df.iloc[i]["hist_value"].to_numpy())

        X = cp.array(X)
        y = cp.array(y)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        n_range = (1, 20)
        best_n = find_best_n_neighbors(X_train, X_test, y_train, y_test, n_range)

        model = KNeighborsRegressor(n_neighbors=best_n, metric="euclidean", algorithm="brute")
        model.fit(X_train, y_train)

        forecast = []
        last_features = X_test[-1].copy()

        for _ in range(t_forecast.shape[0]):
            next_value = model.predict(cp.array([last_features]))
            forecast.append(next_value.get()[0])

            last_features[:-1] = last_features[1:]
            last_features[-1] = next_value

        y_pred = scaler.inverse_transform(cp.array(forecast).reshape(-1, 1))
        y_pred = cp.array(y_pred).flatten()
        print(y_pred)

        pred['date'] = t_forecast['date']
        pred[level2] = y_pred
        pred['level1'] = level1
        pred['adj_include'] = ADJUSTMENT
        pred['id_prj_prc'] = PROCESS
        pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]

    except Exception as e:
        print('\nERROR EXCEPTION FORECASTING ', e)
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
            'mape': [float(mape)],
        })
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT
        err['id_prj_prc'] = PROCESS

    except Exception as e:
        print('\nERROR EXCEPTION EVALUATING ', e)
        rmse = 999999999.0
        r2 = 999999999.0
        bias = 999999999.0
        mape = 999999999.0

        err = cd.DataFrame({'rmse': [rmse], 'r2': [r2], 'bias': [bias], 'mape': [mape]})
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT
        err['id_prj_prc'] = PROCESS

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return pred, err

def run_knn(dbase, t_forecast, dbset):

    project = dbase['id_prj'][0]
    id_version = extract_number(dbset['version_name'][0])
    id_cust = get_id_cust_from_id_prj(project)

    level1_list = dbase['level1'].unique().to_arrow().to_pylist()
    level1_list = sorted(level1_list)

    level2_list = dbase['level2'].unique().to_arrow().to_pylist()
    level2_list = sorted(level2_list)
    
    total_loop = len(level1_list) * len(level2_list)
    current_loop = 0

    lr_settings = dbset[dbset['model_name'] == 'KNN']
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
    forecast_result['id_model'] = 8

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
    error_result['id_model'] = 8
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

def find_best_n_neighbors(X_train, X_test, y_train, y_test, n_range):
    best_n = None
    best_mse = float("inf")

    for n in range(n_range[0], n_range[1] + 1):
        knn = KNeighborsRegressor(n_neighbors=n, metric="euclidean", algorithm="brute")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_n = n

    print(f"Best n_neighbors: {best_n} with MSE: {best_mse} \n")
    return best_n