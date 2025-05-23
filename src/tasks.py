import sys
import os

BASE_DIR = '/home/ubuntu/foreplan_ml_dev/src'
SRC_DIR = os.path.join(BASE_DIR, "scripts")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR) 

from fastapi import BackgroundTasks
import logging
import cudf as cd

from scripts import auto_arima_gpu, linear_regression_gpu, cnn_lstm_pytorch, exponential_smoothing_gpu, prophet_cpu
from scripts import holt_winters_gpu, gradient_boosting_gpu, knn_gpu, cnn_lstm_pytorch_single, svm_gpu, lstnet_pytorch_single
from scripts.connection import *
from scripts.functions import extract_number

from scripts.logger_ml import logging_ml


logger = logging.getLogger(__name__)

def run_forecast_task(id_user, id_prj, version_name, background_tasks):

    try:
        logger.info(f"Fetching data for project {id_prj} with version {version_name}.")
        id_cust = get_id_cust_from_id_prj(id_prj)

        dbase = get_dataset(id_prj, version_name)
        # dbase.to_csv('dataset.csv', index=False)

        # dbase_adj = get_dataset_adj_postgre(id_prj, version_name)
        # dbase_adj['flag_adj'] = 1
        # dbase_non_adj = get_dataset_non_adj_postgre(id_prj, version_name)
        # dbase_non_adj['flag_adj'] = 0
        # dbase = cd.concat([dbase_adj, dbase_non_adj])
        # dbase.to_csv('dataset_adj_non_adj.csv', index=False)


        dbset = get_setting_postgre(id_prj, version_name)

        models = get_models_postgre(id_prj)
        id_version = extract_number(version_name)
        
        logging_ml(id_user, id_prj, id_version, id_cust, str(models), "RUNNING", "Data is fetched", "tasks.py : run_forecast_task")
        
        update_process_status(id_prj, id_version, 'RUNNING')

        total_models = len(models)
        update_running_models(id_prj, id_version, total_models)
        reset_model_finished(id_prj, id_version)
        reset_process_status_progress(id_prj, id_version)
    except Exception as e:
        # logging_ml(id_user, id_prj, id_version, id_cust, str(models), "ERROR", "Data is not fetched", "tasks.py : run_forecast_task : " + str(e))
        logger.error(f"Error in fetching data: {str(e)}")
        return None

    try:
        logger.info(f"Running forecast for project {id_prj} with version {version_name}.")

        # # 1
        # if 'Auto ARIMA' in models:
        #     background_tasks.add_task(run_forecast_auto_arima_bg, id_user, dbase, dbset)
        #     logger.info(f"Auto ARIMA forecast running for project {id_prj} with version {version_name}.")
        
        # # 2
        # if 'Prophet Forecast' in models:
        #     background_tasks.add_task(run_forecast_prophet_bg, id_user, dbase, dbset)
        #     logger.info(f"Prophet forecast running for project {id_prj} with version {version_name}.")
        
        # # 3
        # if 'Linear Regression' in models:
        #     background_tasks.add_task(run_forecast_linear_regression_bg, id_user, dbase, dbset)
        #     logger.info(f"Linear Regression forecast running for project {id_prj} with version {version_name}.")

        # # 4
        # if 'CNN LSTM' in models:
        #     background_tasks.add_task(run_forecast_cnn_lstm_bg, id_user, dbase, dbset)
        #     logger.info(f"CNN LSTM forecast running for project {id_prj} with version {version_name}.")

        # # 5
        # if 'Exponential Smoothing' in models:
        #     background_tasks.add_task(run_forecast_exponential_smoothing_bg, id_user, dbase, dbset)
        #     logger.info(f"Exponential Smoothing forecast running for project {id_prj} with version {version_name}.")

        # # 6
        # if 'Holt Winters' in models:
        #     background_tasks.add_task(run_forecast_holt_winters_bg, id_user, dbase, dbset)
        #     logger.info(f"Holt-Winters forecast running for project {id_prj} with version {version_name}.")

        # # 7
        # if 'Gradient Boosting' in models:
        #     background_tasks.add_task(run_forecast_gradient_boosting_bg, id_user, dbase, dbset)
        #     logger.info(f"Gradient Boosting forecast running for project {id_prj} with version {version_name}.")

        # # 8
        # if 'KNN' in models:
        #     background_tasks.add_task(run_forecast_knn_bg, id_user, dbase, dbset)
        #     logger.info(f"KNN forecast running for project {id_prj} with version {version_name}.")

        # # 9
        # if 'SVM' in models:
        #     background_tasks.add_task(run_forecast_svm_bg, id_user, dbase, dbset)
        #     logger.info(f"SVM forecast running for project {id_prj} with version {version_name}.")

        # 10
        if 'LSTNet' in models:
            background_tasks.add_task(run_forecast_lstnet_bg, id_user, dbase, dbset)
            logger.info(f"LSTNet forecast running for project {id_prj} with version {version_name}.")

        return {"version_name": version_name, "id_prj": id_prj, "models": models}

    except Exception as e:
        logging_ml(id_user, id_prj, id_version, id_cust, str(models), "ERROR", "FAILED TO RUN MODELS", "tasks.py : run_forecast_task : " + str(e))
        logger.error(f"Error in process_forecast: {str(e)}")

def run_forecast_auto_arima_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Auto ARIMA", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_auto_arima_bg")
        auto_arima_gpu.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_auto_arima: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Auto ARIMA", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_auto_arima_bg : " + str(e))


def run_forecast_prophet_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Prophet", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_prophet_bg")
        prophet_cpu.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_prophet: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Prophet", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_prophet_bg : " + str(e))
        

def run_forecast_linear_regression_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Linear Regression", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_linear_regression_bg")
        linear_regression_gpu.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_linear_regression: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Linear Regression", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_linear_regression_bg : " + str(e))

def run_forecast_cnn_lstm_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "CNN LSTM", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_cnn_lstm_bg")
        cnn_lstm_pytorch_single.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_cnn_lstm: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "CNN LSTM", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_cnn_lstm_bg : " + str(e))

def run_forecast_exponential_smoothing_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Exponential Smoothing", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_exponential_smoothing_bg")
        exponential_smoothing_gpu.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_exponential_smoothing: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Exponential Smoothing", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_exponential_smoothing_bg : " + str(e))

def run_forecast_holt_winters_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Holt-Winters", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_holt_winters_bg")
        holt_winters_gpu.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_holt_winters: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Holt-Winters", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_holt_winters_bg : " + str(e))

def run_forecast_gradient_boosting_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Gradient Boosting", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_gradient_boosting_bg")
        gradient_boosting_gpu.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_gradient_boosting: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Gradient Boosting", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_gradient_boosting_bg : " + str(e))

def run_forecast_knn_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "KNN", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_knn_bg")
        knn_gpu.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_knn: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "KNN", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_knn_bg : " + str(e))

def run_forecast_svm_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "SVM", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_svr_bg")
        svm_gpu.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_svr: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "SVM", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_svr_bg : " + str(e))

def run_forecast_lstnet_bg(id_user, dbase, dbset):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "LSTNet", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_svr_bg")
        lstnet_pytorch_single.run_model(id_user, dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_svr: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "LSTNet", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_svr_bg : " + str(e))