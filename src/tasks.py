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

from scripts import auto_arima_gpu, linear_regression_gpu, exponential_smoothing_gpu, prophet_cpu
from scripts import holt_winters_gpu, gradient_boosting_gpu, knn_gpu, cnn_lstm_pytorch_single, svm_gpu, lstnet_pytorch_single
from scripts import deepar_pytorch, tft_pytorch, nbeats_pytorch, deepar_pytorch_v2, deepvar_pytorch
from scripts.connection import *
from scripts.functions import extract_number

from scripts.logger_ml import logging_ml


logger = logging.getLogger(__name__)

def run_forecast_task(id_user, id_prj, version_name, background_tasks):

    try:
        logger.info(f"Fetching data for project {id_prj} with version {version_name}.")
        id_cust = get_id_cust_from_id_prj(id_prj)

        dbase = get_dataset(id_prj, version_name, id_cust)
        dbset = get_setting_postgre(id_prj, version_name, id_cust)

        models = get_models_postgre(id_prj, id_cust)
        id_version = extract_number(version_name)

        logger.info(f'id_user: {id_user}, id_prj: {id_prj}, id_version: {id_version}, id_cust: {id_cust}, models: {models}')
        
        ex_id = logging_ml(id_user, id_prj, id_version, id_cust, str(models), "RUNNING", "Data is fetched", "tasks.py : run_forecast_task")
        
        update_process_status(id_prj, id_version, 'RUNNING')

        total_models = len(models)
        update_running_models(id_prj, id_version, total_models)
        reset_model_finished(id_prj, id_version)
        reset_process_status_progress(id_prj, id_version)
    except Exception as e:
        logging_ml(id_user, id_prj, id_version, id_cust, str(models), "ERROR", "Data is not fetched", "tasks.py : run_forecast_task : " + str(e))
        logger.error(f"Error in fetching data: {str(e)}")
        ask_to_shutdown()
        return None

    try:
        logger.info(f"Running forecast for project {id_prj} with version {version_name}.")

        # 1
        if 'Auto ARIMA' in models:
            background_tasks.add_task(run_forecast_auto_arima_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"Auto ARIMA forecast running for project {id_prj} with version {version_name}.")
        
        # 2
        if 'Prophet Forecast' in models:
            background_tasks.add_task(run_forecast_prophet_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"Prophet forecast running for project {id_prj} with version {version_name}.")
        
        # 3
        if 'Linear Regression' in models:
            background_tasks.add_task(run_forecast_linear_regression_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"Linear Regression forecast running for project {id_prj} with version {version_name}.")

        # 4
        if 'CNN LSTM' in models:
            background_tasks.add_task(run_forecast_cnn_lstm_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"CNN LSTM forecast running for project {id_prj} with version {version_name}.")

        # 5
        if 'Exponential Smoothing' in models:
            background_tasks.add_task(run_forecast_exponential_smoothing_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"Exponential Smoothing forecast running for project {id_prj} with version {version_name}.")

        # 6
        if 'Holt Winters' in models:
            background_tasks.add_task(run_forecast_holt_winters_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"Holt-Winters forecast running for project {id_prj} with version {version_name}.")

        # 7
        if 'Gradient Boosting' in models:
            background_tasks.add_task(run_forecast_gradient_boosting_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"Gradient Boosting forecast running for project {id_prj} with version {version_name}.")

        # 8
        if 'KNN' in models:
            background_tasks.add_task(run_forecast_knn_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"KNN forecast running for project {id_prj} with version {version_name}.")

        # 9
        if 'SVM' in models:
            background_tasks.add_task(run_forecast_svm_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"SVM forecast running for project {id_prj} with version {version_name}.")

        # 10
        if 'LSTNet' in models:
            background_tasks.add_task(run_forecast_lstnet_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"LSTNet forecast running for project {id_prj} with version {version_name}.")

        # 11
        if 'DeepAR' in models:
            background_tasks.add_task(run_forecast_deepar_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"DeepAR forecast running for project {id_prj} with version {version_name}.")

        # 12
        if 'Temporal Fusion Transformer' in models:
            # background_tasks.add_task(run_forecast_tft_bg, id_user, dbase, dbset, ex_id)
            # background_tasks.add_task(run_forecast_nbeats_bg, id_user, dbase, dbset, ex_id)
            background_tasks.add_task(run_forecast_nhits_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"TFT forecast running for project {id_prj} with version {version_name}.")
        
        # 13
        if 'N-BEATS' in models:
            background_tasks.add_task(run_forecast_nbeats_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"N-BEATS forecast running for project {id_prj} with version {version_name}.")

        # 14
        if 'DeepVAR' in models:
            background_tasks.add_task(run_forecast_deepvar_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"DeepVAR forecast running for project {id_prj} with version {version_name}.")

        # 15
        if 'NHITS' in models:
            background_tasks.add_task(run_forecast_nhits_bg, id_user, dbase, dbset, ex_id)
            logger.info(f"NHITS forecast running for project {id_prj} with version {version_name}.")

        return {"version_name": version_name, "id_prj": id_prj, "models": models}

    except Exception as e:
        logging_ml(id_user, id_prj, id_version, id_cust, str(models), "ERROR", "FAILED TO RUN MODELS", "tasks.py : run_forecast_task : " + str(e), execution_id=ex_id)
        logger.error(f"Error in process_forecast: {str(e)}")
        ask_to_shutdown()

def run_forecast_auto_arima_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Auto ARIMA", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_auto_arima_bg", execution_id=ex_id)
        auto_arima_gpu.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_auto_arima: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Auto ARIMA", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_auto_arima_bg : " + str(e), execution_id=ex_id)


def run_forecast_prophet_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Prophet", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_prophet_bg", execution_id=ex_id)
        prophet_cpu.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_prophet: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Prophet", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_prophet_bg : " + str(e), execution_id=ex_id)
        

def run_forecast_linear_regression_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Linear Regression", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_linear_regression_bg", execution_id=ex_id)
        linear_regression_gpu.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_linear_regression: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Linear Regression", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_linear_regression_bg : " + str(e), execution_id=ex_id)

def run_forecast_cnn_lstm_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "CNN LSTM", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_cnn_lstm_bg", execution_id=ex_id)
        cnn_lstm_pytorch_single.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_cnn_lstm: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "CNN LSTM", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_cnn_lstm_bg : " + str(e), execution_id=ex_id)

def run_forecast_exponential_smoothing_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Exponential Smoothing", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_exponential_smoothing_bg", execution_id=ex_id)
        exponential_smoothing_gpu.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_exponential_smoothing: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Exponential Smoothing", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_exponential_smoothing_bg : " + str(e), execution_id=ex_id)

def run_forecast_holt_winters_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Holt-Winters", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_holt_winters_bg", execution_id=ex_id)
        holt_winters_gpu.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_holt_winters: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Holt-Winters", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_holt_winters_bg : " + str(e), execution_id=ex_id)

def run_forecast_gradient_boosting_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "Gradient Boosting", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_gradient_boosting_bg", execution_id=ex_id)
        gradient_boosting_gpu.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_gradient_boosting: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "Gradient Boosting", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_gradient_boosting_bg : " + str(e), execution_id=ex_id)

def run_forecast_knn_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "KNN", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_knn_bg", execution_id=ex_id)
        knn_gpu.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_knn: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "KNN", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_knn_bg : " + str(e), execution_id=ex_id)

def run_forecast_svm_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "SVM", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_svr_bg", execution_id=ex_id)
        svm_gpu.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_svr: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "SVM", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_svr_bg : " + str(e), execution_id=ex_id)

def run_forecast_lstnet_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "LSTNet", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_forecast_svr_bg", execution_id=ex_id)
        lstnet_pytorch_single.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_svr: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "LSTNet", "ERROR", "MODEL IS FAILED", "tasks.py : run_forecast_svr_bg : " + str(e), execution_id=ex_id)

def run_forecast_deepar_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "DeepAR", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_deepar_bg", execution_id=ex_id)
        deepar_pytorch.run_model(id_user, dbase, dbset, ex_id)
        # deepar_pytorch_v2.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_deepar: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "DeepAR", "ERROR", "MODEL IS FAILED", "tasks.py : run_deepar_bg : " + str(e), execution_id=ex_id)

def run_forecast_tft_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "TFT", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_tftbg", execution_id=ex_id)
        tft_pytorch.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_tft: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "TFT", "ERROR", "MODEL IS FAILED", "tasks.py : run_tft_bg : " + str(e), execution_id=ex_id)

def run_forecast_nbeats_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "N-BEATS", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_nbeats_bg", execution_id=ex_id)
        nbeats_pytorch.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_nbeats: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "N-BEATS", "ERROR", "MODEL IS FAILED", "tasks.py : run_nbeats_bg : " + str(e), execution_id=ex_id)

def run_forecast_deepvar_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "DeepVAR", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_deepvar_bg", execution_id=ex_id)
        deepvar_pytorch.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_deepvar: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "DeepVAR", "ERROR", "MODEL IS FAILED", "tasks.py : run_deepvar_bg : " + str(e), execution_id=ex_id)

def run_forecast_nhits_bg(id_user, dbase, dbset, ex_id):
    try:
        id_version = extract_number(dbset['version_name'][0])
        id_prj = dbset['id_prj'][0]
        id_cust = get_id_cust_from_id_prj(id_prj)
        logging_ml(id_user, id_prj, id_version, id_cust, "NHITS", "RUNNING", "MODEL IS RUNNING", "tasks.py : run_nhits_bg", execution_id=ex_id)
        import scripts.nhits_pytorch as nhits_pytorch
        nhits_pytorch.run_model(id_user, dbase, dbset, ex_id)
    except Exception as e:
        logger.error(f"Error in run_forecast_nhits: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')
        logging_ml(id_user, id_prj, id_version, id_cust, "NHITS", "ERROR", "MODEL IS FAILED", "tasks.py : run_nhits_bg : " + str(e), execution_id=ex_id)