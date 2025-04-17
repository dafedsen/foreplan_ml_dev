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

from scripts import auto_arima_gpu, linear_regression_gpu, cnn_lstm_pytorch, exponential_smoothing_gpu, prophet_cpu
from scripts import holt_winters_gpu, gradient_boosting_gpu, knn_gpu
from scripts.connection import *
from scripts.functions import extract_number


logger = logging.getLogger(__name__)

def run_forecast_task(id_prj, version_name, background_tasks):

    try:
        logger.info(f"Fetching data for project {id_prj} with version {version_name}.")
        dbase = get_dataset_postgre(id_prj, version_name)
        dbset = get_setting_postgre(id_prj, version_name)
        models = get_models_postgre(id_prj)

        id_version = extract_number(dbset['version_name'][0])
        update_process_status(id_prj, id_version, 'RUNNING')

        total_models = len(models)
        update_running_models(id_prj, id_version, total_models)
        reset_model_finished(id_prj, id_version)
        reset_process_status_progress(id_prj, id_version)
    except Exception as e:
        logger.error(f"Error in fetching data: {str(e)}")
        return None

    try:
        logger.info(f"Running forecast for project {id_prj} with version {version_name}.")

        # 1
        if 'Auto ARIMA' in models:
            background_tasks.add_task(run_forecast_auto_arima_bg, dbase, dbset)
            logger.info(f"Auto ARIMA forecast running for project {id_prj} with version {version_name}.")
        
        # 2
        if 'Prophet Forecast' in models:
            background_tasks.add_task(run_forecast_prophet_bg, dbase, dbset)
            logger.info(f"Prophet forecast running for project {id_prj} with version {version_name}.")
        
        # 3
        if 'Linear Regression' in models:
            background_tasks.add_task(run_forecast_linear_regression_bg, dbase, dbset)
            logger.info(f"Linear Regression forecast running for project {id_prj} with version {version_name}.")

        # 4
        if 'CNN LSTM' in models:
            background_tasks.add_task(run_forecast_cnn_lstm_bg, dbase, dbset)
            logger.info(f"CNN LSTM forecast running for project {id_prj} with version {version_name}.")

        # 5
        if 'Exponential Smoothing' in models:
            background_tasks.add_task(run_forecast_exponential_smoothing_bg, dbase, dbset)
            logger.info(f"Exponential Smoothing forecast running for project {id_prj} with version {version_name}.")

        6
        if 'Holt Winters' in models:
            background_tasks.add_task(run_forecast_holt_winters_bg, dbase, dbset)
            logger.info(f"Holt-Winters forecast running for project {id_prj} with version {version_name}.")

        # 7
        if 'Gradient Boosting' in models:
            background_tasks.add_task(run_forecast_gradient_boosting_bg, dbase, dbset)
            logger.info(f"Gradient Boosting forecast running for project {id_prj} with version {version_name}.")

        # 8
        if 'KNN' in models:
            background_tasks.add_task(run_forecast_knn_bg, dbase, dbset)
            logger.info(f"KNN forecast running for project {id_prj} with version {version_name}.")

        return {"version_name": version_name, "id_prj": id_prj, "models": models}

    except Exception as e:
        logger.error(f"Error in process_forecast: {str(e)}")

def run_forecast_auto_arima_bg(dbase, dbset):
    id_version = extract_number(dbset['version_name'][0])
    id_prj = dbset['id_prj'][0]
    try:
        auto_arima_gpu.run_model(dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_auto_arima: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def run_forecast_prophet_bg(dbase, dbset):
    id_version = extract_number(dbset['version_name'][0])
    id_prj = dbset['id_prj'][0]
    try:
        prophet_cpu.run_model(dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_prophet: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def run_forecast_linear_regression_bg(dbase, dbset):
    id_version = extract_number(dbset['version_name'][0])
    id_prj = dbset['id_prj'][0]
    try:
        linear_regression_gpu.run_model(dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_linear_regression: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def run_forecast_cnn_lstm_bg(dbase, dbset):
    id_version = extract_number(dbset['version_name'][0])
    id_prj = dbset['id_prj'][0]
    try:
        cnn_lstm_pytorch.run_model(dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_cnn_lstm: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def run_forecast_exponential_smoothing_bg(dbase, dbset):
    id_version = extract_number(dbset['version_name'][0])
    id_prj = dbset['id_prj'][0]
    try:
        exponential_smoothing_gpu.run_model(dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_exponential_smoothing: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def run_forecast_holt_winters_bg(dbase, dbset):
    id_version = extract_number(dbset['version_name'][0])
    id_prj = dbset['id_prj'][0]
    try:
        holt_winters_gpu.run_model(dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_holt_winters: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def run_forecast_gradient_boosting_bg(dbase, dbset):
    id_version = extract_number(dbset['version_name'][0])
    id_prj = dbset['id_prj'][0]
    try:
        gradient_boosting_gpu.run_model(dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_gradient_boosting: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')

def run_forecast_knn_bg(dbase, dbset):
    id_version = extract_number(dbset['version_name'][0])
    id_prj = dbset['id_prj'][0]
    try:
        knn_gpu.run_model(dbase, dbset)
    except Exception as e:
        logger.error(f"Error in run_forecast_knn: {str(e)}")
        update_process_status(id_prj, id_version, 'ERROR')