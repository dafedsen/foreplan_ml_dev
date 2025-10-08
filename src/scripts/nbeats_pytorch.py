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
import torch.distributions as D
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import NBeats
from pytorch_forecasting.data import GroupNormalizer, TorchNormalizer
from torchmetrics import MeanSquaredError
from pytorch_forecasting.metrics import RMSE
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from cuml.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ID_MODEL = 12  # N-BEATS 13

def run_model(id_user, dbase, dbset, ex_id):
    try:
        logger.info("NBEATS Single forecast running.")
        id_prj = int(dbase['id_prj'].iloc[0].item())
        version_name = dbase['version_name'].iloc[0]

        id_cust = get_id_cust_from_id_prj(id_prj)
        id_version = extract_number(version_name)

        logging_ml(id_user, id_prj, id_version, id_cust, "NBEATS", "RUNNING", "Model is running", "nbeats_pytorch.py : run_model", execution_id=ex_id)

        t_forecast = get_forecast_time(dbase, dbset)    

        start_time = time.time()
        pred, err = run_nbeats(dbase, t_forecast, dbset)
        end_time = time.time()

        logger.info("Sending NBEATS Single forecast result.")
        send_process_result(pred, id_cust)

        logger.info("Sending NBEATS Single forecast evaluation.")
        send_process_evaluation(err, id_cust)

        print(str(timedelta(seconds=end_time - start_time)))
        status = check_update_process_status_success(id_prj, id_version)

        if status:
            update_end_date(id_prj, id_version)

        logging_ml(id_user, id_prj, id_version, id_cust, "NBEATS", "FINISHED", "Finished running model", "nbeats_pytorch.py : run_model",
                   start_date=start_time, end_date=end_time, execution_id=ex_id)

        return str(timedelta(seconds=end_time - start_time))
    
    except Exception as e:
        logger.error(f"Error in nbeats_pytorch.run_model : {str(e)}")
        logging_ml(id_user, id_prj, id_version, id_cust, "NBEATS", "ERROR", "Error in running model", "nbeats_pytorch.py : run_model : " + str(e), execution_id=ex_id)
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

        cutoff = int(len(df) * 0.9)
        df_train = df.iloc[:cutoff].copy().sort_values(by='hist_date')
        df_test = df.iloc[cutoff:].copy().sort_values(by='hist_date')

        # Data Preparation
        try:
            # df = df.sort_values(by='hist_date')
            scaler = GPUMinMaxScaler()
            df_train["hist_value"] = cp.asarray(scaler.fit_transform(df_train["hist_value"])).flatten()
        except Exception as e:
            logger.error(f'Error Scaling : {str(e)}')

        try:
            df_t = df_train[['hist_date', 'hist_value']].copy()
            df_t['time_idx'] = (df_t['hist_date'] - df_t['hist_date'].min()).dt.days
            df_t['time_idx'] = df_t['time_idx'].astype(int)
            df_t['series'] = 1
            df_t['series'] = df_t['series'].astype(str)
            # df_t["month"] = pd.Categorical(
            #     df_t["hist_date"].dt.month, 
            #     categories=list(range(1,13)),
            #     ordered=False
            # ).astype(str).astype('category')
            # df_t["dayofweek"] = pd.Categorical(
            #     df_t["hist_date"].dt.dayofweek, 
            #     categories=list(range(0,7)),
            #     ordered=False
            # ).astype(str).astype('category')
            df_t = df_t.to_pandas()
            logger.info(f'df_t type : {df_t.dtypes}')
        except Exception as e:
            logger.error(f'Error add format data : {str(e)}')

        try:
            encoder_length = min(14, max(2, df_t.shape[0] // 3))
            prediction_length = t_forecast.shape[0] + df_test.shape[0]

            max_encoder_length = encoder_length
            max_prediction_length = prediction_length

            training_cutoff = df_t["time_idx"].max() - prediction_length
        except Exception as e:
            logger.error(f'Error define params : {str(e)}')

        try:
            print('Define training data')
            # training = TimeSeriesDataSet(
            #     df_t[df_t["time_idx"] <= training_cutoff],
            #     time_idx="time_idx",
            #     target="hist_value",
            #     group_ids=["series"],
            #     min_encoder_length=max_encoder_length,
            #     max_encoder_length=max_encoder_length,
            #     min_prediction_length=max_prediction_length,
            #     max_prediction_length=max_prediction_length,
            #     # time_varying_known_categoricals=["month", "dayofweek"],
            #     time_varying_unknown_reals=["hist_value"],  # target di masa lalu
            #     target_normalizer=TorchNormalizer(method="standard"),  # bisa pakai Normalizer jika perlu
            #     allow_missing_timesteps=True,
            # )
            training, validation = build_universal_nbeats_dataset(
                df_t,
                time_col="hist_date",
                group_col="series",
                target_col="hist_value"
            )
            logger.info(f'training type : {type(training)}')
        except Exception as e:
            logger.error(f'Error training data : {str(e)}')

        # try:
        #     validation = TimeSeriesDataSet.from_dataset(training, df_t, predict=True, stop_randomization=True)
        # except Exception as e:
        #     logger.error(f'Error validation data : {str(e)}')

        try:
            train_loader = training.to_dataloader(
                train=True, 
                batch_size=64, 
                num_workers=0
                )
            val_loader = validation.to_dataloader(
                train=False, 
                batch_size=64, 
                num_workers=0
                )
        except Exception as e:
            logger.error(f'Error data loader : {str(e)}')

        try:
            model = NBeats.from_dataset(
                training,
                learning_rate=1e-3,
                log_interval=10,
                log_val_interval=1,
                weight_decay=1e-2,
                widths=[512, 512],
                backcast_loss_ratio=0.0,
                loss=RMSE(),   # bisa ganti ke RMSE() juga kalau mau lebih mirip TFT
                optimizer="adam"
            )
        except Exception as e:
            logger.error(f'Error from dataset : {str(e)}')

        try:
            # checkpoint_callback = ModelCheckpoint(
            #     save_top_k=1,
            #     monitor="val_loss",
            #     mode="min"
            # )

            trainer = pl.Trainer(
                max_epochs=50,
                accelerator='cuda',
                devices=1,
                enable_checkpointing=False,
                enable_progress_bar=True,
                logger=False,
                num_nodes=1, 
                num_sanity_val_steps=0,
            )

            trainer.fit(
                model,
                train_loader,
                val_loader
            )
        except Exception as e:
            logger.error(f'Error trainer : {str(e)}')

        # try:
        #     raw_predictions = model.predict(val_loader)
        #     logger.info(f'raw type : {type(raw_predictions)}')

        #     y_pred = raw_predictions.cpu().numpy()
        #     y_pred = scaler.inverse_transform(cp.array(y_pred).reshape(-1, 1)).reshape(y_pred.shape)

        #     if y_pred.ndim == 2 and y_pred.shape[0] == 1:
        #         y_pred = y_pred[0]

        #     # ensure same length
        #     if len(y_pred) < len(t_forecast):
        #         diff = len(t_forecast) - len(y_pred)
        #         last_val = y_pred[-1] if len(y_pred) > 0 else np.nan
        #         y_pred = np.concatenate([y_pred, np.full(diff, last_val)])
        #     elif len(y_pred) > len(t_forecast):
        #         y_pred = y_pred[:len(t_forecast)]
        # except Exception as e:
        #     logger.error(f'Error predict y_pred : {str(e)}')

        try:
            # --- Setup awal ---
            horizon = len(t_forecast)
            y_pred_total = []

            # ambil panjang prediksi dari dataset (lebih stabil daripada model.hparams)
            pred_len = getattr(training, "max_prediction_length", 1)
            if not isinstance(pred_len, int) or pred_len <= 0:
                pred_len = 1  # fallback default

            logger.info(f"Forecasting autoregressive: horizon={horizon}, step={pred_len}")

            # siapkan dataset untuk iterasi
            dataset_iter = training
            current_input = dataset_iter
            total_generated = 0

            # --- Loop autoregressive ---
            while total_generated < horizon:
                # predict batch
                raw_predictions = model.predict(current_input, mode="prediction")

                # pastikan tensor di CPU
                if isinstance(raw_predictions, torch.Tensor):
                    raw_predictions = raw_predictions.detach().cpu().numpy()
                else:
                    raw_predictions = np.array(raw_predictions)

                # ubah ke bentuk 1D
                y_pred_batch = raw_predictions.flatten()

                # inverse scaling (jika pakai cupy scaler)
                try:
                    y_pred_batch = scaler.inverse_transform(cp.array(y_pred_batch).reshape(-1, 1)).get().flatten()
                except Exception:
                    y_pred_batch = scaler.inverse_transform(y_pred_batch.reshape(-1, 1)).flatten()

                # simpan hasil prediksi
                y_pred_total.extend(y_pred_batch.tolist())
                total_generated += len(y_pred_batch)
                logger.info(f"Generated {total_generated}/{horizon} steps")

                # hentikan kalau sudah cukup
                if total_generated >= horizon:
                    break

                # tambahkan prediksi ke dataset untuk step berikutnya
                next_data = dataset_iter.data.copy()
                max_time_idx = next_data["time_idx"].max()

                # tambahkan data baru hasil prediksi
                for i, val in enumerate(y_pred_batch):
                    next_data = pd.concat([
                        next_data,
                        pd.DataFrame({
                            "time_idx": [max_time_idx + i + 1],
                            "hist_value": [val],
                            "series": [next_data["series"].iloc[0]],
                        })
                    ], ignore_index=True)

                # buat ulang dataset untuk langkah berikutnya
                current_input = TimeSeriesDataSet.from_dataset(dataset_iter, next_data, predict=True)

            # pastikan panjang sama dengan horizon
            y_pred = np.array(y_pred_total[:horizon])

        except Exception as e:
            logger.error(f'Error autoregressive predict : {str(e)}')


        logger.info(f't_forecast shape : {t_forecast.shape}')
        logger.info(f'y_pred shape : {y_pred.shape}')
        pred['date'] = t_forecast['date']
        pred[level2] = y_pred[-t_forecast.shape[0]:]
        pred['level1'] = level1
        pred['adj_include'] = ADJUSTMENT
        pred['id_prj_prc'] = PROCESS
        pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]
        logger.info("\n%s", pred[["date", level2]])


    except Exception as e:
        logger.error(f'Error Predict TFT: {e}')
        pred['date'] = t_forecast['date']
        pred[level2] = 0
        pred['level1'] = level1
        pred['adj_include'] = ADJUSTMENT
        pred['id_prj_prc'] = PROCESS
        pred = pred[['adj_include', 'id_prj_prc', 'date', 'level1', level2]]
        logger.error(f"Error in predict_model: {str(e)}")
        
    
    try:
        # y_test = y_pred[:df_test.shape[0]]
        # y_act = df_test['hist_value']
        y_test = cp.asarray(y_pred[:df_test.shape[0]])
        y_act = cp.asarray(df_test['hist_value'].values)
        logger.info(f'y_test shape : {y_test.shape}')
        logger.info(f'y_act shape : {y_act.shape}')

        rmse = cp.sqrt(mean_squared_error(y_act, y_test))
        r2 = r2_score(y_act, y_test)
        bias = (cp.sum(y_test - y_act) / cp.sum(y_act)) * 100
        mape = cp.mean(cp.abs((y_test - y_act) / y_act)) * 100

        if math.isinf(mape) == True:
            mape = 999999999.0

        if math.isinf(r2) == True:
            r2 = 999999999.0

        err = cd.DataFrame({
            'rmse': [float(rmse)], 
            'r2': [float(r2)], 
            'bias': [float(bias)], 
            'mape': [float(mape)]
        }, dtype=float)
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT
        err['id_prj_prc'] = PROCESS

        logger.info("\n%s", err)
    except Exception as e:
        logger.error(f'Error in evaluate_model TFT : {e}')

        err = cd.DataFrame({
            'rmse': [float(999999999.0)], 
            'r2': [float(999999999.0)], 
            'bias': [float(999999999.0)], 
            'mape': [float(999999999.0)]
        }, dtype=float)
        err['level1'] = level1
        err['level2'] = level2
        err['adj_include'] = ADJUSTMENT
        err['id_prj_prc'] = PROCESS

    return pred, err


def run_nbeats(dbase, t_forecast, dbset):

    project = int(dbase['id_prj'].iloc[0].item())
    id_version = extract_number(dbset['version_name'].iloc[0])
    id_cust = get_id_cust_from_id_prj(project)

    level1_list = dbase['level1'].unique().to_arrow().to_pylist()
    level1_list = sorted(level1_list)

    level2_list = dbase['level2'].unique().to_arrow().to_pylist()
    level2_list = sorted(level2_list)
    
    total_loop = len(level1_list) * len(level2_list)
    current_loop = 0

    # Update nanti klo deploy
    lr_settings = dbset[dbset['model_name'] == 'Temporal Fusion Transformer']
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
    forecast_result['id_model'] = ID_MODEL

    forecast_result['date'] = cd.to_datetime(forecast_result['date'])
    forecast_result = forecast_result.groupby(['level1', 'level2', 'adj_include', 'id_prj_prc']).apply(
        lambda group: group.sort_values('date').head(t_forecast.shape[0])
    ).reset_index(drop=True)
    forecast_result = forecast_result[['date', 'id_prj_prc', 'level1', 'level2', 'hist_value', 'id_model']]
    forecast_result.rename(columns={'date': 'fcast_date', 'hist_value': 'fcast_value'}, inplace=True)
    forecast_result['partition_cust_id'] = id_cust
    forecast_result = forecast_result.dropna()
    forecast_result['fcast_value'] = forecast_result['fcast_value'].astype(float).round(3)

    for col in ['rmse', 'r2', 'mape', 'bias']:
        error_result[col] = error_result[col].astype(float)
    
    print(error_result.dtypes)

    error_result = error_result.melt(
        id_vars=['level1', 'adj_include', 'id_prj_prc', 'level2', 'id_prj', 'id_version'],
        value_vars=['rmse', 'r2', 'mape', 'bias'],
        var_name='err_method',
        value_name='err_value'
    )
    error_result['id_model'] = ID_MODEL
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

    return forecast_result, error_result

def build_universal_nbeats_dataset(df, time_col, group_col, target_col):
    df = df.sort_values([group_col, time_col])
    df[time_col] = pd.to_datetime(df[time_col])

    # --- Drop series terlalu pendek ---
    valid_groups = df.groupby(group_col).filter(lambda x: len(x) >= 3)
    removed = set(df[group_col]) - set(valid_groups[group_col])
    if removed:
        logger.warning(f"Removed {len(removed)} short series (len < 3): {list(removed)[:3]}...")
    df = valid_groups

    # --- Detect frequency ---
    try:
        inferred_freq = pd.infer_freq(df[time_col].sort_values())
    except Exception:
        inferred_freq = None
    freq = inferred_freq or "D"
    logger.info(f"Detected frequency: {freq}")

    # --- Min length per series ---
    min_series_len = df.groupby(group_col)[time_col].count().min()
    logger.info(f"Min series length: {min_series_len}")

    # --- Auto-adjust encoder/prediction ---
    if freq.startswith("D"):
        encoder_length = int(min(60, max(14, min_series_len // 2)))
        prediction_length = int(min(30, max(7, encoder_length // 2)))
    elif freq.startswith("W"):
        encoder_length = int(min(12, max(4, min_series_len // 2)))
        prediction_length = int(min(4, max(2, encoder_length // 3)))
    elif freq.startswith("M"):
        encoder_length = int(min(24, max(6, min_series_len // 2)))
        prediction_length = int(min(3, max(1, encoder_length // 4)))
    else:
        encoder_length = int(min(30, max(7, min_series_len // 2)))
        prediction_length = int(min(7, max(2, encoder_length // 3)))

    logger.info(f"Adjusted params -> encoder={encoder_length}, prediction={prediction_length}")

    # --- Buat time_idx per series ---
    df = df.copy()
    df["time_idx"] = df.groupby(group_col).cumcount()

    # --- Tentukan cutoff otomatis ---
    max_time_idx = df["time_idx"].max()
    cutoff = int(max_time_idx - prediction_length)
    logger.info(f"Building dataset with cutoff={cutoff}, max_time_idx={max_time_idx}")

    try:
        training = TimeSeriesDataSet(
            df[df["time_idx"] <= cutoff],
            time_idx="time_idx",
            target=target_col,
            group_ids=[group_col],
            min_encoder_length=encoder_length,
            max_encoder_length=encoder_length,
            min_prediction_length=prediction_length,
            max_prediction_length=prediction_length,
            time_varying_unknown_reals=[target_col],
            allow_missing_timesteps=True,  # ✅ penting untuk weekly/monthly
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            df,
            predict=True,
            stop_randomization=True
        )

        logger.info(f"Dataset created successfully with {len(training)} samples.")
        return training, validation

    except Exception as e:
        logger.error(f"Error from dataset : {e}")
        raise