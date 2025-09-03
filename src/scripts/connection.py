import cudf as cd
import pandas as pd

from sqlalchemy import create_engine, event
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text
from psycopg2.extras import execute_values
import math
import datetime
import pytz
import requests

from dotenv import load_dotenv
import os
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
SCHEMA = os.getenv("DB_SCHEMA")

SHUTDOWN = os.getenv("SHUTDOWN")

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800
)

@event.listens_for(engine, "connect")
def set_search_path(dbapi_connection, connection_record):
    schema = os.getenv("DB_SCHEMA", "public")
    cursor = dbapi_connection.cursor()
    cursor.execute(f'SET search_path TO {schema}')
    cursor.close()

def get_models_postgre(id_prj, id_cust):

    query = text(f"SELECT distinct model_name FROM {SCHEMA}.v_projhist WHERE id_prj = :id_prj AND partition_cust_id = :id_cust")
    params = {
        "id_prj": int(id_prj),
        "id_cust": int(id_cust)
    }

    df = pd.read_sql(query, engine, params=params)
    df_db = list(df['model_name'])

    return df_db

def get_dataset_postgre(id_prj, version_name, id_cust):
    cols = 'id_prj, id_version, version_name, level1, level2, hist_date, hist_value, flag_adj'
    query = text(f"SELECT {cols} FROM {SCHEMA}.v_prm_adj WHERE id_prj = :id_prj AND version_name = :version_name AND partition_cust_id = :id_cust")
    params = {
        "id_prj": int(id_prj),
        "version_name": str(version_name),
        "id_cust": int(id_cust)
    }
    df = pd.read_sql(query, engine, params=params)
    df = df.drop(columns=['id'])
    df['hist_date'] = pd.to_datetime(df['hist_date'], format='%Y-%m-%d')

    df_db = cd.DataFrame.from_pandas(df)
    df_db['hist_value'] = df_db['hist_value'].astype(float)
    df_db.sort_values('hist_date')

    return df_db

def get_dataset(id_prj, version_name, id_cust):
    cols = 'id_prj, id_version, version_name, level1, level2, hist_date, hist_value'
    query = text(f"SELECT {cols} FROM {SCHEMA}.v_ml_final_hist_adj WHERE id_prj = :id_prj AND version_name = :version_name AND partition_cust_id = :id_cust")
    params = {
        "id_prj": int(id_prj),
        "version_name": str(version_name),
        "id_cust": int(id_cust)
    }
    df_adj = pd.read_sql(query, engine, params=params)
    df_adj['hist_date'] = pd.to_datetime(df_adj['hist_date'], format='%Y-%m-%d')
    df_adj['flag_adj'] = 1

    query = text(f"SELECT {cols} FROM {SCHEMA}.v_ml_final_hist_non_adj WHERE id_prj = :id_prj AND version_name = :version_name AND partition_cust_id = :id_cust")
    params = {
        "id_prj": int(id_prj),
        "version_name": str(version_name),
        "id_cust": int(id_cust)
    }
    df_non_adj = pd.read_sql(query, engine, params=params)
    df_non_adj['hist_date'] = pd.to_datetime(df_non_adj['hist_date'], format='%Y-%m-%d')
    df_non_adj['flag_adj'] = 0

    df = pd.concat([df_adj, df_non_adj])
    df_db = cd.DataFrame.from_pandas(df)
    df_db['hist_value'] = df_db['hist_value'].astype(float)
    df_db.sort_values('hist_date')

    return df_db

def get_dataset_adj_postgre(id_prj, version_name, id_cust):
    cols = 'id_prj, id_version, version_name, level1, level2, hist_date, hist_value'
    query = text(f"SELECT {cols} FROM {SCHEMA}.v_ml_final_hist_adj WHERE id_prj = :id_prj AND version_name = :version_name AND partition_cust_id = :id_cust")
    params = {
        "id_prj": int(id_prj),
        "version_name": str(version_name),
        "id_cust": int(id_cust)
    }
    df = pd.read_sql(query, engine, params=params)
    df['hist_date'] = pd.to_datetime(df['hist_date'], format='%Y-%m-%d')

    df_db = cd.DataFrame.from_pandas(df)
    df_db['hist_value'] = df_db['hist_value'].astype(float)
    df_db.sort_values('hist_date')

    return df_db

def get_dataset_non_adj_postgre(id_prj, version_name, id_cust):
    cols = 'id_prj, id_version, version_name, level1, level2, hist_date, hist_value'
    query = text(f"SELECT {cols} FROM {SCHEMA}.v_ml_final_hist_non_adj WHERE id_prj = :id_prj AND version_name = :version_name AND partition_cust_id = :id_cust")
    params = {
        "id_prj": int(id_prj),
        "version_name": str(version_name),
        "id_cust": int(id_cust)
    }
    df = pd.read_sql(query, engine, params=params)
    df['hist_date'] = pd.to_datetime(df['hist_date'], format='%Y-%m-%d')

    df_db = cd.DataFrame.from_pandas(df)
    df_db['hist_value'] = df_db['hist_value'].astype(float)
    df_db.sort_values('hist_date')

    return df_db

def get_setting_postgre(id_prj, version_name, id_cust):
    cols = 'id_prj, id_prj_prc, fcast_type, fcast_type_number, version_name, adj_include, level1, level2, model_name, out_std_dev, ad_smooth_method'
    query = text(f"SELECT {cols} FROM {SCHEMA}.v_prj_prc WHERE id_prj = :id_prj AND version_name = :version_name AND partition_cust_id = :id_cust")
    params = {
        "id_prj": int(id_prj),
        "version_name": str(version_name),
        "id_cust": int(id_cust)
    }
    df = pd.read_sql(query, engine, params=params)
    df_db = cd.DataFrame.from_pandas(df)

    return df_db

def get_id_cust_from_id_prj(id_prj):

    query = text(f"SELECT id_cust FROM {SCHEMA}.v_prj_cust WHERE id_prj = :id_prj")
    params = {
        "id_prj": int(id_prj)
    }

    with engine.connect() as conn:
        result = conn.execute(query, params)
        result = result.fetchone()
        id_cust = result[0]

    return id_cust

def send_process_result(df, id_cust):
    conn = engine.raw_connection()
    cursor = conn.cursor()

    if isinstance(df, cd.DataFrame):    
        df = df.to_pandas()

    partition_name = f'fplan_prj_prc_result_custid_{id_cust}'  # Target partition
    fallback_partition = 'fplan_prj_prc_result_custid_other'
    id_prj_prc_values = df['id_prj_prc'].unique().tolist()
    id_model_values = df['id_model'].unique().tolist()  # Get all project IDs

    try:
        # Step 1: DELETE existing records for the given id_prj_prc
        delete_query = f"DELETE FROM {SCHEMA}.fplan_prj_prc_result WHERE id_prj_prc IN %s AND id_model IN %s"
        cursor.execute(delete_query, (tuple(id_prj_prc_values), tuple(id_model_values)))

        check_partition_query = "SELECT to_regclass(%s)"
        cursor.execute(check_partition_query, (partition_name,))
        result = cursor.fetchone()
        target_partition = partition_name if result[0] else fallback_partition

        # Step 2: INSERT new records
        columns = [col for col in df.columns if col != 'id']  # Exclude 'id' if auto-increment
        insert_columns = ', '.join(columns)
        values_placeholder = ', '.join(['%s'] * len(columns))
        print(values_placeholder)
        insert_query = f"INSERT INTO {target_partition} ({insert_columns}) VALUES {values_placeholder}"

        # Convert DataFrame to list of tuples excluding 'id' if necessary
        records = [tuple(row[col] for col in columns) for _, row in df.iterrows()]

        if records:
            execute_values(cursor, insert_query.replace(f"VALUES {values_placeholder}", "VALUES %s"), records)
            conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")

    finally:
        cursor.close()
        conn.close()

def send_process_evaluation(df, id_cust):
    conn = engine.raw_connection()
    cursor = conn.cursor()

    if isinstance(df, cd.DataFrame):    
        df = df.to_pandas()

    df['err_value'] = df['err_value'].apply(lambda x: round(x, 3))

    partition_name = f'fplan_prj_prc_eval_custid_{id_cust}'  # Target partition
    fallback_partition = 'fplan_prj_prc_eval_custid_other'
    id_prj_prc_values = df['id_prj_prc'].unique().tolist()  # Get all project IDs
    id_model_values = df['id_model'].unique().tolist()

    try:
        # Step 1: DELETE existing records for the given id_prj_prc
        delete_query = f"DELETE FROM {SCHEMA}.fplan_prj_prc_eval WHERE id_prj_prc IN %s AND id_model IN %s"
        cursor.execute(delete_query, (tuple(id_prj_prc_values), tuple(id_model_values)))

        check_partition_query = "SELECT to_regclass(%s)"
        cursor.execute(check_partition_query, (partition_name,))
        result = cursor.fetchone()
        target_partition = partition_name if result[0] else fallback_partition

        # Step 2: INSERT new records
        columns = [col for col in df.columns if col != 'id']
        insert_columns = ', '.join(columns)
        values_placeholder = ', '.join(['%s'] * len(columns))  # Generate placeholders dynamically
        print(values_placeholder)
        insert_query = f"INSERT INTO {target_partition} ({insert_columns}) VALUES {values_placeholder}"

        # Convert DataFrame to list of tuples excluding 'id' if necessary
        records = [tuple(row[col] for col in columns) for _, row in df.iterrows()]

        if records:
            execute_values(cursor, insert_query.replace(f"VALUES {values_placeholder}", "VALUES %s"), records)
            conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")

    finally:
        cursor.close()
        conn.close()


def update_process_status(id_prj, id_version, status):

    query = text(f"UPDATE {SCHEMA}.fplan_prj_prc SET status = :status WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
        "status": str(status)
    }
    
    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()

def check_update_process_status_success(id_prj, id_version):

    query = text(f"SELECT status_progress FROM {SCHEMA}.fplan_prj_prc WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
    }
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        result = result.fetchone()
        status_progress = result[0]

    if status_progress >= 100:
        update_process_status(id_prj, id_version, "SUCCESS")
        return True
    
    return False

def reset_process_status_progress(id_prj, id_version):

    query = text(f"UPDATE {SCHEMA}.fplan_prj_prc SET status_progress = :status_progress WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
        "status_progress": int(0)
    }
    
    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()

def update_process_status_progress(id_prj, id_version):
    query = text(f"SELECT running_model FROM {SCHEMA}.fplan_prj_prc WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
    }
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        result = result.fetchone()
        running_models = result[0]

    query = text(f"SELECT model_finished FROM {SCHEMA}.fplan_prj_prc WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
    }
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        result = result.fetchone()
        model_finished = result[0]

    status_progress = math.ceil((model_finished / running_models) * 100)
    update_date = datetime.datetime.now(pytz.timezone('Asia/Jakarta'))

    query = text(f"UPDATE {SCHEMA}.fplan_prj_prc SET status_progress = :status_progress , updated_date = :update_date WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
        "status_progress": int(status_progress),
        "update_date": update_date
    }
    
    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()

def update_running_models(id_prj, id_version, total_models):

    query = text(f"UPDATE {SCHEMA}.fplan_prj_prc SET running_model = :running_model WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
        "running_model": int(total_models)
    }
    
    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()

def reset_model_finished(id_prj, id_version):

    query = text(f"UPDATE {SCHEMA}.fplan_prj_prc SET model_finished = :model_finished WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
        "model_finished": float(0.0)
    }
    
    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()

def update_model_finished(id_prj, id_version, current_process):
    query = text(f"SELECT model_finished FROM {SCHEMA}.fplan_prj_prc WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
    }
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        result = result.fetchone()
        model_finished = result[0]

    model_finished_update = model_finished + current_process

    query = text(f"UPDATE {SCHEMA}.fplan_prj_prc SET model_finished = :model_finished WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
        "model_finished": float(model_finished_update)
    }

    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()

def update_end_date(id_prj, id_version):
    end_date = datetime.datetime.now(pytz.timezone('Asia/Jakarta'))
    query = text(f"UPDATE {SCHEMA}.fplan_prj_prc SET end_date = :end_date WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
        "end_date": end_date
    }

    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()
# def update_running_model_process(id_prj, id_version, current_process):

#     query = text("SELECT running_model FROM fplan_prj_prc WHERE id_prj = :id_prj AND id_version = :id_version")
#     params = {
#         "id_prj": int(id_prj),
#         "id_version": int(id_version),
#     }
    
#     with engine.connect() as conn:
#         result = conn.execute(query, params)
#         result = result.fetchone()
#         running_models = result[0]

#     query = text("UPDATE fplan_prj_prc SET model_finished = :model_finished WHERE id_prj = :id_prj AND id_version = :id_version")
#     params = {
#         "id_prj": int(id_prj),
#         "id_version": int(id_version),
#         "model_finished": float(current_process)
#     }

#     with engine.connect() as conn:
#         conn.execute(query, params)
#         conn.commit()


#     status_progress = math.ceil((current_process / running_models) * 100)
#     query = text("UPDATE fplan_prj_prc SET status_progress = :status_progress WHERE id_prj = :id_prj AND id_version = :id_version")
#     params = {
#         "id_prj": int(id_prj),
#         "id_version": int(id_version),
#         "status_progress": int(status_progress)
#     }
    
#     with engine.connect() as conn:
#         conn.execute(query, params)
#         conn.commit()

#     engine.dispose()

# def get_dataset(dbpath):
#     df = cd.read_csv(dbpath)
#     df['hist_date'] = cd.to_datetime(df['hist_date'], format='%Y-%m-%d')
#     df.sort_values('hist_date')
#     df['hist_value'] = df['hist_value'].astype(float)
#     return df

# def get_setting(stpath):
#     st = cd.read_csv(stpath)
#     return st

def get_forecast_time(dbase, dbset):
    df = dbase.to_pandas()
    st = dbset.to_pandas()

    time_type = st['fcast_type'][0]
    fcast_time = st['fcast_type_number'][0]

    last_date = df['hist_date'].max()

    if time_type == 'Daily':
       time_freq = 'D'
    elif time_type == 'Weekly':
       time_freq = 'W'
    elif time_type == 'Monthly':
       time_freq = 'M'
    elif time_type == 'Yearly':
       time_freq = 'Y'
    else:
       return 'Error Forecsat Time Setting'

    new_dates = pd.date_range(start=last_date, periods=fcast_time+1, freq=time_freq)
    new_dates_df = pd.DataFrame({'date': new_dates})
    new_dates_df = new_dates_df.iloc[1:]
    new_dates_df = cd.DataFrame.from_pandas(new_dates_df)
    print(new_dates_df)

    return new_dates_df

# def get_default_time(dbpath, stpath):
#     hist = get_dataset(dbpath)['hist_date']
#     hist.name = 'date'
#     hist.drop_duplicates(inplace=True)
#     hist = hist.sort_values()

#     fcast = get_forecast_time(dbpath, stpath)['date']

#     def_time = cd.concat([hist, fcast], ignore_index=True)
#     def_time = cd.DataFrame(def_time, columns=['date'])
#     def_time.reset_index(inplace=True, drop=True)

#     return def_time

# update_process_status_progress(101, 1, 0.1)

# print(models)

def ask_to_shutdown():
    if SHUTDOWN == 'True':
        try:
            api_url = "http://103.197.188.37/shutdown_vm/"
            requests.get(api_url)
        except:
            pass