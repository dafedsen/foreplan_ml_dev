from dotenv import load_dotenv
import os
load_dotenv()

from sqlalchemy import create_engine, event
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text

import uuid

import datetime
import pytz

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
SCHEMA = os.getenv("DB_SCHEMA")

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

def generate_unique_execution_id(conn):
    while True:
        new_id = str(uuid.uuid4())
        with conn.connection.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {SCHEMA}.bplan_logging_ml WHERE execution_id = %s", (new_id,))
            if cur.fetchone() is None:
                return new_id

def logging_ml(id_user, id_prj, id_version, id_cust, model_name, status, description, details, 
               start_date=None, end_date=None, execution_id=None):  
    
    date = datetime.datetime.now(pytz.timezone('Asia/Jakarta')).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(start_date, (int, float)):
        start_date = datetime.datetime.fromtimestamp(start_date)
    if isinstance(end_date, (int, float)):
        end_date = datetime.datetime.fromtimestamp(end_date)

    with engine.connect() as conn:
        if execution_id is None:
            execution_id = generate_unique_execution_id(conn)

        query = text(f"""
            INSERT INTO {SCHEMA}.bplan_logging_ml (
                id_user, id_prj, id_version, id_cust,
                model_name, status, description, details, datetime,
                start_date, end_date, execution_id
            ) VALUES (
                :id_user, :id_prj, :id_version, :id_cust,
                :model_name, :status, :description, :details, :datetime,
                :start_date, :end_date, :execution_id
            )
        """)
        params = {
            "id_user": int(id_user),
            "id_prj": int(id_prj),
            "id_version": int(id_version),
            "id_cust": int(id_cust),
            "model_name": str(model_name),
            "status": str(status),
            "description": str(description),
            "details": str(details),
            "datetime": str(date),
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "execution_id": str(execution_id),
        }

        conn.execute(query, params)
        conn.commit()

    engine.dispose()
    return execution_id

# logging_ml(2, 101, 1, 1, "Linear Regression", "RUNNING", "API ACCEPTED", "main.py : process_forecast")
