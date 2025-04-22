from dotenv import load_dotenv
import os
load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text

import datetime

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800
)

def logging_ml(id_user, id_prj, id_version, id_cust, model_name, status, description, details):  
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    query = text("""
        INSERT INTO bplan_logging_ml (
            id_user, id_prj, id_version, id_cust,
            model_name, status, description, details, datetime
        ) VALUES (
            :id_user, :id_prj, :id_version, :id_cust,
            :model_name, :status, :description, :details, :datetime
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
        "datetime": str(date)
    }

    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()

logging_ml(2, 101, 1, 1, "Linear Regression", "RUNNING", "API ACCEPTED", "main.py : process_forecast")
