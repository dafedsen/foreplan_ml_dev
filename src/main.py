import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "scripts")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import time
import logging

from tasks import run_forecast_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# try:
#     client = Client("tcp://127.0.0.1:8790")
#     logger.info("Connected to Dask scheduler successfully.")
# except Exception as e:
#     logger.error("Failed to connect to Dask scheduler. Continuing without it.")
#     client = None

# @app.get("/")
# async def root():
#     try:
#         client = Client("tcp://127.0.0.1:8790")
#         return {"status": "Dask connected!"}
#     except Exception as e:
#         return {"error": str(e)}
    
class RequestData(BaseModel):
    id_user: int
    id_prj: int
    version_name: str

# def upload_file(client):
#     client.upload_file(os.path.join(BASE_DIR, "tasks.py"))
#     client.upload_file(os.path.join(BASE_DIR, "scripts/linear_regression_gpu.py"))
#     client.upload_file(os.path.join(BASE_DIR, "scripts/prophet_cpu.py"))

@app.post("/process_forecast/")
async def process_forecast(data: RequestData, background_tasks: BackgroundTasks):
    # upload_file(client)

    id_user = data.id_user
    id_prj = data.id_prj
    version_name = data.version_name    
    
    logger.info(f"Processing forecast for project {id_prj} with version {version_name}.")

    try:
        background_tasks.add_task(run_forecast_main, id_user, id_prj, version_name, background_tasks)
        return {"version_name": version_name, "id_prj": id_prj, "status": "Processing"}
    except Exception as e:
        logger.error(f"Error in process_forecast: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    
def run_forecast_main(id_user, id_prj, version_name, background_tasks):
    try:
        run_forecast_task(id_user, id_prj, version_name, background_tasks)
        logger.info(f"Forecast is in progress for project {id_prj} with version {version_name}.")
    except Exception as e:
        logger.error(f"Error in forecast: {str(e)}")