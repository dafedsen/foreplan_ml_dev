from fastapi import FastAPI, Request
from celery.result import AsyncResult
from pydantic import BaseModel

from tasks import run_forecast

app = FastAPI()

class RequestData(BaseModel):
    id_prj: int
    version_name: str

@app.post("/process_forecast/")
def process_forecast(data: RequestData):

    id_prj = data.id_prj
    version_name = data.version_name

    result = run_forecast.delay(id_prj, version_name)

    return {'task_id': result.task_id, 'version_name': version_name, 'id_prj': id_prj}