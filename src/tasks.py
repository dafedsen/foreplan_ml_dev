from celery import Celery

from scripts import linear_regression_gpu

celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery.task
def run_linear_regression(id_prj, version_name):
    result = linear_regression_gpu.run_model(id_prj, version_name)
    return {'id_prj': id_prj, 'version_name': version_name, 'result': result}