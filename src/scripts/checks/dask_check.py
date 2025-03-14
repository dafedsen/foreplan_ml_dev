from dask.distributed import Client
client = Client("tcp://127.0.0.1:8790")

# This will return the Python path for all workers
python_paths = client.run(lambda: __import__('sys').executable)
print(python_paths)