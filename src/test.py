import requests

url = "http://127.0.0.1:8000/linear_regression/"
data = {"id_prj": 101, "version_name": "Version 1"}

response = requests.post(url, json=data)
print(response.json())