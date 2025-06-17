# Manual test that the API is working
import random

import requests
from tqdm import tqdm

from src.input_preprocessing import LightGBMDataResult

# host_url = "http://localhost:8000"
host_url = "http://52.53.160.136"
path_to_lightgbm_data = "data/model/input_inference/test/subsample_0.1_42"
num_samples_to_predict = 5
seed = 123
random.seed(seed)

print("Testing health check...")
response = requests.get(f"{host_url}/health")
print(response.json())

print("Testing readiness check...")
response = requests.get(f"{host_url}/ready")
print(response.json())

print("Testing prediction...")
# Load a sample prediction request
sample_data = LightGBMDataResult.load(path_to_lightgbm_data)

for i in tqdm(range(num_samples_to_predict)):
    row_id = random.randint(0, len(sample_data.data) - 1)
    print(f"Sample {i} row ID: {row_id}")
    sample_request = {
        "data": sample_data.data.to_dict(orient="records")[row_id],
        "feature_names": sample_data.feature_names,
        "sample": "test",
    }

    print(f"Sending prediction request {i + 1} of {num_samples_to_predict}...")
    response = requests.post(f"{host_url}/predict", json=sample_request)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("--------------------------------")
    print("--------------------------------")
