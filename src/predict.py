import requests
import random
import hydra
from omegaconf import DictConfig

@hydra.main(config_name=None)
def predict(cfg: DictConfig):
    example_version = cfg.example_version
    hostname = cfg.hostname
    port = int(cfg.port)
    random_state = int(cfg.random_state)

    # Set the seed for reproducibility
    random.seed(random_state)

    # Generate a random feature sample
    feature_sample = {
        "feature1": random.random(),
        "feature2": random.random(),
        "feature3": random.random()
    }

    url = f"http://{hostname}:{port}/invocations"
    headers = {"Content-Type": "application/json"}
    data = {
        "columns": list(feature_sample.keys()),
        "data": [list(feature_sample.values())]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        prediction = response.json()
        print(f"Prediction for example version {example_version}: {prediction}")
    else:
        print(f"Failed to get prediction. Status code: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    predict()
