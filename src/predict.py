import numpy as np
import requests
# import sklearn.metrics
# import torch
# import zenml
import hydra
from model import load_features
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from omegaconf import open_dict
# import sklearn
import json

def predict(cfg):
    X_test, y_test = load_features(
        name="features_target",
        version=cfg.example_version,
        # target_col=cfg.datasets.target_col
    )

    inputs = X_test[[0], :]
    # inputs = inputs[:, np.newaxis]
    inputs = inputs.tolist()

    input_data = {
        "inputs": inputs
    }
    y_true = y_test[0]

    response = requests.post(
        url=f"http://{cfg.hostname}:{cfg.port}/invocations",
        data=json.dumps(input_data),
        headers={"Content-Type": "application/json"},
    )

    print(response)
    print(response.json())
    print("y_true", y_true)

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    predict(cfg)

if __name__ == "__main__":
    main()