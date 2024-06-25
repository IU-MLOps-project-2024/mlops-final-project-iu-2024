import os
import pandas as pd
import hydra
import dvc.api

@hydra.main(config_path="../configs", config_name = "main", version_base=None)
def sample_data(cfg = None):
    data_url = dvc.api.get_url(
        path=cfg.data.path,
        remote=cfg.data.remote,
        repo=cfg.data.repo,
        rev=cfg.data.version
    )
    sample_size = cfg.data.sample_size

    # Take a sample of the data
    data = pd.read_csv(data_url)
    sample = data.sample(frac=sample_size, random_state=cfg.data.random_state)

    os.makedirs('data/samples', exist_ok=True)
    sample.to_csv('data/samples/sample.csv', index=False)

    # Stage the sample data file for DVC and push the changes with DVC
    os.system('dvc add data/samples/sample.csv')
    os.system('dvc push')

if __name__ == "__main__":
    sample_data()
