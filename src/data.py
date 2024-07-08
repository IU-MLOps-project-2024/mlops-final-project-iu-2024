"""Script to create sample"""
import pandas as pd
import hydra
import dvc.api

@hydra.main(config_path="../configs", config_name = "main", version_base=None)
def sample_data(cfg = None):
    """Main function of script"""
    data_url = dvc.api.get_url(
        path=cfg.data.path,
        remote=cfg.data.remote,
        repo=cfg.data.repo,
        rev=cfg.data.version
    )
    sample_size = cfg.data.sample_size

    # Take a sample of the data
    data = pd.read_csv(data_url)
    sample = data.iloc[:int(len(data) * sample_size)]
    sample.to_csv('data/samples/sample.csv', index=False)

if __name__ == "__main__":
    sample_data()
