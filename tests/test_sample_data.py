import os
import pytest
import pandas as pd
from unittest import mock

# Mock the hydra.main decorator
from hydra import main as hydra_main

# Import the sample_data function
# import sys
# sys.path.append('/mnt/c/GitHub/mlops-final-project-iu-2024')
from src.data import sample_data

@pytest.fixture
def mock_cfg():
    class MockCfg:
        class Data:
            path = 'data/raw/data.csv'
            remote = 'origin'
            repo = '.'
            version = 'main'
            sample_size = 0.1
            random_state = 42
        
        data = Data()

    return MockCfg()

@pytest.fixture
def mock_dvc_api(monkeypatch):
    def mock_get_url(path, remote, repo, rev):
        return f'{repo}/{path}'
    
    monkeypatch.setattr('dvc.api.get_url', mock_get_url)

@pytest.fixture
def mock_csv(monkeypatch):
    def mock_read_csv(filepath):
        return pd.DataFrame({
            'col1': range(10),
            'col2': range(10)
        })

    monkeypatch.setattr('pandas.read_csv', mock_read_csv)

def test_sample_data_creates_sample_file(mock_cfg, mock_dvc_api, mock_csv, tmpdir, monkeypatch):
    # Mock os.makedirs and os.system
    monkeypatch.setattr('os.makedirs', lambda path, exist_ok: None)
    monkeypatch.setattr('os.system', lambda command: 0)
    
    # Run the sample_data function
    with mock.patch('hydra.main', return_value=hydra_main):
        sample_data(mock_cfg)

    # Check if the sample file is created
    sample_path = '/home/datapaf/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv'
    assert os.path.exists(sample_path), f"{sample_path} does not exist."

    # Check if the sample file has the correct number of rows
    sample = pd.read_csv(sample_path)
    assert len(sample) == 10, f"Sample size is incorrect: expected 10, got {len(sample)}"
