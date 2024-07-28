"""Script to create sample"""
import pandas as pd
import hydra
import dvc.api

from great_expectations.data_context import DataContext
from great_expectations.data_context import FileDataContext

from zenml.client import Client
import zenml
import yaml
import subprocess

from gx_checkpoint import validate_initial_data
from omegaconf import OmegaConf

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

import os
import pickle

@hydra.main(config_path="../configs", config_name = "main", version_base=None)
def sample_data(cfg = None):
    """Main function of script"""
    data_url = dvc.api.get_url(
        path=cfg.data.path,
        remote=cfg.data.remote,
        repo=cfg.data.repo,
        rev=cfg.data.version
    )
    version = cfg.version
    sample_size = cfg.data.sample_size

    # Take a sample of the data
    data = pd.read_csv(data_url)
    sample = data.iloc[:int(len(data) * sample_size * version)]
    sample.to_csv('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv', index=False)

def get_data_version(
    config_file='/home/aleksandr-vashchenko/Desktop/mlops-final-project-iu-2024/configs/data_version.yaml'
):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    if not 'version' in config:
        raise KeyError("No version in config")
    if config['version'] is None:
        raise KeyError("Version is None")
    return str(config['version'])

def read_datastore():
    data_version = get_data_version()
    df = pd.read_csv('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv')
    return df, data_version

def preprocess_data(df):

    def vectorize_text(name, text, max_features=None):
        vectorizer = TfidfVectorizer(max_features=max_features)
        vectorizer.fit(text)
        with open(f'../vectorizer_{name}.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)

        return vectorizer.transform(text).toarray()

    # get rid of nan values
    df.loc[df['item_name'].isna(), 'item_name'] = ""
    df.loc[df['item_description'].isna(), 'item_description'] = ""
    df.loc[df['item_variation'].isna(), 'item_variation'] = ""
    df.dropna(inplace=True, subset=['category'])

    # scale continuous values
    scaler = StandardScaler()
    
    scaler.fit(df[['price', 'stock']].to_numpy())
    df[['price', 'stock']] = scaler.transform(df[['price', 'stock']].to_numpy())

    with open('../scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # break down item creation date
    df['item_creation_date'] = pd.to_datetime(df['item_creation_date'])
    df['year'] = df['item_creation_date'].dt.year
    df['month'] = df['item_creation_date'].dt.month
    df['day'] = df['item_creation_date'].dt.day
    df.drop(columns=['item_creation_date'], inplace=True)

    # assign labels to the target feature
    encoder = LabelEncoder()
        
    encoder.fit(df['category'].to_numpy())
    df['category'] = encoder.transform(df['category'].to_numpy())

    with open('../encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

    # prepare X and y datasets
    numerical_features = [
        'itemid', 'shopid', 'price', 'stock', 'cb_option',
        'is_preferred', 'sold_count', 'year', 'month', 'day'
    ]
    X = df[numerical_features].to_numpy()
    y = df['category'].to_numpy()

    X = np.concatenate(
        (
            X,
            vectorize_text('item_name', df['item_name'], max_features=100),
            vectorize_text('item_description', df['item_description'], max_features=100),
            vectorize_text('item_variation', df['item_variation'], max_features=100)
        ),
        axis=1
    )

    scaler2 = StandardScaler()
    X = scaler2.fit_transform(X)

    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=['category'])

    return X, y

def validate_features(X, y, name_suite="transformed_suite"):

    df = pd.concat([X, y], axis=1)

    context = FileDataContext(project_root_dir="./services")

    ds = context.sources.add_or_update_pandas(name="transformed_data")
    da = ds.add_dataframe_asset(name = "transformed_dataframe_asset")
    
    br = da.build_batch_request(dataframe=df)

    context.add_or_update_expectation_suite(name_suite)
    
    validator = context.get_validator(
        batch_request=br,
        expectation_suite_name=name_suite
    )

    for column in df.columns:
        validator.expect_column_values_to_not_be_null(column)

    validator.save_expectation_suite(
    	discard_failed_expectations = False
    )

    batch_list = da.get_batch_list_from_batch_request(br)
    batch_request_list = [batch.batch_request for batch in batch_list]

    validations = [
        {
            "batch_request": batch.batch_request,
            "expectation_suite_name": name_suite
        }
        for batch in batch_list
    ]

    checkpoint = context.add_or_update_checkpoint(
        name="transformed_checkpoint",
        validations=validations
    )

    results = checkpoint.run()

    if not results['success']:
        raise ValueError("Feature validation failed")

    X = df.drop('category', axis=1)
    y = df[['category']]

    return X, y

def load_features(X, y, data_version):

    df = pd.concat([X, y], axis=1)

    zenml.save_artifact(data=df, name='features_target', tags=[data_version])

    client = Client()

    l = client.list_artifact_versions(name="features_target", sort_by="version").items
    l.reverse()
    df = l[0].load()

    saved_X = df.drop('category', axis=1)
    saved_y = df[['category']]

    return saved_X, saved_y

def validate_sample(**kwargs):
    validate_initial_data("~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv")

@hydra.main(config_path="../configs", config_name="data_version", version_base=None)
def version_sample(cfg=None):

    cfg.version += 1
    OmegaConf.save(cfg, "configs/data_version.yaml")

    subprocess.run([
        'sh',
        '~/Desktop/mlops-final-project-iu-2024/scripts/version_sample.sh',
        'data/samples/sample.csv',
        str(cfg.version)
    ])


if __name__ == "__main__":
    sample_data()
