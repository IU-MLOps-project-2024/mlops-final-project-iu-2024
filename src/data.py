"""Script to create sample"""
import pandas as pd
import hydra
import dvc.api

# from great_expectations.dataset.pandas_dataset import PandasDataset
# from great_expectations.core.batch import BatchRequest
# from great_expectations.checkpoint import LegacyCheckpoint
from great_expectations.data_context import DataContext
from great_expectations.data_context import FileDataContext

from zenml.client import Client
# from zenml.artifact_stores import LocalArtifactStore
# from zenml.artifacts.data_artifact import DataArtifact
# from zenml.io import fileio
import zenml
import yaml

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
    sample.to_csv('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv', index=False)

def get_data_version(
    config_file='/home/datapaf/Desktop/mlops-final-project-iu-2024/configs/data_version.yaml'
):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return str(config['version'])

def read_datastore():
    data_version = get_data_version()
    df = pd.read_csv('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv')
    return df, data_version

def preprocess_data(df):
    target_feature_name = 'category'
    X = df.drop(target_feature_name, axis=1)
    y = df[[target_feature_name]]
    return X, y

def validate_features(X, y, name_suite="transformed_suite"):

    df = pd.concat([X, y], axis=1)

    # context = DataContext(project_root_dir="../services")
    context = FileDataContext(project_root_dir="../services")

    ds = context.sources.add_or_update_pandas(name="transformed_data")
    da = ds.add_dataframe_asset(name = "transformed_dataframe_asset")
    
    br = da.build_batch_request(dataframe=df)

    context.add_or_update_expectation_suite(name_suite)
    
    validator = context.get_validator(
        batch_request=br,
        expectation_suite_name=name_suite
    )

    column = "itemid"
    validator.expect_column_values_to_be_unique(column=column)
    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_min_to_be_between(column=column, min=0)

    column = "shopid"
    validator.expect_column_min_to_be_between(column=column, min=0)
    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_values_to_be_of_type(column=column, type_="int64")

    column = "cb_option"
    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_distinct_values_to_be_in_set(
        column=column,
        value_set=[0, 1]
    )
    validator.expect_column_quantile_values_to_be_between(
        column=column,
        quantile_ranges={
            "quantiles": [0, 0.2, 1],
            "value_ranges": [[0, 0], [0, 1], [1, 1]]
        }
    )

    column = "is_preferred"
    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_distinct_values_to_be_in_set(
        column=column,
        value_set=[0, 1]
    )
    validator.expect_column_quantile_values_to_be_between(
        column=column,
        quantile_ranges={
            "quantiles": [0, 0.8, 1],
            "value_ranges": [[0, 0], [0, 1], [1, 1]]
        }
    )

    column = "item_creation_date"
    validator.expect_column_values_to_be_dateutil_parseable(column=column)
    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_min_to_be_between(
        column=column,
        min="01-01-2016"
    )

    column = "price"
    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_values_to_be_between(column=column, min_value=0, max_value=1e10)
    validator.expect_column_quantile_values_to_be_between(
        column=column,
        quantile_ranges={
            "quantiles": [0, 0.5, 1],
            "value_ranges": [[0, 1], [5, 30], [100, 1e10]]
        }
    )

    column = "stock"
    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_values_to_be_between(column=column, min_value=0, max_value=1e10)
    validator.expect_column_quantile_values_to_be_between(
        column=column,
        quantile_ranges={
            "quantiles": [0, 0.5, 1],
            "value_ranges": [[0, 1], [100, 1000], [1000, 1e10]]
        }
    )

    column = "category"
    validator.expect_column_distinct_values_to_equal_set(
        column=column,
        value_set=['Mobile & Gadgets', "Women's Apparel", "Men's Wear",
           'Health & Beauty', 'Bags & Luggage', 'Toys, Kids & Babies',
           'Sports & Outdoors', "Men's Shoes", "Women's Shoes",
           'Jewellery & Accessories', 'Home & Living', 'Games & Hobbies',
           'Pet Accessories', 'Design & Crafts', 'Computers & Peripherals',
           'Watches', 'Miscellaneous ', 'Home Appliances', 'Food & Beverages',
           'Tickets & Vouchers']
    )
    validator.expect_column_most_common_value_to_be_in_set(
        column=column,
        value_set=["Women's Apparel", "Mobile & Gadgets", "Jewellery & Accessories", "Men's Wear"]
    )
    validator.expect_column_to_exist(column=column)

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

if __name__ == "__main__":
    sample_data()
