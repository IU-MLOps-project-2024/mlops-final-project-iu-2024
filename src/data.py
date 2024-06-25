"""Script to create sample"""
import os
import pandas as pd
import hydra
import dvc.api
import great_expectations as gx

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
    sample = data.sample(frac=sample_size, random_state=cfg.data.random_state)

    os.makedirs('data/samples', exist_ok=True)
    sample.to_csv('data/samples/sample.csv', index=False)

    # Stage the sample data file for DVC and push the changes with DVC
    os.system('dvc add data/samples/sample.csv')
    os.system('dvc push')


def validate_initial_data():
    """Create expectations and validate initial data"""
    context = gx.get_context(context_root_dir="services/gx")
    ds = context.sources.add_or_update_pandas(name="Test_Pandas")
    da = ds.add_csv_asset(
        name="asset01",
        filepath_or_buffer="data/raw/Test_Pandas.csv"
    )
    br = da.build_batch_request()
    context.add_or_update_expectation_suite("my_expectation_suite")
    context.list_expectation_suite_names()
    validator = context.get_validator(
        batch_request=br,
        expectation_suite_name="my_expectation_suite",
    )
    print(validator.head())
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
        min="01-01-2016",
        max="01-02-2016"
    )
    column = "price"

    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_values_to_be_between(column=column, min_value=0, max_value=1e10)
    validator.expect_column_quantile_values_to_be_between(
        column=column,
        quantile_ranges={
            "quantiles": [0, 0.5, 1],
            "value_ranges": [[0, 1], [5, 30], [1e5, 1e10]]
        }
    )
    column = "stock"

    validator.expect_column_values_to_not_be_null(column=column)
    validator.expect_column_values_to_be_between(column=column, min_value=0, max_value=1e10)
    validator.expect_column_quantile_values_to_be_between(
        column=column,
        quantile_ranges={
            "quantiles": [0, 0.5, 1],
            "value_ranges": [[0, 1], [100, 1000], [1e5, 1e10]]
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


    context.add_or_update_checkpoint(
        name="my_checkpoint",
        validations=[
            {
                "batch_request": br,
                "expectation_suite_name": "my_expectation_suite"
            }
        ]
    )
    context.build_data_docs()

if __name__ == "__main__":
    sample_data()
    validate_initial_data()
