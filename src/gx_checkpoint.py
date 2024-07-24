from great_expectations.data_context import FileDataContext

def validate_initial_data(csv_path, name_suit="my_expectation_suite"):
    context = FileDataContext(project_root_dir="../services")
    ds = context.sources.add_or_update_pandas(name="Test_Pandas")
    da = ds.add_csv_asset(
        name = "asset01",
        filepath_or_buffer=csv_path,
    )
    br = da.build_batch_request()
    context.add_or_update_expectation_suite(name_suit)
    validator = context.get_validator(
        batch_request=br,
        expectation_suite_name=name_suit
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
            "expectation_suite_name": "my_expectation_suite"
        }
        for batch in batch_list
    ]

    checkpoint = context.add_or_update_checkpoint(
        name="my_validator_checkpoint",
        validations=validations
    )

    checkpoint_result = checkpoint.run()

    assert checkpoint_result.success

if __name__ == "__main__":
    validate_initial_data('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv')
