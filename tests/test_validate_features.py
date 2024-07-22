import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import validate_features

class TestValidateFeatures(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
            'itemid': [1, 2, 3],
            'shopid': [10, 20, 30],
            'cb_option': [0, 1, 1],
            'is_preferred': [0, 1, 0],
            'item_creation_date': ['2017-01-01', '2018-01-01', '2019-01-01'],
            'price': [100, 200, 300],
            'stock': [10, 20, 30]
        })
        self.y = pd.DataFrame({
            'category': ['Mobile & Gadgets', "Women's Apparel", 'Health & Beauty']
        })

    @patch('src.data.FileDataContext')
    @patch('src.data.pd.concat')
    def test_validate_features_success(self, mock_concat, mock_FileDataContext):
        # Mock the concatenation of X and y
        df = pd.concat([self.X, self.y], axis=1)
        mock_concat.return_value = df

        # Mock the FileDataContext and its methods
        mock_context = MagicMock()
        mock_FileDataContext.return_value = mock_context

        # Mock the methods of the context
        mock_datasource = MagicMock()
        mock_context.sources.add_or_update_pandas.return_value = mock_datasource
        mock_dataframe_asset = MagicMock()
        mock_datasource.add_dataframe_asset.return_value = mock_dataframe_asset
        mock_batch_request = MagicMock()
        mock_dataframe_asset.build_batch_request.return_value = mock_batch_request
        mock_validator = MagicMock()
        mock_context.get_validator.return_value = mock_validator
        mock_checkpoint = MagicMock()
        mock_context.add_or_update_checkpoint.return_value = mock_checkpoint
        mock_results = {'success': True}
        mock_checkpoint.run.return_value = mock_results

        # Call the function
        X, y = validate_features(self.X, self.y)

        # Assert the calls
        mock_FileDataContext.assert_called_once_with(project_root_dir="../services")
        mock_context.sources.add_or_update_pandas.assert_called_once_with(name="transformed_data")
        mock_datasource.add_dataframe_asset.assert_called_once_with(name="transformed_dataframe_asset")
        mock_dataframe_asset.build_batch_request.assert_called_once_with(dataframe=df)
        mock_context.add_or_update_expectation_suite.assert_called_once_with("transformed_suite")
        mock_context.get_validator.assert_called_once_with(
            batch_request=mock_batch_request,
            expectation_suite_name="transformed_suite"
        )
        mock_validator.save_expectation_suite.assert_called_once_with(discard_failed_expectations=False)
        mock_context.add_or_update_checkpoint.assert_called_once()
        mock_checkpoint.run.assert_called_once()


    @patch('src.data.FileDataContext')
    @patch('src.data.pd.concat')
    def test_validate_features_failure(self, mock_concat, mock_FileDataContext):
        # Mock the concatenation of X and y
        df = pd.concat([self.X, self.y], axis=1)
        mock_concat.return_value = df

        # Mock the FileDataContext and its methods
        mock_context = MagicMock()
        mock_FileDataContext.return_value = mock_context

        # Mock the methods of the context
        mock_datasource = MagicMock()
        mock_context.sources.add_or_update_pandas.return_value = mock_datasource
        mock_dataframe_asset = MagicMock()
        mock_datasource.add_dataframe_asset.return_value = mock_dataframe_asset
        mock_batch_request = MagicMock()
        mock_dataframe_asset.build_batch_request.return_value = mock_batch_request
        mock_validator = MagicMock()
        mock_context.get_validator.return_value = mock_validator
        mock_checkpoint = MagicMock()
        mock_context.add_or_update_checkpoint.return_value = mock_checkpoint
        mock_results = {'success': False}
        mock_checkpoint.run.return_value = mock_results

        # Call the function and assert it raises a ValueError
        with self.assertRaises(ValueError):
            validate_features(self.X, self.y)

        # Assert the calls
        mock_FileDataContext.assert_called_once_with(project_root_dir="../services")
        mock_context.sources.add_or_update_pandas.assert_called_once_with(name="transformed_data")
        mock_datasource.add_dataframe_asset.assert_called_once_with(name="transformed_dataframe_asset")
        mock_dataframe_asset.build_batch_request.assert_called_once_with(dataframe=df)
        mock_context.add_or_update_expectation_suite.assert_called_once_with("transformed_suite")
        mock_context.get_validator.assert_called_once_with(
            batch_request=mock_batch_request,
            expectation_suite_name="transformed_suite"
        )
        mock_validator.save_expectation_suite.assert_called_once_with(discard_failed_expectations=False)
        mock_context.add_or_update_checkpoint.assert_called_once()
        mock_checkpoint.run.assert_called_once()

if __name__ == '__main__':
    unittest.main()
