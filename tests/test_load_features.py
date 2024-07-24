import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import load_features

class TestLoadFeatures(unittest.TestCase):

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
        self.data_version = '1.0.0'
        self.df_combined = pd.concat([self.X, self.y], axis=1)

    @patch('src.data.pd.concat')
    @patch('src.data.Client')
    @patch('src.data.zenml.save_artifact')
    def test_load_features_success(self, mock_save_artifact, mock_Client, mock_concat):
        # Mock the concatenation of X and y
        mock_concat.return_value = self.df_combined

        # Mock the Client and its methods
        mock_client = MagicMock()
        mock_Client.return_value = mock_client

        # Mock the artifact listing
        mock_artifact = MagicMock()
        mock_artifact.load.return_value = self.df_combined
        mock_client.list_artifact_versions.return_value.items = [mock_artifact]

        # Call the function
        saved_X, saved_y = load_features(self.X, self.y, self.data_version)

        # Assert the calls
        mock_concat.assert_called_once_with([self.X, self.y], axis=1)
        mock_save_artifact.assert_called_once_with(data=self.df_combined, name='features_target', tags=[self.data_version])
        mock_Client.assert_called_once()
        mock_client.list_artifact_versions.assert_called_once_with(name="features_target", sort_by="version")
        mock_artifact.load.assert_called_once()

        # Assert the returned DataFrames
        pd.testing.assert_frame_equal(saved_X, self.X)
        pd.testing.assert_frame_equal(saved_y, self.y)

    @patch('src.data.pd.concat')
    @patch('src.data.Client')
    @patch('src.data.zenml.save_artifact')
    def test_load_features_no_artifacts(self, mock_save_artifact, mock_Client, mock_concat):
        # Mock the concatenation of X and y
        mock_concat.return_value = self.df_combined

        # Mock the Client and its methods
        mock_client = MagicMock()
        mock_Client.return_value = mock_client

        # Mock the artifact listing with no items
        mock_client.list_artifact_versions.return_value.items = []

        # Call the function and assert it raises an IndexError
        with self.assertRaises(IndexError):
            load_features(self.X, self.y, self.data_version)

        # Assert the calls
        mock_concat.assert_called_once_with([self.X, self.y], axis=1)
        mock_save_artifact.assert_called_once_with(data=self.df_combined, name='features_target', tags=[self.data_version])
        mock_Client.assert_called_once()
        mock_client.list_artifact_versions.assert_called_once_with(name="features_target", sort_by="version")

if __name__ == '__main__':
    unittest.main()
