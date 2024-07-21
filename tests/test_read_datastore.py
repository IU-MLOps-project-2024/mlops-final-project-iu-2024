import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import read_datastore

class TestReadDatastore(unittest.TestCase):

    @patch('src.data.pd.read_csv')
    @patch('src.data.get_data_version')
    def test_read_datastore_success(self, mock_get_data_version, mock_read_csv):
        # Mock return value of get_data_version
        mock_get_data_version.return_value = '1.0.0'
        
        # Mock return value of pd.read_csv
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df
        
        # Call the function
        df, data_version = read_datastore()
        
        # Assert the mock was called with the expected arguments
        mock_read_csv.assert_called_once_with('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv')
        mock_get_data_version.assert_called_once()
        
        # Assert the return values are as expected
        self.assertEqual(data_version, '1.0.0')
        pd.testing.assert_frame_equal(df, mock_df)

    @patch('src.data.pd.read_csv')
    @patch('src.data.get_data_version')
    def test_read_datastore_different_version(self, mock_get_data_version, mock_read_csv):
        # Mock return value of get_data_version
        mock_get_data_version.return_value = '2.1.3'
        
        # Mock return value of pd.read_csv
        mock_df = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})
        mock_read_csv.return_value = mock_df
        
        # Call the function
        df, data_version = read_datastore()
        
        # Assert the mock was called with the expected arguments
        mock_read_csv.assert_called_once_with('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv')
        mock_get_data_version.assert_called_once()
        
        # Assert the return values are as expected
        self.assertEqual(data_version, '2.1.3')
        pd.testing.assert_frame_equal(df, mock_df)

    @patch('src.data.pd.read_csv', side_effect=FileNotFoundError('File not found'))
    @patch('src.data.get_data_version')
    def test_read_datastore_file_not_found(self, mock_get_data_version, mock_read_csv):
        # Mock return value of get_data_version
        mock_get_data_version.return_value = '1.0.0'
        
        # Call the function and assert it raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            read_datastore()

        # Assert the mock was called with the expected arguments
        mock_read_csv.assert_called_once_with('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv')
        mock_get_data_version.assert_called_once()

if __name__ == '__main__':
    unittest.main()
