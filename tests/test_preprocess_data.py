import unittest
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import preprocess_data

class TestPreprocessData(unittest.TestCase):

    def test_preprocess_data_success(self):
        # Create a sample DataFrame
        data = {
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'category': ['A', 'B', 'C']
        }
        df = pd.DataFrame(data)
        
        # Expected outputs
        expected_X = df.drop('category', axis=1)
        expected_y = df[['category']]
        
        # Call the function
        X, y = preprocess_data(df)
        
        # Assert the outputs are as expected
        pd.testing.assert_frame_equal(X, expected_X)
        pd.testing.assert_frame_equal(y, expected_y)

    def test_preprocess_data_different_structure(self):
        # Create a different sample DataFrame
        data = {
            'col1': [7, 8, 9],
            'col2': [10, 11, 12],
            'category': ['X', 'Y', 'Z']
        }
        df = pd.DataFrame(data)
        
        # Expected outputs
        expected_X = df.drop('category', axis=1)
        expected_y = df[['category']]
        
        # Call the function
        X, y = preprocess_data(df)
        
        # Assert the outputs are as expected
        pd.testing.assert_frame_equal(X, expected_X)
        pd.testing.assert_frame_equal(y, expected_y)

    def test_preprocess_data_category_missing(self):
        # Create a DataFrame without 'category' column
        data = {
            'col1': [7, 8, 9],
            'col2': [10, 11, 12]
        }
        df = pd.DataFrame(data)
        
        # Call the function and assert it raises KeyError
        with self.assertRaises(KeyError):
            preprocess_data(df)

if __name__ == '__main__':
    unittest.main()
