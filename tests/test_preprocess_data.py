import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data import preprocess_data, sample_data

# # Helper function to create a sample DataFrame
# def create_sample_df():
#     # data = {
#     #     'itemid': [1, 2, 3],
#     #     'shopid': [10, 20, 30],
#     #     'price': [100.0, 200.0, 300.0],
#     #     'stock': [50, 60, 70],
#     #     'cb_option': [1, 2, 3],
#     #     'is_preferred': [0, 1, 0],
#     #     'sold_count': [5, 15, 25],
#     #     'item_creation_date': ['2022-01-01', '2023-02-02', None],
#     #     'item_name': ['item1', 'item2', None],
#     #     'item_description': ['desc1', 'desc2', 'desc3'],
#     #     'item_variation': [None, 'var2', 'var3'],
#     #     'category': ['cat1', 'cat2', 'cat1']
#     # }
#     return pd.DataFrame(data)

def test_preprocess_data():
    
    sample_data()
    df = pd.read_csv('~/Desktop/mlops-final-project-iu-2024/data/samples/sample.csv')
    
    X, y = preprocess_data(df)
    
    # Test dimensions
    assert X.shape == (len(df), 310)  # Expect 3 rows and 109 features: 9 numerical + 100 from each text field
    assert y.shape == (len(df), 1)    # Expect 3 rows and 1 target column
    
    # Test missing value handling
    assert df['item_name'].isna().sum() == 0
    assert df['item_description'].isna().sum() == 0
    assert df['item_variation'].isna().sum() == 0
    assert df['category'].isna().sum() == 0
    
    # Test numerical feature scaling
    scaler = StandardScaler()
    scaler.fit(df[['price', 'stock']])
    scaled_values = scaler.transform(df[['price', 'stock']])
    assert np.allclose(X[['2', '3']].values, scaled_values)  # Column indices 0 and 1 correspond to 'price' and 'stock'
    
    # Test label encoding
    encoder = LabelEncoder()
    encoder.fit(df['category'])
    encoded_labels = encoder.transform(df['category'])
    assert np.array_equal(y.values.flatten(), encoded_labels)

if __name__ == "__main__":
    pytest.main()
