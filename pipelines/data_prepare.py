# from zenml.pipelines import pipeline
# from zenml.steps import step

import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import read_datastore, preprocess_data, validate_features, load_features


# Step definitions
@step
def extract() -> Tuple[
                    Annotated[
                        pd.DataFrame,
                        ArtifactConfig(
                            name="extracted_data", 
                            tags=["data_preparation"]
                        )
                    ],
                    Annotated[
                        str,
                        ArtifactConfig(name="data_version",
                        tags=["data_preparation"])]
                ]:
    df, version = read_datastore()
    return df, version

@step
def transform(df: pd.DataFrame) -> Tuple[
                    Annotated[
                        pd.DataFrame,
                        ArtifactConfig(
                            name="input_features",
                            tags=["data_preparation"]
                        )
                    ],
                    Annotated[
                        pd.DataFrame,
                        ArtifactConfig(
                            name="input_target",
                            tags=["data_preparation"]
                        )
                    ]
                ]:
    X, y = preprocess_data(df)
    return X, y

@step
def validate(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[
                    Annotated[
                        pd.DataFrame,
                        ArtifactConfig(
                            name="valid_input_features",
                            tags=["data_preparation"]
                        )
                    ],
                    Annotated[
                        pd.DataFrame,
                        ArtifactConfig(
                            name="valid_target",
                            tags=["data_preparation"]
                        )
                    ]
                ]:
    X, y = validate_features(X, y)
    return X, y

@step
def load(X: pd.DataFrame, y: pd.DataFrame, version: str) -> Tuple[
                    Annotated[
                        pd.DataFrame, 
                        ArtifactConfig(
                            name="features",
                            tags=["data_preparation"]
                        )
                    ],
                    Annotated[
                        pd.DataFrame,
                        ArtifactConfig(
                            name="target",
                            tags=["data_preparation"]
                        )
                    ]
                ]:
    X, y = load_features(X, y, version)
    return X, y

@pipeline
def data_prepare_pipeline():
    df, version = extract()
    X, y = transform(df)
    X, y = validate(X, y)
    X, y = load(X, y, version)

if __name__ == "__main__":
    data_prepare_pipeline()
