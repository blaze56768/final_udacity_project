"""
This module includes unit tests for the ML model

"""
from pathlib import Path
import logging
import pandas as pd
import pytest
from data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


CENSUS_DATASET_PATH = 'data/census_cleaned.csv'
TRAINED_MODEL_PATH = 'model/model.pkl'

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(name='census_data')
def census_data():
    """
    Fixture to provide census data for unit tests.
    """
    yield pd.read_csv(CENSUS_DATASET_PATH)


def test_census_data_loading(census_data):
    """
    Verify the integrity of the loaded census data.
    """
    assert isinstance(census_data, pd.DataFrame)
    assert census_data.shape[0] > 0
    assert census_data.shape[1] > 0


def test_model_type():
    """
    Ensure the loaded model is of the correct type.
    """
    trained_model = load_artifact(TRAINED_MODEL_PATH)
    assert isinstance(trained_model, RandomForestClassifier)


def test_data_processing(census_data):
    """
    Validate the data processing and splitting procedure.
    """
    training_set, _ = train_test_split(census_data, test_size=0.20)
    X_processed, y_processed, _, _ = process_data(training_set, cat_features, label='salary')
    assert len(X_processed) == len(y_processed)