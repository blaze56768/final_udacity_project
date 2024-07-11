""" This module tests the root and the prediction end points """
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_get_root():
    """ Testing main page for a succesful response"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Hi": "This application analyzes census data to predict whether an individual's annual income is likely to surpass $50,000."}


def test_post_predict_up():
    """ Test a scenario where an individual's income is below $50,000. """

    r = client.post("/predict-income", json={
        "age": 42,
        "workclass": "Private",
        "fnlgt": 159449,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 5178,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": "<=50K"}


def test_post_predict_down():
    """ Test a scenario where an individual's income is higher than $50,000. """
    r = client.post("/predict-income", json={
        "age": 36,
        "workclass": "Private",
        "fnlgt": 155537,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"Income prediction": "<=50K"}
