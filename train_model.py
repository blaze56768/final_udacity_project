""" 
Script to train the machine learning model.
"""

import logging
from sklearn.model_selection import train_test_split
from data import load_data, process_data, get_categorical_features
from model import train_model, compute_model_metrics, inference
import joblib


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load the datasetlogging.info("Importing data")
CENSUS_FILE = 'data/census_cleaned.csv'
census_data = load_data(CENSUS_FILE)

# Split data into training and testing sets
train, test = train_test_split(census_data, test_size=0.20)

# Prepare the data for model training
logging.info("Preprocessing data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=get_categorical_features(), label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=get_categorical_features(), label="salary",
     training=False, encoder=encoder, lb=lb)


# Train the machine learning model
logging.info("Training model")
model = train_model(X_train, y_train)

# Evaluate model performance
logging.info("Evaluating model on the test set")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Persist model and processing artifacts
logging.info("Saving artifacts")
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')