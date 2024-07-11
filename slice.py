"""
This module outputs the performance of the model on slices of the data for categorical features.
"""
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from data import load_artifact, process_data
from model import compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

DATASET_FILE = 'data/census_cleaned.csv'
MODEL_DIRECTORY = 'model'

# Features representing demographic categories
demographic_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def evaluate_model_slices():
    """ Assess model performance across different demographic subgroups """

    dataset = pd.read_csv(DATASET_FILE)
    _, evaluation_set = train_test_split(dataset, test_size=0.20)

    trained_model = load_artifact(os.path.join(MODEL_DIRECTORY, 'model.pkl'))
    feature_encoder = load_artifact(os.path.join(MODEL_DIRECTORY, 'encoder.pkl'))
    label_binarizer = load_artifact(os.path.join(MODEL_DIRECTORY, 'lb.pkl'))

    slice_performance = []

    for feature in demographic_features:
        for category in evaluation_set[feature].unique():
            subset = evaluation_set[evaluation_set[feature] == category]

            X_eval, y_eval, _, _ = process_data(
                subset,
                demographic_features,
                label="salary",
                encoder=feature_encoder,
                lb=label_binarizer,
                training=False)

            y_predicted = trained_model.predict(X_eval)

            precision, recall, fbeta = compute_model_metrics(y_eval, y_predicted)
            result = f"{feature} - {category} :: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
            slice_performance.append(result)

            with open('slice_metrics/slice_output.txt', 'w') as output_file:
                for result in slice_performance:
                    output_file.write(result + '\n')

    logging.info("Slice-wise performance metrics saved to slice_output.txt")


if __name__ == '__main__':
    evaluate_model_slices()


