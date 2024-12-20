# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from .ml import process_data, train_model, save_model, inference, compute_model_metrics, save_data
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('starter/data/clean_census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

with open('starter/model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('starter/model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('starter/model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)

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

column = 'workclass'

with open('starter/starter/slice_output.txt', 'w') as f:
    filter_data = pd.DataFrame()
    for name in test[column].unique():
        filter_data = test[test[column] == name]
        

        X_test_feature, y_test_feature, encoder, lb = process_data(
            filter_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )

        # Evaluate
        f.write(f'Evaluate on sclicing with value {name} from {column} feature.\n')
        y_pred_feature = inference(model, X_test_feature)
        model.score(X_test_feature, y_test_feature)
        precision, recall, fbeta = compute_model_metrics(y_test_feature, y_pred_feature)
        f.write(f"precision: {precision}, recal: {recall}, fbeta: {fbeta}\n")
        f.write('--------------------\n')
