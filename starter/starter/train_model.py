# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from .ml import process_data, train_model, save_model, inference, compute_model_metrics, save_data
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('data/clean_census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=10)

train.to_csv('xoa.csv')
exit()
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train model.
model = train_model(X_train, y_train)
# import pickle
# with open('model/model.pkl', 'rb') as f:
#     model = pickle.load(f)
# Save model
save_model(model, encoder, lb)
save_data(X_train, y_train, X_test, y_test)

# Evaluate
print('Evaluate on test set')
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print(f"precision: {precision}, recal: {recall}, fbeta: {fbeta}")

# for i, score in enumerate(y_train):
#     if y_train[i]==1:
#         print(i)
#         break
# print(train.iloc[2])
# print(train.iloc[-2])

