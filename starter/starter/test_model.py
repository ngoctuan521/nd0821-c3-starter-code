import pickle
import numpy as np
from .ml import inference

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)
with open('data/data.pkl', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

def test_evaluate_train_set():
    """Check accuracy of test set, should > 0.7."""
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.7

def test_inference():
    """Check number of sample after inference."""
    _, n_feature = X_train.shape
    n_sample = 5
    samples = np.random.normal(loc=0.0, scale=1.0, size=(n_sample, n_feature))
    y_pred = inference(model, samples)
    assert y_pred.shape[0] == n_sample

def test_sample():
    """Check data is normalized."""
    assert np.all(X_train <= 1) and np.all(X_test <= 1)
