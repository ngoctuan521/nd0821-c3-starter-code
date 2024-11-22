from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_greeting():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == "Welcome to my project."

def test_post_datapoint_lessthan_50k():
    body = {
        "age": 23,
        "workclass": "Private",
        "fnlgt": 122272,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 30,
        "native_country": "United-States"
    }
    response = client.post('/predict', json=body)
    assert response.status_code == 200
    assert response.json() == {'predict': '<=50K'}

def test_post_datapoint_morethan_50k():
    body = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    response = client.post('/predict', json=body)
    assert response.status_code == 200
    assert response.json() == {'predict': '>50K'}
