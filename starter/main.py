# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel, Field
from starter.ml import inference, process_data

app = FastAPI()

class DataField(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., exmaple=13)
    marital_status: str = Field(..., exmaple="Never-married")
    occupation: str = Field(..., exmaple="Adm-clerical")
    relationship: str = Field(..., exmaple="Not-in-family")
    race: str = Field(..., exmaple="White")
    sex: str = Field(..., exmaple="Male")
    capital_gain: int = Field(..., exmaple=2174)
    capital_loss: int = Field(..., exmaple=0)
    hours_per_week: int = Field(..., exmaple=40)
    native_country: str = Field(..., exmaple="United-States")

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)

@app.get('/')
async def greeting():
    print('greeting################')
    return "Welcome to my project."

@app.post("/predict")
async def predict(data_point: DataField):
    transform_datapoint = {}
    # for k, v in data_point.items():
    for k, v in data_point.__dict__.items():
        k = k.replace("_", "-")
        transform_datapoint[k] = v
    sample = pd.DataFrame([transform_datapoint])
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    sample, _, _, _ = process_data(X=sample, categorical_features=cat_features,
        training=False, encoder=encoder, lb=lb)
    pred = inference(model, sample)

    rs = {}
    if pred[0] == 0:
        rs['predict'] = '<=50K'
    else:
        rs['predict'] = '>50K'

    return rs

