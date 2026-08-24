import pickle
import pandas as pd
from train import engineer_features, encode_cat, apply_target_encoding

from fastapi import FastAPI
import uvicorn
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    gender: Literal['Male', 'Female'] = Field(..., alias='Gender')
    age: int = Field(..., ge=0, alias='Age')
    drivinglicense: Literal[0, 1] = Field(..., alias='Driving_License')
    regioncode: int = Field(..., ge=0, alias='Region_Code')
    previouslyinsured: Literal[0, 1] = Field(..., alias='Previously_Insured')
    vehicleage: Literal['< 1 Year', '1-2 Year', '> 2 Years'] = Field(..., alias='Vehicle_Age')
    vehicledamage: Literal['Yes', 'No'] = Field(..., alias='Vehicle_Damage')
    annualpremium: float = Field(..., ge=0, alias='Annual_Premium')
    policysaleschannel: int = Field(..., ge=0, alias='Policy_Sales_Channel')
    vintage: int = Field(..., ge=0, alias='Vintage')

class PredictionResponse(BaseModel):
    convert_prob: float
    convert: bool
    convert_response: Literal['Interested in vehicle insurance', 'Not interested in vehicle insurance']

MODEL_FILE = 'insurance_model.bin'

def load_artifacts(filename=MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f_in:
        artifacts = pickle.load(f_in)
    return artifacts

def predict_customer(customer, artifacts):
    df = pd.DataFrame([customer])
    df = engineer_features(df)
    df = encode_cat(df)
    df = df[artifacts['num_cols'] + artifacts['cat_cols']]
    df = apply_target_encoding(
        df,
        artifacts['region_rates'],
        artifacts['channel_rates'],
        artifacts['global_rate'],
    )

    X_sc = artifacts['scaler'].transform(df)

    proba = artifacts['model'].predict_proba(X_sc)[:, 1][0]
    pred = (proba >= artifacts['threshold']).astype(int)

    return proba, pred

def predict(customer: Customer) -> PredictionResponse:
    artifacts = load_artifacts()
    proba, pred = predict_customer(customer.model_dump(by_alias=True), artifacts)
    response = "Not interested in vehicle insurance" if pred == 0 else "Interested in vehicle insurance"

    return PredictionResponse(
        convert_prob=proba,
        convert=pred,
        convert_response=response
    )


if __name__ == '__main__':
    customer_ex = Customer(
        gender='Male',
        age=35,
        drivinglicense=1,
        regioncode=28.0,
        previouslyinsured=0,
        vehicleage='1-2 Year',
        vehicledamage='Yes',
        annualpremium=32000.0,
        policysaleschannel=26.0,
        vintage=150,
    )

    response = predict(customer_ex)
 
    print(f'Response probability: {response.convert_prob}')
    print(f'Convert value: {response.convert}')
    print(f'Predicted response: {response.convert_response}')


