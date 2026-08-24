import pickle
import pandas as pd
from train import engineer_features, encode_cat, apply_target_encoding

from fastapi import FastAPI
import uvicorn
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gender: Literal['Male', 'Female']
    age: int = Field(..., ge=0)
    drivinglicense: Literal[0, 1]
    regioncode: int = Field(..., ge=0)
    previouslyinsured: Literal[0, 1]
    vehicleage: Literal['< 1 Year', '1-2 Year', '> 2 Years']
    vehicledamage: Literal['Yes', 'No']
    annualpremium: float = Field(..., ge=0)
    policysaleschannel: int = Field(..., ge=0)
    vintage: int = Field(..., ge=0)

MODEL_FILE = 'insurance_model.bin'

def load_artifacts(filename=MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f_in:
        artifacts = pickle.load(f_in)
    return artifacts

def predict(customer, artifacts):
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

if __name__ == '__main__':
    customer_ex = {
        'Gender': 'Male',
        'Age': 35,
        'Driving_License': 1,
        'Region_Code': 28.0,
        'Previously_Insured': 0,
        'Vehicle_Age': '1-2 Year',
        'Vehicle_Damage': 'Yes',
        'Annual_Premium': 32000.0,
        'Policy_Sales_Channel': 26.0,
        'Vintage': 150,
    }
 
    artifacts = load_artifacts()
    proba, pred = predict(customer_ex, artifacts)
 
    print(f'Response probability: {proba:.4f}')
    print(f'Predicted response:   {pred} (threshold={artifacts["threshold"]})')


