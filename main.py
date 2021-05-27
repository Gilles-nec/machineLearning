# Put the code for your API here.
import pickle
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.model import inference
from starter.ml.data import process_data

app = FastAPI()


class ModelData(BaseModel):
    cat_features: List[str]


@app.get("/")
def read_root():
    return "Welcome to the main page"


@app.post("/inference", response_model=ModelData, status_code=201)
async def run_inference(data: ModelData):
    modelFile = './model/model.sav'
    model = pickle.load(open(modelFile, 'rb'))
    csv = pd.read_csv("data/clean_census.csv")
    train, test = train_test_split(csv, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=data.cat_features, 
        label="salary", training=True
        )
    X_test, y_test, encoder_test, lb_test = process_data(
        test, categorical_features=data.cat_features, 
        label="salary", training=True
        )
    pred = inference(model, X_test)
    print(pred)
    