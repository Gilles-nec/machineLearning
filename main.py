# Put the code for your API here.
import os
import pickle
import json
from fastapi.responses import HTMLResponse
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.model import inference
from starter.ml.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


class ModelData(BaseModel):
    cat_features: List[str]


@app.get("/")
def read_root():
    reponse = {"msg": "Welcome to the main page"}
    return reponse


@app.post("/inference", response_model=ModelData)
async def run_inference(data: ModelData):
    html_content = """
    <html>
        <head>
            <title>Inference</title>
        </head>
        <body>
            <h1>Model Inference</h1>
        </body>
    </html>
    """
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
    return HTMLResponse(content=html_content, status_code=200)
    #return pred.tolist()


