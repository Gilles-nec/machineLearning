#Tests on 3 model functions

import pandas as pd

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model
from starter.ml.model import inference

#load data
data = pd.read_csv("data/clean_census.csv")

train, test = train_test_split(data, test_size=0.20)

def test_size_of_process_data():
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
    assert len(process_data(train, categorical_features=cat_features, label="salary", training=True)) == 4

