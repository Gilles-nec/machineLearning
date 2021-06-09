# Tests on 3 model functions
import sklearn
import numpy
import pandas as pd

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model

# Load data
data = pd.read_csv("data/clean_census.csv")
train, test = train_test_split(data, test_size=0.20)


def test_type_of_process_data():
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
    assert isinstance(process_data(
        train, categorical_features=cat_features, label="salary", training=True
        ), tuple)


def test_type_of_train_model():
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

    # Train and save a model.
    model = train_model(X_train, y_train)
    assert isinstance(model, sklearn.ensemble._forest.RandomForestClassifier)


def test_type_of_predict():
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

    # Train and save a model.
    model = train_model(X_train, y_train)
    assert isinstance(model.predict(X_train), numpy.ndarray)
