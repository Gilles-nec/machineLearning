import requests
import json

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

#define the requests payload
data = {'cat_features': cat_features}
data = json.dumps(data)

#Send post request and get the response 
response = requests.post("http://127.0.0.1:8000/inference", data)
print(response.status_code)
print(response.content)