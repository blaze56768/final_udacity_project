import requests


df = {  "age": 28,
        "workclass": "Private",
        "fnlgt": 183175,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
r = requests.post('https://censusprojectfinal-c17c513fc053.herokuapp.com/predict-income/', json=df)
print(r.json())
print(r.status_code)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())

## Experimental push to check CI