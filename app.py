from flask import Flask
from flask import render_template
import joblib
import pandas as pd

app=Flask(__name__)

model=joblib.load("models/house_price_model.pkl")

city_columns = joblib.load("models/city_columns.pkl")
statezip_columns = joblib.load("models/statezip_columns.pkl")

@app.route("/")
def home():
    cities = [col.replace("city_", "") for col in city_columns]

    statezips = [
        col.replace("statezip_", "")
        for col in statezip_columns
    ]

    print(cities[:5])
    print(statezips[:5])

    return render_template(
        "index.html",
        cities=cities,
        statezips=statezips
    )


@app.route("/predict",methods=["post"])
def predict():
    from flask import request
    bedroom=int(request.form["bedrooms"])
    bathroom=float(request.form["bathrooms"])
    sqft_living=int(request.form["sqft_living"])
    floors=int(request.form["floors"])
    waterfront=int(request.form["waterfront"])
    sqft_lot=int(request.form["sqft_lot"])
    view=int(request.form["view"])
    condition=int(request.form["condition"])
    yr_built=int(request.form["yr_built"])
    city=request.form["city"]
    statezip=request.form["statezip"]
    sample_house=pd.DataFrame([{"bedrooms": bedroom,
        "bathrooms": bathroom,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_living,
        "sqft_basement": 0,
        "yr_built": yr_built,
        "yr_renovated": 0,}])
    
    for col in city_columns:
        sample_house[col]=0
    for col in statezip_columns:
        sample_house[col] = 0

    city_col=f"city_{city}"
    if city_col in sample_house.columns:
        sample_house[city_col]=1

    zip_col=f"zip_{statezip}"
    if zip_col in sample_house.columns:
        sample_house[zip_col]=1
    
    print(sample_house.shape)
    
    prediction=model.predict(sample_house)
    return f"predicted_price : ${prediction[0]:,.2f}"

if __name__=="__main__":
    app.run(debug=True)

