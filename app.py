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
    return render_template("index.html")

@app.route("/predict",methods=["post"])
def predict():
    from flask import request
    bedroom=int(request.form["bedrooms"])
    bathroom=float(request.form["bathrooms"])
    sqft_living=int(request.form["sqft_living"])
    sample_house=pd.DataFrame([{"bedrooms": bedroom,
        "bathrooms": bathroom,
        "sqft_living": sqft_living,
        "sqft_lot": 5000,
        "floors": 1,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "yr_built": 2000,
        "yr_renovated": 0,}])
    
    for col in city_columns:
        sample_house[col]=0
    for col in statezip_columns:
        sample_house[col] = 0
    
    print(sample_house.shape)
    
    prediction=model.predict(sample_house)
    return f"predicted_price : ${prediction[0]:,.2f}"

if __name__=="__main__":
    app.run(debug=True)

