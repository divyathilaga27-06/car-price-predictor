from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("linear_regression_car_price.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    new_car = pd.DataFrame({
        "Car ID": [int(request.form["car_id"])],
        "Brand": [request.form["brand"]],
        "Year": [int(request.form["year"])],
        "Engine Size": [float(request.form["engine_size"])],
        "Fuel Type": [request.form["fuel_type"]],
        "Transmission": [request.form["transmission"]],
        "Mileage": [int(request.form["mileage"])],
        "Condition": [request.form["condition"]],
        "Model": [request.form["model"]]
    })

    prediction = model.predict(new_car)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Price: {prediction:,.2f}"
    )
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)