
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model= joblib.load("house_price_model.pkl")

@app.route("/")
def home():
  return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
  data = request.form
  features=np.array([[
      data["bedrooms"],
      data["bathrooms"],
      data["sqft_living"],
      data["floors"],
      data["waterfront"],
      data["view"],
      data["condition"],
      data["grade"],
      data["sqft_above"],
      data["sqft_basement"]
  ]])
  prediction = model.predict(features)[0]
  return jsonify({
      "pridicted_price":round(float(prediction),2)
  })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)