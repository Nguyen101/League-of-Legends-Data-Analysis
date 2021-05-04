import os 
import pickle 
from flask import Flask, jsonify, request 

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Hello World!</h1>"

@app.route("/predict", methods = ["GET"])
def predict():
    first_blood = request.args.get("first_blood", "")
    first_tower = request.args.get("first_tower", "")
    first_inhib = request.args.get("first_inhib", "")
    first_baron = request.args.get("first_baron", "")
    first_dragon = request.args.get("first_dragon", "")
    first_riftherald = request.args.get("first_riftherald", "")


    prediction = "1"

    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400

if __name__ == "__main__":
    app.run()