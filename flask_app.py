import os 
import pickle 
import ast
import copy
from mysklearn.myclassifiers import MyRandomForestClassifier
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

    print(first_blood)
    print(first_tower)
    print(first_inhib)
    print(first_baron)
    print(first_dragon)
    print(first_riftherald)

    best_trees = []
    with open("best_tree.txt", "r") as data:
        best_tree = ast.literal_eval(data.read())

    rf = MyRandomForestClassifier()
    rf.trees = copy.deepcopy(best_tree)

    x = [int(first_blood), int(first_tower), int(first_inhib), int(first_baron), int(first_dragon), int(first_riftherald)]
    prediction = rf.predict([x])

    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400

if __name__ == "__main__":
    app.run()