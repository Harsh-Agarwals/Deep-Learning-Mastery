from flask import Flask, render_template, request
import pandas as pd
from prediction_model.predict import predict_values

app = Flask(__name__)

@app.route("/")
def home():
    return '<h3>{"Message": "Welcome to Loan Prediction app"}</h3><h4>To go to app: /loan-prediction</h4>'

@app.route("/loan-prediction", methods=['GET'])
def home_page():
    if request.method == 'GET':
        if request.args.get("submit") == None:
            return render_template("index.html")
        elif request.args.get("submit") == "":
            return "<h1>INVALID, please try again!</h1>"
        else:
            data = {}
            for i in request.args:
                data[i] = request.args.get(i)
            df = pd.DataFrame([data])
            print(df)
            result = predict_values({data})
            print(result)