from flask import Flask, render_template, request, jsonify
import pandas as pd
from prediction_model.predict import predict_values

app = Flask(__name__)

def transform_to_integers(data):
    columns_to_convert = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

    for col in columns_to_convert:
        try:
            data[col] = int(data[col])
        except ValueError:
            print(f"Warning: Could not convert '{col}' to integer. Value: {data[col]}")
    return data

def transform_to_float(data):
    columns_to_convert = ['Loan_Amount_Term', 'Credit_History']

    for col in columns_to_convert:
        try:
            data[col] = float(data[col])
        except ValueError:
            print(f"Warning: Could not convert '{col}' to integer. Value: {data[col]}")
    return data

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/loan-prediction", methods=['POST'])
def home_page():
    try:
        if request.method == 'POST':
            request_data = dict(request.form)
            request_data = {k:v for k, v in request_data.items()}
            request_data = transform_to_integers(request_data)
            request_data = transform_to_float(request_data)
            df = pd.DataFrame([request_data]).iloc[0, :]
            result = predict_values([df])
            prediction = 'Eligible' if result['Predictions'][0] == 'Y' else 'Ineligible'
            return jsonify(final=prediction)
            # if request.args.get("submit") == None:
            #     return 
            # elif request.args.get("submit") == "":
            #     return "<h1>INVALID, please try again!</h1>"
            # else:
            #     data = {}
            #     for i in request.args:
            #         data[i] = request.args.get(i)
            #     del data['submit']
            #     data = transform_to_integers(data)
            #     data = transform_to_float(data)
            #     df = pd.DataFrame([data]).iloc[0, :]
            #     result = predict_values([df])
            #     prediction = 'Eligible' if result['Predictions'][0] == 'Y' else 'Ineligible'
            #     return jsonify(final=prediction)
    except Exception as e:
        print(f"Error: {e}")
        return str(e)
        
if __name__ == "__main__":
    app.run(debug=True)