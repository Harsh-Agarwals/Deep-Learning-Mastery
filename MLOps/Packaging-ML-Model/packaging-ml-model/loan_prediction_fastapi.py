import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prediction_model.config import config
from prediction_model.processing import data_handling
from prediction_model.predict import predict_values
import uvicorn

app = FastAPI()
MODEL = data_handling.load_pipeline(config.MODEL_NAME)

class UserData(BaseModel):
    Loan_ID: str
    Gender: str 
    Married: str 
    Dependents: str
    Education: str
    Self_Employed: str 
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int 
    Loan_Amount_Term: int
    Credit_History: int
    Property_Area: str 

@app.get("/")
async def index():
    return {"Welcome": "Welcome to the loan prediction app"}

@app.post("/predict")
async def predict(user_dict: UserData):
    data = user_dict.model_dump()
    data_dict = {
        "Loan_ID": data['Loan_ID'],
        "Gender": data['Gender'],
        "Married": data['Married'],
        "Dependents": data['Dependents'],
        "Education": data['Education'],
        "Self_Employed": data['Self_Employed'],
        "ApplicantIncome": int(data['ApplicantIncome']),
        "CoapplicantIncome": int(data['CoapplicantIncome']),
        "LoanAmount": int(data['LoanAmount']),
        "Loan_Amount_Term": float(data['Loan_Amount_Term']),
        "Credit_History": float(data['Credit_History']),
        "Property_Area": data['Property_Area']
        }
    df = pd.DataFrame([data_dict]).iloc[0, :]
    result = predict_values([df])['Predictions'][0]
    return {'Result': result}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


