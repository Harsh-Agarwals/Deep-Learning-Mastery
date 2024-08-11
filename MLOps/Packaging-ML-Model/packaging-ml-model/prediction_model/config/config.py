import os
import pathlib
import sys
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT, 'datasets')

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MODEL_NAME = 'classification.pkl'
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TARGET = "Loan_Status"
FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

FEATURES_TO_ENCODE = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

FEATURES_TO_MODIFY = ['ApplicantIncome']
FEATURES_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']
LOG_FEATURES = ['ApplicantIncome', 'LoanAmount']
