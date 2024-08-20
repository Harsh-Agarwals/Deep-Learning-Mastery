import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import os
import mlflow 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("train.csv")
numerical_cols = df.select_dtypes(exclude=["string", "bool", "object"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["string", "object"]).columns.tolist()

categorical_cols.remove("Loan_Status")
categorical_cols.remove('Loan_ID')

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)

df[numerical_cols] = df[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

df['LoanAmount'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome'] = np.log(df['TotalIncome'])

df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=23)
grid_params_rf = {
    'n_estimators': [200, 400, 700],
    'max_depth': [10, 20, 30],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [50, 100]
}
grid_rf = GridSearchCV(estimator=rf, param_grid=grid_params_rf, cv=5, n_jobs=-1, scoring='accuracy', verbose=0)
model_rf = grid_rf.fit(X_train, y_train)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=32)
grid_params_dt = {
    'max_depth': [3, 5, 7, 9, 11, 13],
    'criterion': ['gini', 'entropy']
}
grid_dt = GridSearchCV(estimator=dt, param_grid=grid_params_dt, cv=5, n_jobs=-1, scoring='accuracy', verbose=0)
model_dt = grid_dt.fit(X_train, y_train)

# Logistic Regression
lr = LogisticRegression(random_state=26)
grid_params_lr = {
    'C': [100, 10, 1.0, 0.1, 0.01],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_lr = GridSearchCV(estimator=lr, param_grid=grid_params_lr, cv=5 ,n_jobs=-1, scoring='accuracy', verbose=0)
model_lr = grid_lr.fit(X_train, y_train)

def eval_metrics(preds, real):
    accuracy = accuracy_score(preds, real)
    f1_sc = f1_score(preds, real)
    fpr,tpr, _ = roc_curve(preds, real)
    aucx = auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%aucx)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return(accuracy, f1_sc, aucx)

mlflow.set_experiment("loan_prediction2")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

def mlflow_logging(model, X, y, name):
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("name", name)
        preds = model.predict(X)
        (accuracy, f1s, aucx) = eval_metrics(preds, y)
        # metrics
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1-score", f1s)
        mlflow.log_metric("AUC", aucx)
        # params
        mlflow.log_params(model.best_params_)
        # artifacts
        mlflow.log_artifact("plots/ROC_curve.png")
        # log model
        mlflow.sklearn.log_model(model, name)

        mlflow.end_run()


mlflow_logging(model_rf, X_test, y_test, 'RandomForest')
mlflow_logging(model_dt, X_test, y_test, 'DecisionTree')
mlflow_logging(model_lr, X_test, y_test, 'LogisticRegression')