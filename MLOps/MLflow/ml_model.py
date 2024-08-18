import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import argparse

TARGET="quality"

def get_data():
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        data = pd.read_csv(data_url, sep=";")
        return data
    except Exception as e:
        raise e

def evaluate(preds, test):
    mse = mean_squared_error(preds, test)
    mae = mean_absolute_error(preds, test)
    r2_s = r2_score(preds, test)
    return mse, mae, r2_s

def model(learning_rate, n_estimator, criterion, max_depth):
    df = get_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimator, criterion=criterion, max_depth=max_depth)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse, mae, r2_s = evaluate(preds, y_test)

    mlflow.set_experiment("ml-model")
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
    with mlflow.start_run(run_name="gdb-model"):
        mlflow.log_param("learning-rate", learning_rate)
        mlflow.log_param("n-estimator", n_estimator)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max-depth", max_depth)

        mlflow.log_metrics({
            "mse": mse,
            "mae": mae,
            "r2_score": r2_s
        })

        mlflow.sklearn.log_model(model, "gradient-boosting-model")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ml-model with mlflow")
    parser.add_argument("--learning_rate", type=float, default=0.2)
    parser.add_argument("--n_estimator", type=int, default=100)
    parser.add_argument("--criterion", type=str, default="friedman_mse")
    parser.add_argument("--max_depth", type=int, default=5)
    parsed_args = parser.parse_args()
    model(parsed_args.learning_rate, parsed_args.n_estimator, parsed_args.criterion, parsed_args.max_depth)
