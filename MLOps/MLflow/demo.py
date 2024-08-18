import os
import time
import mlflow
import argparse
import numpy as np

def evaluate(p1, p2):
    return np.power(p1, 2) + np.power(p2, 2)

def main(param1, param2):
    mlflow.set_experiment("Demo-experiment")
    with mlflow.start_run(run_name="Demo Demo"):
        mlflow.set_tag("version", '1.0.0')
        mlflow.log_param('parameter_1', param1)
        mlflow.log_param('parameter_2', param2)
        metric = evaluate(param1, param2)
        mlflow.log_metric('Eval metric', metric)
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write(f"Artifact created at {time.asctime()}")
        mlflow.log_artifact("dummy")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--param1', '-p1', type=int, default=5)
    args.add_argument('--param2', '-p2', type=int, default=5)
    parsed_args = args.parse_args()
    main(parsed_args.param1, parsed_args.param2)