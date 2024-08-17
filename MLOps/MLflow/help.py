import mlflow
mlflow.set_tracking_uri("https://localhost:5000")

print(mlflow.get_tracking_uri())

expt_id = mlflow.create_experiment('Loan Prediction')

with mlflow.start_run(run_name='DecisionTreeClassifier') as run:
    mlflow.set_tag("version", "1.0.0")

mlflow.end_run()

n_estimator = 10
criterion = 'gini'

mlflow.log_param('n_estimator', n_estimator)
mlflow.log_param('criterion', criterion)
mlflow.log_metric('accuracy')