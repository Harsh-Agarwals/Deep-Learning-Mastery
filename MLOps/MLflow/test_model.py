import mlflow
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

lr_run_ID = "15a371cd5b06492fa93a778189ee757b"
lr_uri = f"runs:/{lr_run_ID}/model"
lr_model_name = "LogisticRegression"

rf_run_ID = "a546e8a4e25a424eb3b25b9f7b67fce8"
rf_uri = f"runs:/{rf_run_ID}/model"
rf_model_name = "RandomForestClassifier"

with mlflow.start_run():
    mlflow.register_model(model_uri=lr_uri, name=lr_model_name)
    mlflow.register_model(model_uri=rf_uri, name=rf_model_name)

lr_model_uri = f"runs:/{lr_run_ID}/{lr_model_name}"
rf_model_uri = f"runs:/{rf_run_ID}/RandomForest"

model_lr = mlflow.sklearn.load_model(model_uri=lr_model_uri)
model_rf = mlflow.sklearn.load_model(model_uri=rf_model_uri)

data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]

pred1 = model_lr.predict(pd.DataFrame(data))
pred2 = model_rf.predict(pd.DataFrame(data))

print(f"pred_lr: {pred1}, pred_rf: {pred2}")