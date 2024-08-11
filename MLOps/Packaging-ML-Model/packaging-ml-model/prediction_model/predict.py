import numpy as np
import pandas as pd
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset

model_pipeline = load_pipeline(config.MODEL_NAME)

# Generating predictions on random data
def predict_values(data_file):
    data = pd.DataFrame(data_file)
    preds = model_pipeline.predict(data[config.FEATURES])
    output = np.where(preds==1, "Y", "N")
    result = {'Predictions': output}
    return result

# Generating predictions on test data
def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    preds = model_pipeline.predict(test_data[config.FEATURES])
    output = np.where(preds == 1, 'Y', 'N')
    # result = {'Predictions': output}
    return output

if __name__=="__main__":
    generate_predictions()
