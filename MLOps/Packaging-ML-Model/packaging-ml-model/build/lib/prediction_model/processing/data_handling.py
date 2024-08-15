import os
import sys
import joblib
import pandas as pd
from pathlib import Path

from prediction_model.config import config

# Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    data = pd.read_csv(filepath)
    return data

# Serialization
def save_pipeline(pipeline_to_save):
    model_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, model_path)
    print(f"Model saved to the {model_path}")

# Deserialization
def load_pipeline(pipeline_to_load):
    model_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_NAME)
    model = joblib.load(model_path)
    print(f"{config.MODEL_NAME} loaded from {model_path}")
    return model