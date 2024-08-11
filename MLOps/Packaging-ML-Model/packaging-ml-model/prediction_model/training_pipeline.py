import numpy as np
import pandas as pd

from pathlib import Path
import os
import sys
# This makes sure our package is found
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline
from prediction_model.processing import preprocessing as pp
import prediction_model.pipeline as pipeline

import warnings
warnings.filterwarnings('ignore')

def perform_training():
    data = load_dataset(config.TRAIN_FILE)
    train_y = data[config.TARGET].map({'NO': 0, 'YES': 1})
    pipeline.classification_pipeline.fit(data[config.FEATURES], train_y)
    save_pipeline(pipeline.classification_pipeline)

if __name__=="__main__":
    perform_training()