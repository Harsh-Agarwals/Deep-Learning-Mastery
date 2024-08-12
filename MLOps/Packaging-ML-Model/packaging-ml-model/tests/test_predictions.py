import pytest 

from pathlib import Path
import os
import sys
# This makes sure our package is found
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import predict_values

import warnings
warnings.filterwarnings('ignore')

# pytest search all the files inside the current directory and it's subdirectories that is of the format test_*.py or *_test.py
# when we run pytest, it will first search for the file that starts/ends with test. Then inside the script, it search for all
# the functions that starts with test.
# Before these test functions are run, we have to make sure that these script_testing functions are run.
# And instead of running same function multiple times, we'll attach fixture function to my test which will run this function and return the data in before each test
# fixtures => used when we have a common function(here script_testing) that will run before the execution of each of the testing cases.

# adding decorator
@pytest.fixture # will keep my output ready which will be used further in our functions
def script_testing():
    data = load_dataset(config.TEST_FILE)
    single_row = data.iloc[:1, :]
    result = predict_values(data)
    return result

# PYTEST FUNCTIONS
# Chekcing if the result is not null
def test_single_pred_not_null(script_testing):
    assert script_testing['Predictions'][0] is not None

# Checking if the result is 'Y'
def test_single_pred_is_Y(script_testing):
    assert script_testing['Predictions'][0]=='Y'

# Checking the data type is string
def test_single_pred_is_string(script_testing):
    assert isinstance(script_testing['Predictions'][0], str)
