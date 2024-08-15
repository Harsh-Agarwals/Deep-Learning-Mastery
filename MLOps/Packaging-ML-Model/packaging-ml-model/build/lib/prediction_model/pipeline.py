import numpy as np
from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

classification_pipeline = Pipeline([
    # We have created our own data transformation classes
    ('Mean Imputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
    ('Mode Imputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
    ('Domain Processing', pp.DomainProcessing(variable_to_add=config.FEATURES_TO_ADD, variable_to_modify=config.FEATURES_TO_MODIFY)),
    ('Drop Columns', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
    ('Label Encoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
    ('Log Transform', pp.LogTransform(variables=config.LOG_FEATURES)),
    ('MinMax Scaling', MinMaxScaler()),
    ('Logistic Classifier', LogisticRegression(random_state=42))
])