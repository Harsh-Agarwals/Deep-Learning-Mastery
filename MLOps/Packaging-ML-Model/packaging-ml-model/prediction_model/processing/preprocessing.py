from sklearn.base import BaseEstimator,TransformerMixin
from prediction_model.config import config
import numpy as np

# fit and transform methods are defined here because these classes will be used in sklearn pipeline
# which expects classes to have fit and transform methods

# Mean imputation
class MeanImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self
    
    def transform(self,X):
        for col in self.variables:
            X[col].fillna(self.mean_dict[col],inplace=True)
        return X
    
# Mode Imputation
class ModeImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mode_dict = {}
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()
        return self
    
    def transform(self,X):
        for col in self.variables:
            X[col].fillna(self.mode_dict[col],inplace=True)
        return X
    
# Drop Columns
class DropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X.drop(columns=[self.variables_to_drop], inplace=True)
        return X
    
# Domain Processing
class DomainProcessing(BaseEstimator,TransformerMixin):
    def __init__(self,variable_to_modify = None, variable_to_add = None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for feature in self.variable_to_modify:
            X[feature] = X[feature] + X[self.variable_to_add]
        return X
    
# Custom Label Encoder
class CustomLabelEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables=variables
    
    def fit(self, X,y):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X
    
# Log Transform
class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            X[var] = np.log(X[var])
        return X