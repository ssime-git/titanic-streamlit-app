import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#Local Files
from config import config

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y=None): # very important should have X and y otherwise you will have an error
        return self        

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].str[0]
            return X