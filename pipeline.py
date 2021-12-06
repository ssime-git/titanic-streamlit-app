# pipeline
from sklearn.pipeline import Pipeline

# scaler
from sklearn.preprocessing import StandardScaler

# Logistic regression model
from sklearn.linear_model import LogisticRegression

# for the preprocessors
from sklearn.base import BaseEstimator, TransformerMixin

# for imputation
from feature_engine.imputation import (AddMissingIndicator, # for numerical var
                                        MeanMedianImputer, # for numerical values
                                        CategoricalImputer, # for rare and missing in cat var
                                    )

# for encoding categorical variables
from feature_engine.encoding import (RareLabelEncoder, OneHotEncoder)

# Import config
from config import config

# Import custom classes
from preprocessing.preprocessors import ExtractLetterTransformer

# set up the pipeline
titanic_pipe = Pipeline(
    [
    # ===== IMPUTATION =====
    # impute categorical variables with string 'missing'
    ('categorical_imputation', CategoricalImputer(imputation_method='missing', variables=config.CATEGORICAL_VARIABLES)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=config.NUMERICAL_VARIABLES)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(imputation_method='median', variables=config.NUMERICAL_VARIABLES)),


    # Extract first letter from cabin
    ('extract_letter', ExtractLetterTransformer(variables=config.CABIN)),

    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare' (=name by default)
    ('rare_label_encoder', RareLabelEncoder(tol=config.RARE_LABEL_FRAC, 
                            n_categories=1, variables=config.CATEGORICAL_VARIABLES)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(drop_last=True, variables=config.CATEGORICAL_VARIABLES)),

    # scale using standardization
    ('scaler', StandardScaler()),

    # logistic regression (use C=0.0005 and random_state=0)
    ('Logit', LogisticRegression(C=config.C, random_state=config.RANDOM_STATE)),
]
)