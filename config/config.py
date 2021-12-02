# this file contains all the constants used in our final research notebook
# list of variables to be used in the pipeline's transformers

# File path: IMPORTANT: These datapaths need to be changed in the final package
DATASETS_DIR = "./datasets/"
SAVED_MODEL_PATH = "./trained_models/"
TRAIN_FILE = DATASETS_DIR + 'titanic.csv'
#TEST_FILE = 'test.csv'
MODEL_NAME = 'titanic-prediction-model'
MODEL_VERSION = '0.0.0'

# Target feature
TARGET = 'survived'

# Features to keep
KEEP_FEATURES = ['pclass', 'sex', 'age', 
                'sibsp', 'parch', 'fare', 'cabin','embarked', 'title']

# Features to drop (if any)
#DROP_FEATURES = ['']

# Numerical features
NUMERICAL_VARIABLES = ['age', 'fare']

# Categorical features
CATEGORICAL_VARIABLES = ['sex', 'cabin', 'embarked', 'title']

# Feature to encode (if any)
#FEATURES_TO_ENCODE =

# Temporal feature (if any)
#TEMPORAL_FEATURES =
#TEMPORAL_COMPARISON =

# Features to transform (if any)
#LOG_FEATURES = [''] #Features for Log Transform
CABIN = ['cabin'] # To extract the first letter

# Rare categories fraction
RARE_LABEL_FRAC = .05

# Test Split fraction
SPLIT_FRAC = .2

# Random state
RANDOM_STATE = 0

# C PARAM
C = .0005