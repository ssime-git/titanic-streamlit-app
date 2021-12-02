from config import config
import pandas as pd
import joblib
import os

def load_dataset(file_name):
    """Function to load the dataframe"""
    _data = pd.read_csv(file_name)
    return _data

def save_pipeline(pipeline_to_save):
    """Function to save the model after training"""

    # set the save file name
    save_file_name = f'titanic_classification_v_{config.MODEL_VERSION}.pkl'

    # set the saved file path
    save_path = config.SAVED_MODEL_PATH + save_file_name

    # save the model
    joblib.dump(pipeline_to_save, save_path)
    #print("Saved Pipeline : ", save_file_name)

def load_pipeline(pipeline_to_load):
    """Function to load the model for a prediction (after training)"""
    # Set the saved model directory
    #model_dir = config.SAVED_MODEL_PATH

    # Set the model path
    #save_path = model_dir + pipeline_to_load
    #print('save_path from data_mana: ', save_path)

    if os.path.exists(pipeline_to_load):
        trained_model = joblib.load(pipeline_to_load)
        return trained_model
    else:
        print("No model saved with that name !")