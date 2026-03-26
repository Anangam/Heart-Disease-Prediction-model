import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score # Changed from r2_score for Classification
from src.exception import CustomException

def save_object(file_path, obj):
    """Saves a python object (like a model or preprocessor) to a .pkl file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads a .pkl file back into a python object."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Trains multiple models and returns a dictionary of their accuracy scores."""
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            
            # Training the model
            model.fit(X_train, y_train)

            # Making predictions
            y_test_pred = model.predict(X_test)

            # Using accuracy_score because heart disease is 0 or 1
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
        
    except Exception as e:
        raise CustomException(e, sys)
    