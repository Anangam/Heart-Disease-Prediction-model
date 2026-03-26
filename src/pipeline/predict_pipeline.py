import sys
import pandas as pd
import os
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # 2. CRITICAL: Absolute paths for Vercel
            # Ye file 'src/pipeline/' mein hai, toh humein 2 level up jaana hai
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_path, 'artifacts', 'model.pkl')
            preprocessor_path = os.path.join(base_path, 'artifacts', 'preprocessor.pkl')

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            print(f"Error in Prediction: {str(e)}")
            raise e

class CustomData:
    def __init__(self, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age], "sex": [self.sex], "cp": [self.cp],
                "trestbps": [self.trestbps], "chol": [self.chol], "fbs": [self.fbs],
                "restecg": [self.restecg], "thalach": [self.thalach], "exang": [self.exang],
                "oldpeak": [self.oldpeak], "slope": [self.slope], "ca": [self.ca], "thal": [self.thal],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise e