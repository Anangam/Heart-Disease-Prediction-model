from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from flask import Flask, request, render_template
app = Flask(__name__)


## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # We are collecting the medical stats for the Heart Disease Model
        data = CustomData(
            age = int(request.form.get('age')),
            sex = int(request.form.get('sex')), # Usually 1 for Male, 0 for Female
            cp = int(request.form.get('cp')),   # Chest Pain type (0-3)
            trestbps = int(request.form.get('trestbps')), # Resting BP
            chol = int(request.form.get('chol')),         # Cholesterol
            fbs = int(request.form.get('fbs')),           # Fasting Blood Sugar
            restecg = int(request.form.get('restecg')),   # ECG results
            thalach = int(request.form.get('thalach')),   # Max Heart Rate
            exang = int(request.form.get('exang')),       # Exercise Induced Angina
            oldpeak = float(request.form.get('oldpeak')), # ST depression
            slope = int(request.form.get('slope')),       # Slope of ST segment
            ca = int(request.form.get('ca')),             # Number of major vessels
            thal = int(request.form.get('thal'))          # Thalassemia
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        # This will give you the 0 or 1 result
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Logic to make the output more readable for the user
        outcome = "Heart Disease Detected" if results[0] == 1 else "No Heart Disease Detected"

        return render_template('home.html', results=outcome)
    

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)