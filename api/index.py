import os
import sys
from flask import Flask, render_template, request

# 1. CRITICAL: Root folder ko path mein add karo taaki 'src' import ho sake
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Ab imports kaam karenge
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def index():
    # Aapka main landing page
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Form se data uthao (13 features for Heart Disease)
        data = CustomData(
            age=int(request.form.get('age')),
            sex=int(request.form.get('sex')),
            cp=int(request.form.get('cp')),
            trestbps=int(request.form.get('trestbps')),
            chol=int(request.form.get('chol')),
            fbs=int(request.form.get('fbs')),
            restecg=int(request.form.get('restecg')),
            thalach=int(request.form.get('thalach')),
            exang=int(request.form.get('exang')),
            oldpeak=float(request.form.get('oldpeak')),
            slope=int(request.form.get('slope')),
            ca=int(request.form.get('ca')),
            thal=int(request.form.get('thal'))
        )
        
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Result ko home.html pe dikhao
        return render_template('home.html', results=results[0])

# Vercel doesn't need app.run()