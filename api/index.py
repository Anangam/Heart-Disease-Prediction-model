import os
from flask import Flask, render_template, request
import pandas as pd
import joblib

# '../templates' tells Flask to go out of 'api' and into 'templates'
app = Flask(__name__, template_folder='../templates')

# Path to your model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../artifacts/model.pkl')

# Load your model (adjust the filename if yours is different)
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Example: capture data from form
    # features = [float(x) for x in request.form.values()]
    # prediction = model.predict([features])
    return render_template('index.html', prediction_text="Model result goes here")

# Do NOT use app.run() here; Vercel handles the execution