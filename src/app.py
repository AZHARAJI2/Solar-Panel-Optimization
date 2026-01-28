from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import datetime

app = Flask(__name__)

# Paths
BASE_DIR = r"d:\Solar-Panel-Optimization"
MODEL_FILE = os.path.join(BASE_DIR, "solar_model.pkl")

# Load Model
print("Loading model...")
try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        irradiation = float(data.get('irradiation'))
        temperature = float(data.get('temperature'))
        
        # We also need HOUR and MONTH for the model as they were features during training
        # Ideally, the user selects a time, or we use current time.
        # Let's handle both: If 'datetime' provided, use it. Else use current.
        date_str = data.get('datetime')
        if date_str:
            dt = pd.to_datetime(date_str)
        else:
            dt = datetime.datetime.now()
            
        hour = dt.hour
        month = dt.month
        
        # Create input dataframe matching training features
        features = pd.DataFrame([{
            'IRRADIATION': irradiation,
            'MODULE_TEMPERATURE': temperature,
            'HOUR': hour,
            'MONTH': month
        }])
        
        # Predict "Ideal" Power
        predicted_power = model.predict(features)[0]
        
        # Dust Detection Logic
        # If the user provides "Actual Power", we can detect dust.
        actual_power = data.get('actual_power')
        recommendation = "N/A"
        status = "normal"
        
        if actual_power is not None and actual_power != "":
            actual_power = float(actual_power)
            # Calculate loss
            loss = predicted_power - actual_power
            loss_percent = (loss / predicted_power) * 100 if predicted_power > 1 else 0
            
            # Thresholds (Example: >20% loss when sunny)
            if irradiation > 0.1 and loss_percent > 20:
                recommendation = "يوجد غبار في الألواح او خلل في الطاقة"
                status = "alert"
            elif loss_percent > 10:
                recommendation = "يرجى مراقبة الأداء"
                status = "warning"
            else:
                recommendation = "النظام يعمل بكفاءة"
                status = "ok"
                
        response = {
            'predicted_power': round(float(predicted_power), 2),
            'recommendation': recommendation,
            'status': status,
            'details': {
                'hour': hour,
                'month': month,
                'input_irradiation': irradiation,
                'input_temp': temperature
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
