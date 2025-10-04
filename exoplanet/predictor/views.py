from django.shortcuts import render
import pandas as pd
import joblib
import os
from django.conf import settings

# Build paths to the model files
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'random_forest_model.joblib')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'scaler.joblib')

# Load the model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_view(request):
    context = {}
    if request.method == 'POST':
        # 1. Get user input from the form
        features = {
            'koi_fpflag_nt': float(request.POST.get('koi_fpflag_nt')),
            'koi_fpflag_ss': float(request.POST.get('koi_fpflag_ss')),
            'koi_fpflag_co': float(request.POST.get('koi_fpflag_co')),
            'koi_fpflag_ec': float(request.POST.get('koi_fpflag_ec')),
            'koi_period': float(request.POST.get('koi_period')),
            'koi_duration': float(request.POST.get('koi_duration')),
            'koi_depth': float(request.POST.get('koi_depth')),
            'koi_prad': float(request.POST.get('koi_prad')),
            'koi_teq': float(request.POST.get('koi_teq')),
            'koi_insol': float(request.POST.get('koi_insol')),
            'koi_impact': float(request.POST.get('koi_impact')),
        }

        # 2. Convert to DataFrame and scale
        input_df = pd.DataFrame([features])
        input_scaled = scaler.transform(input_df)

        # 3. Make prediction
        prediction_val = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # 4. Prepare result for frontend
        if prediction_val == 1:
            prediction_text = "CONFIRMED Exoplanet"
            confidence = prediction_proba[1] * 100
        else:
            prediction_text = "FALSE POSITIVE"
            confidence = prediction_proba[0] * 100

        context = {
            'prediction': prediction_text,
            'confidence': f"{confidence:.2f}%",
            'submitted_data': features, # Send back the user's data
        }

    return render(request, 'predictor/index.html', context)