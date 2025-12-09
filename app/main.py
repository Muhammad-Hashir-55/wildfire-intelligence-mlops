import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from app.schemas import WeatherInput, PredictionOutput

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="Wildfire Intelligence API",
    description="Predicts Fire Intensity, Risk Levels, and Recovery Zones using Random Forest & K-Means.",
    version="1.0.0"
)

# Global dictionary to hold loaded models
models = {}

# Helper to get absolute path (Fixes "File not found" errors)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# [cite_start]2. Startup Event: Load Models Once (Efficient) [cite: 27]
@app.on_event("startup")
def load_models():
    """Loads ML models from the 'models/' directory on app startup."""
    try:
        # Construct safe paths
        reg_path = os.path.join(MODEL_DIR, "regression_model.pkl")
        clf_path = os.path.join(MODEL_DIR, "classification_model.pkl")
        clus_path = os.path.join(MODEL_DIR, "clustering_model.pkl")
        enc_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

        models['regression'] = joblib.load(reg_path)
        models['classification'] = joblib.load(clf_path)
        models['clustering'] = joblib.load(clus_path)
        models['encoder'] = joblib.load(enc_path)
        print(f"✅ Models loaded successfully from {MODEL_DIR}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load models. {e}")
        print(f"   (Checked path: {MODEL_DIR})")
        # In production, we might want to crash here, but for now we print error.

# 3. Health Check Endpoint
@app.get("/health")
def health_check():
    """Returns the health status of the API."""
    if not models:
        return {"status": "unhealthy", "detail": "Models not loaded"}
    return {"status": "healthy", "models_loaded": list(models.keys())}

# [cite_start]4. Prediction Endpoint [cite: 25, 26]
@app.post("/predict", response_model=PredictionOutput)
def predict(data: WeatherInput):
    """
    Accepts weather data -> Returns predictions for all 3 ML tasks.
    """
    if not models:
        # Try loading again if missing (Safety Net)
        load_models()
        if not models:
            raise HTTPException(status_code=503, detail="Models are not loaded.")

    try:
        # Convert Pydantic object to DataFrame (required by Scikit-Learn)
        # UPDATE: .dict() is deprecated in Pydantic V2, use .model_dump()
        input_data = data.model_dump()
        input_df = pd.DataFrame([input_data])

        # --- Task 1: Regression (Predict Burning Index) ---
        # Features: tmmn, tmmx, rmin, rmax, vs, pr, erc
        reg_features = ['tmmn', 'tmmx', 'rmin', 'rmax', 'vs', 'pr', 'erc']
        bi_pred = models['regression'].predict(input_df[reg_features])[0]

        # --- Task 2: Classification (Predict Risk Level) ---
        # Predict class index first
        risk_index = models['classification'].predict(input_df[reg_features])[0]
        # Decode index to string (e.g., 0 -> "Low")
        risk_label = models['encoder'].inverse_transform([risk_index])[0]

        # --- Task 3: Clustering (Assign Recovery Zone) ---
        # Uses: latitude, longitude, and the PREDICTED Burning Index
        cluster_input = input_df[['latitude', 'longitude']].copy()
        cluster_input['bi'] = bi_pred 
        
        zone = models['clustering'].predict(cluster_input)[0]

        return {
            "burning_index_prediction": round(float(bi_pred), 2),
            "risk_level_prediction": risk_label,
            "cluster_zone": int(zone)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# 5. Root Endpoint
@app.get("/")
def home():
    return {"message": "Wildfire Intelligence System is Online. Go to /docs for testing."}