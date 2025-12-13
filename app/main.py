import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from app.schemas import WeatherInput, PredictionOutput

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="Wildfire Intelligence API",
    description="Predicts Fire Intensity, Risk Levels, Recovery Zones, PCA visualization, and Seasonal Trends.",
    version="1.1.0"
)

# Global dictionary to hold loaded models
models = {}

# Helper to get absolute path (Fixes "File not found" errors)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 2. Startup Event: Load Models Once (Efficient)
@app.on_event("startup")
def load_models():
    """Loads ML models from the 'models/' directory on app startup."""
    try:
        # Construct safe paths
        reg_path = os.path.join(MODEL_DIR, "regression_model.pkl")
        clf_path = os.path.join(MODEL_DIR, "classification_model.pkl")
        clus_path = os.path.join(MODEL_DIR, "clustering_model.pkl")
        enc_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        pca_path = os.path.join(MODEL_DIR, "pca_model.pkl")
        seasonal_path = os.path.join(MODEL_DIR, "seasonal_model.pkl")

        models['regression'] = joblib.load(reg_path)
        models['classification'] = joblib.load(clf_path)
        models['clustering'] = joblib.load(clus_path)
        models['encoder'] = joblib.load(enc_path)
        models['pca'] = joblib.load(pca_path)
        models['seasonality'] = joblib.load(seasonal_path)
        
        print(f"✅ All Models (Regression, Classification, Clustering, PCA, Seasonality) Loaded from {MODEL_DIR}")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Model file not found. {e}")
        print(f"   (Checked path: {MODEL_DIR})")
        print("   Make sure you have run the training script first.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load models. {e}")
        print(f"   (Checked path: {MODEL_DIR})")

# 3. Health Check Endpoint
@app.get("/health")
def health_check():
    """Returns the health status of the API."""
    if not models:
        return {"status": "unhealthy", "detail": "Models not loaded"}
    
    loaded_models = list(models.keys())
    expected_models = ['regression', 'classification', 'clustering', 'encoder', 'pca', 'seasonality']
    missing_models = [m for m in expected_models if m not in loaded_models]
    
    if missing_models:
        return {
            "status": "partial", 
            "loaded": loaded_models,
            "missing": missing_models,
            "detail": f"Missing {len(missing_models)} model(s)"
        }
    
    return {
        "status": "healthy", 
        "models_loaded": loaded_models,
        "version": "1.1.0"
    }

# 4. Prediction Endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(data: WeatherInput):
    """
    Accepts weather data -> Returns predictions for all ML tasks.
    """
    if not models:
        # Try loading again if missing (Safety Net)
        load_models()
        if not models:
            raise HTTPException(status_code=503, detail="Models are not loaded.")

    try:
        # Convert Pydantic object to DataFrame (required by Scikit-Learn)
        input_data = data.model_dump()
        input_df = pd.DataFrame([input_data])

        # Define common features for regression, classification, and PCA
        reg_features = ['tmmn', 'tmmx', 'rmin', 'rmax', 'vs', 'pr', 'erc']
        
        # Validate required features are present
        missing_features = [f for f in reg_features if f not in input_df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {missing_features}"
            )

        # --- Task 1: Regression (Predict Burning Index) ---
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

        # --- Task 4: DIMENSIONALITY REDUCTION (PCA) ---
        # Project this user's input onto the 2D PCA plane
        pca_coords = models['pca'].transform(input_df[reg_features])[0]
        
        # --- Task 5: TIME SERIES SEASONALITY ---
        # Get the 12-month seasonal trend
        monthly_trend = models['seasonality']

        return {
            "burning_index_prediction": round(float(bi_pred), 2),
            "risk_level_prediction": risk_label,
            "cluster_zone": int(zone),
            "pca_x": float(pca_coords[0]),
            "pca_y": float(pca_coords[1]),
            "seasonal_trend": monthly_trend
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# 5. Additional Endpoints for Enhanced Functionality

@app.get("/seasonal_trend")
def get_seasonal_trend():
    """Returns the full monthly seasonal trend."""
    if 'seasonality' not in models:
        raise HTTPException(status_code=503, detail="Seasonality model not loaded")
    
    return {
        "description": "Average Burning Index by Month",
        "trend": models['seasonality'],
        "units": "Burning Index (BI)"
    }

@app.post("/pca_projection")
def pca_projection(data: WeatherInput):
    """Returns PCA projection coordinates for visualization."""
    if 'pca' not in models:
        raise HTTPException(status_code=503, detail="PCA model not loaded")
    
    try:
        input_data = data.model_dump()
        input_df = pd.DataFrame([input_data])
        reg_features = ['tmmn', 'tmmx', 'rmin', 'rmax', 'vs', 'pr', 'erc']
        
        pca_coords = models['pca'].transform(input_df[reg_features])[0]
        
        return {
            "pca_x": float(pca_coords[0]),
            "pca_y": float(pca_coords[1]),
            "explained_variance_ratio": models['pca'].explained_variance_ratio_.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PCA Projection Error: {str(e)}")

@app.get("/model_info")
def model_info():
    """Returns information about the loaded models."""
    if not models:
        return {"error": "No models loaded"}
    
    info = {}
    for name, model in models.items():
        if name == 'seasonality':
            info[name] = {
                "type": "dictionary",
                "keys": list(model.keys()) if isinstance(model, dict) else str(type(model))
            }
        elif name == 'encoder':
            info[name] = {
                "type": "LabelEncoder",
                "classes": model.classes_.tolist() if hasattr(model, 'classes_') else "unknown"
            }
        elif name == 'pca':
            info[name] = {
                "type": "PCA",
                "n_components": model.n_components_,
                "explained_variance": model.explained_variance_ratio_.tolist()
            }
        elif hasattr(model, 'n_estimators'):  # Random Forest
            info[name] = {
                "type": "RandomForest",
                "n_estimators": model.n_estimators,
                "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown"
            }
        elif hasattr(model, 'n_clusters'):  # KMeans
            info[name] = {
                "type": "KMeans",
                "n_clusters": model.n_clusters,
                "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown"
            }
        else:
            info[name] = {"type": str(type(model))}
    
    return info

# 6. Root Endpoint
@app.get("/")
def home():
    return {
        "message": "Wildfire Intelligence System v1.1.0 is Online",
        "endpoints": {
            "predict": "POST /predict - Get all predictions",
            "health": "GET /health - Check API health",
            "seasonal": "GET /seasonal_trend - Get monthly trends",
            "pca": "POST /pca_projection - Get PCA coordinates",
            "model_info": "GET /model_info - Get model details",
            "docs": "GET /docs - Interactive API documentation"
        }
    }

