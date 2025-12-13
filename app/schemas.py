from pydantic import BaseModel, ConfigDict
from typing import Dict

# Input Data Model
class WeatherInput(BaseModel):
    tmmn: float  # temp min (Kelvin)
    tmmx: float  # temp max (Kelvin)
    rmin: float  # humidity min (%)
    rmax: float  # humidity max (%)
    vs: float    # wind speed (m/s)
    pr: float    # precipitation (mm)
    erc: float   # energy release component
    latitude: float
    longitude: float

    # Pydantic V2 Config
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tmmn": 290.5,
                "tmmx": 305.2,
                "rmin": 12.5,
                "rmax": 45.0,
                "vs": 5.4,
                "pr": 0.0,
                "erc": 48.0,
                "latitude": 34.05,
                "longitude": -118.25
            }
        }
    )

# Output Data Model
class PredictionOutput(BaseModel):
    burning_index_prediction: float
    risk_level_prediction: str
    cluster_zone: int
    pca_x: float
    pca_y: float
    seasonal_trend: Dict[int, float]  # Maps Month (1-12) to Avg Intensity
    
    # Pydantic V2 Config with example
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "burning_index_prediction": 65.8,
                "risk_level_prediction": "Medium",
                "cluster_zone": 2,
                "pca_x": -1.25,
                "pca_y": 0.83,
                "seasonal_trend": {
                    1: 42.3, 2: 44.1, 3: 47.8, 4: 52.4, 
                    5: 58.9, 6: 67.2, 7: 75.6, 8: 78.3, 
                    9: 71.8, 10: 62.4, 11: 51.2, 12: 45.7
                }
            }
        }
    )

# Optional: Model for PCA Projection endpoint
class PCAProjectionOutput(BaseModel):
    pca_x: float
    pca_y: float
    explained_variance_ratio: list[float]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pca_x": -1.25,
                "pca_y": 0.83,
                "explained_variance_ratio": [0.65, 0.25]
            }
        }
    )

# Optional: Model for Seasonal Trend endpoint
class SeasonalTrendOutput(BaseModel):
    description: str
    trend: Dict[int, float]
    units: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Average Burning Index by Month",
                "trend": {
                    1: 42.3, 2: 44.1, 3: 47.8, 4: 52.4, 
                    5: 58.9, 6: 67.2, 7: 75.6, 8: 78.3, 
                    9: 71.8, 10: 62.4, 11: 51.2, 12: 45.7
                },
                "units": "Burning Index (BI)"
            }
        }
    )

# Optional: Health check response model
class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str] = []
    missing: list[str] = []
    detail: str = ""
    version: str = ""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "models_loaded": ["regression", "classification", "clustering", "encoder", "pca", "seasonality"],
                "missing": [],
                "detail": "",
                "version": "1.1.0"
            }
        }
    )

# Optional: Model info response model
class ModelInfo(BaseModel):
    type: str
    details: dict
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "regression": {
                    "type": "RandomForest",
                    "n_estimators": 50,
                    "n_features": 7
                }
            }
        }
    )