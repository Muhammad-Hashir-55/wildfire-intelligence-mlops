from pydantic import BaseModel, ConfigDict

# Input Data Model
class WeatherInput(BaseModel):
    tmmn: float
    tmmx: float
    rmin: float
    rmax: float
    vs: float
    pr: float
    erc: float
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