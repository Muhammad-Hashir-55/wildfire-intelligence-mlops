import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app, load_models

# 1. FORCE LOAD MODELS MANUALLY
# This ensures they are ready before tests run
print("⚙️  Forcing Model Load for Testing...")
load_models()

client = TestClient(app)

def test_health_check():
    """Test if the API is alive and models are loaded."""
    print("\n🔍 Testing Health Endpoint...")
    response = client.get("/health")
    
    # Debugging print
    print(f"   Response: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    
    # Crucial check: Status MUST be healthy now
    assert data["status"] == "healthy"
    print("✅ Health Check Passed!")

def test_prediction_endpoint():
    """Test the End-to-End Prediction pipeline with valid data."""
    print("\n🔍 Testing Prediction Endpoint...")
    
    payload = {
        "tmmn": 290.0, "tmmx": 305.0, "rmin": 15.0, "rmax": 45.0,
        "vs": 6.5, "pr": 0.0, "erc": 50.0,
        "latitude": 34.0, "longitude": -118.0
    }
    
    response = client.post("/predict", json=payload)
    
    # If this fails, print the error detail
    if response.status_code != 200:
        print(f"❌ API Error: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    
    # Verify Content
    assert "burning_index_prediction" in data
    assert "risk_level_prediction" in data
    assert "cluster_zone" in data
    
    print("✅ Prediction Logic Passed!")
    print(f"   🔥 Predicted BI: {data['burning_index_prediction']}")
    print(f"   ⚠️ Risk Level:   {data['risk_level_prediction']}")

if __name__ == "__main__":
    try:
        test_health_check()
        test_prediction_endpoint()
        print("\n🎉 ALL API TESTS PASSED SUCCESSFULLY.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: Assertion Error")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")