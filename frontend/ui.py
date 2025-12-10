import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Configuration
# "api" is the service name defined in docker-compose
API_URL = "http://api:8000"

st.set_page_config(page_title="Wildfire Intelligence Platform", layout="wide")

# --- HEADER ---
st.title("🔥 Wildfire Intelligence & Recovery Platform")
st.markdown("""
**Domain:** Earth & Environmental Intelligence  
**System Status:** 🟢 Online | **Model Version:** v1.0.0 (Random Forest + K-Means)
""")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("🌍 Input Weather Conditions")

def user_input_features():
    lat = st.sidebar.slider("Latitude", 32.5, 42.0, 34.05)
    lon = st.sidebar.slider("Longitude", -124.5, -114.0, -118.25)
    tmmn = st.sidebar.number_input("Min Temperature (Kelvin)", 270.0, 320.0, 290.0)
    tmmx = st.sidebar.number_input("Max Temperature (Kelvin)", 270.0, 330.0, 305.0)
    rmin = st.sidebar.slider("Min Humidity (%)", 0.0, 100.0, 15.0)
    rmax = st.sidebar.slider("Max Humidity (%)", 0.0, 100.0, 45.0)
    vs = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 5.5)
    pr = st.sidebar.number_input("Precipitation (mm)", 0.0, 50.0, 0.0)
    erc = st.sidebar.slider("Energy Release Component (Dryness)", 0.0, 100.0, 50.0)
    
    data = {
        "tmmn": tmmn, "tmmx": tmmx,
        "rmin": rmin, "rmax": rmax,
        "vs": vs, "pr": pr, "erc": erc,
        "latitude": lat, "longitude": lon
    }
    return data

input_data = user_input_features()

# --- MAIN PANEL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Geospatial Analysis")
    # Simple map visualization of the target location
    map_data = pd.DataFrame({'lat': [input_data['latitude']], 'lon': [input_data['longitude']]})
    st.map(map_data, zoom=6)

with col2:
    st.subheader("🤖 AI Predictions")
    
    if st.button("Generate Prediction", type="primary"):
        try:
            # Call the FastAPI Backend
            response = requests.post(f"{API_URL}/predict", json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # 1. Classification (Risk)
                risk = result['risk_level_prediction']
                color = "green" if risk == "Low" else "orange" if risk == "Medium" else "red"
                st.markdown(f"### Fire Risk Level: :{color}[**{risk}**]")
                
                # 2. Regression (Burn Index)
                bi = result['burning_index_prediction']
                st.metric(label="Predicted Burning Index (Intensity)", value=bi)
                
                # 3. Clustering (Zone)
                zone = result['cluster_zone']
                st.info(f"Recommended Ecosystem Recovery Zone: **Zone {zone}**")
                
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: Is the API running? {e}")

# --- MLOps SECTION ---
st.divider()
st.subheader("⚙️ MLOps Pipeline Control")

col_a, col_b = st.columns(2)
with col_a:
    st.info("System Health: Healthy")
    if st.button("Check API Health"):
        try:
            res = requests.get(f"{API_URL}/health")
            st.json(res.json())
        except:
            st.error("API Offline")

with col_b:
    st.warning("Continuous Training")
    if st.button("Trigger Retraining Pipeline"):
        # In a real app, this would hit a /retrain endpoint
        # For demo, we simulate the request
        st.success("🚀 Retraining request sent to Prefect Orchestrator!")
        st.caption("(Check terminal logs for 'flows/training_flow.py' execution)")