import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime


# Configuration
# "api" is the service name defined in docker-compose
API_URL = "http://api:8000"

# import os

# # Configuration
# # "api" is the service name defined in docker-compose
# # If running on Docker (Hugging Face), use localhost. 
# # If running locally with Docker Compose, use 'backend'.
# API_URL = os.getenv("API_URL", "http://api:8000")


st.set_page_config(
    page_title="Wildfire Intelligence Platform", 
    layout="wide",
    page_icon="🔥"
)

# --- HEADER ---
st.title("🔥 Wildfire Intelligence & Recovery Platform")
st.markdown("""
**Domain:** Earth & Environmental Intelligence  
**System Status:** 🟢 Online | **Model Version:** v1.2.0 (RF + K-Means + PCA + Gemini AI)
""")

# --- SIDEBAR: INPUTS & LIVE DATA ---
st.sidebar.header("🌍 Input Weather Conditions")

def user_input_features():
    # 1. Location Selection
    st.sidebar.subheader("📍 Location")
    lat = st.sidebar.number_input("Latitude", 32.0, 42.0, 34.05, 0.01)
    lon = st.sidebar.number_input("Longitude", -125.0, -114.0, -118.25, 0.01)
    
    # 2. Live Weather Button logic
    # We use session state to ensure values stick after button press
    if 'tmmn' not in st.session_state:
        # Initialize defaults
        st.session_state.update({
            'tmmn': 290.5, 'tmmx': 305.2, 'rmin': 12.5, 'rmax': 45.0,
            'vs': 5.4, 'pr': 0.0, 'erc': 48.0
        })

    if st.sidebar.button("📡 Fetch Live Weather (Free)"):
        try:
            with st.spinner("Connecting to Open-Meteo Satellite..."):
                # Open-Meteo API Call (No Key Required)
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
                w_data = requests.get(url).json()
                
                curr = w_data['current']
                daily = w_data['daily']
                
                # Map API data to our model features & update Session State
                st.session_state['tmmn'] = daily['temperature_2m_min'][0] + 273.15 # C to Kelvin
                st.session_state['tmmx'] = daily['temperature_2m_max'][0] + 273.15
                st.session_state['rmin'] = float(curr['relative_humidity_2m']) - 10 # Estimate min
                st.session_state['rmax'] = float(curr['relative_humidity_2m']) + 10 # Estimate max
                st.session_state['vs'] = float(curr['wind_speed_10m']) / 3.6 # km/h to m/s
                st.session_state['pr'] = float(curr['rain'])
                # ERC is hard to get from simple weather APIs, keep default or randomize slightly
                
            st.sidebar.success("✅ Live Data Loaded!")
        except Exception as e:
            st.sidebar.error(f"Weather Fetch Failed: {e}")

    # 3. The Sliders (Bound to Session State)
    st.sidebar.subheader("️Condition Parameters")
    tmmn = st.sidebar.number_input("Min Temperature (K)", 270.0, 320.0, value=st.session_state['tmmn'])
    tmmx = st.sidebar.number_input("Max Temperature (K)", 270.0, 330.0, value=st.session_state['tmmx'])
    rmin = st.sidebar.slider("Min Humidity (%)", 0.0, 100.0, value=st.session_state['rmin'])
    rmax = st.sidebar.slider("Max Humidity (%)", 0.0, 100.0, value=st.session_state['rmax'])
    vs = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, value=st.session_state['vs'])
    pr = st.sidebar.number_input("Precipitation (mm)", 0.0, 50.0, value=st.session_state['pr'])
    erc = st.sidebar.slider("Energy Release Component (Dryness)", 0.0, 100.0, value=st.session_state['erc'])
    
    data = {
        "tmmn": tmmn, "tmmx": tmmx,
        "rmin": rmin, "rmax": rmax,
        "vs": vs, "pr": pr, "erc": erc,
        "latitude": lat, "longitude": lon
    }
    return data

input_data = user_input_features()

# Display current inputs in sidebar
st.sidebar.divider()
st.sidebar.subheader("📊 Current Inputs")
st.sidebar.json(input_data)

# --- MAIN PANEL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Geospatial Analysis")
    # Simple map visualization of the target location
    map_data = pd.DataFrame({
        'lat': [input_data['latitude']], 
        'lon': [input_data['longitude']],
        'size': [20]
    })
    st.map(map_data, zoom=6)
    
    # Add location info
    st.caption(f"📍 **Location**: {input_data['latitude']:.2f}°N, {input_data['longitude']:.2f}°W")

with col2:
    st.subheader("🤖 AI Predictions")
    
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        try:
            with st.spinner("Calling AI models..."):
                # Call the FastAPI Backend
                response = requests.post(f"{API_URL}/predict", json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Store result in session state for later use
                st.session_state.prediction_result = result
                st.session_state.last_input = input_data # Save input for Gemini
                
                # 1. Classification (Risk)
                risk = result['risk_level_prediction']
                color = "green" if risk == "Low" else "orange" if risk == "Medium" else "red"
                st.markdown(f"### Fire Risk Level: :{color}[**{risk}**]")
                
                # Risk explanation
                risk_explanations = {
                    "Low": "✅ Normal conditions - minimal fire danger",
                    "Medium": "⚠️ Elevated conditions - monitor for changes",
                    "High": "🔥 Critical conditions - high fire danger"
                }
                st.caption(risk_explanations.get(risk, "Risk Status"))
                
                # 2. Regression (Burn Index)
                bi = result['burning_index_prediction']
                st.metric(
                    label="Predicted Burning Index (Intensity)", 
                    value=f"{bi:.1f}",
                    help="Higher values indicate greater fire intensity potential"
                )
                
                # Intensity gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = bi,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fire Intensity"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 40], 'color': "green"},
                            {'range': [40, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': bi
                        }
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # 3. Clustering (Zone)
                zone = result['cluster_zone']
                zone_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
                zone_color = zone_colors[zone % len(zone_colors)]
                
                st.markdown(f"""
                <div style="background-color:{zone_color}20; padding:15px; border-radius:10px; border-left:5px solid {zone_color}">
                    <h4 style="margin:0; color:{zone_color}">🌱 Recovery Zone {zone}</h4>
                    <p style="margin:5px 0 0 0; font-size:14px;">
                    Recommended for ecosystem monitoring and post-fire recovery planning
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection Error: API service is not running. Make sure to start the FastAPI server first.")
            st.info("Run: `docker-compose up api` or `uvicorn app.main:app --reload`")
        except Exception as e:
            st.error(f"⚠️ Unexpected Error: {str(e)}")
    else:
        # Show placeholder before prediction
        if 'prediction_result' not in st.session_state:
            st.info("👆 Click 'Generate Prediction' to run the AI models")

# If we have results, show additional visualizations AND Gemini AI
if 'prediction_result' in st.session_state:
    result = st.session_state.prediction_result
    
    # --- NEW: GEMINI AI SECTION ---
    st.divider()
    st.subheader("🧠 Gemini Commander AI")
    st.caption("Generative AI Tactical Response Strategy")
    
    col_ai1, col_ai2 = st.columns([1, 3])
    
    with col_ai1:
        if st.button("📝 Generate Tactical Plan", use_container_width=True):
            with st.spinner("Analyzing tactical options with Gemini 2.5 Flash..."):
                try:
                    # Prepare payload for LLM
                    # We use the inputs stored in session state + prediction results
                    llm_payload = {
                        "risk_level": result['risk_level_prediction'],
                        "bi": result['burning_index_prediction'],
                        "location": f"{input_data['latitude']}, {input_data['longitude']}",
                        "conditions": st.session_state.get('last_input', input_data)
                    }
                    
                    # Call backend LLM endpoint
                    llm_res = requests.post(f"{API_URL}/generate_report", json=llm_payload)
                    
                    if llm_res.status_code == 200:
                        report = llm_res.json()["report"]
                        st.session_state.ai_report = report # Save it
                    else:
                        st.error("AI Service Unavailable")
                except Exception as e:
                    st.error(f"AI Error: {e}")

    with col_ai2:
        if 'ai_report' in st.session_state:
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; border-left: 5px solid #6366f1; color:black;">
                {st.session_state.ai_report}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click the button to generate an incident response plan.")

    # --- NEW VISUAL 1: TIME SERIES FORECAST ---
    st.divider()
    col_ts1, col_ts2 = st.columns([3, 1])
    
    with col_ts1:
        st.subheader("📅 Annual Fire Forecast")
        st.caption("Projected Fire Intensity based on Seasonal Trends")
        
        # === FIX START: Handle JSON String Keys ===
        trend_data_raw = result['seasonal_trend']
        # Convert string keys "1" back to int 1
        trend_data = {int(k): v for k, v in trend_data_raw.items()}
        
        months = sorted(list(trend_data.keys()))
        intensities = [trend_data[m] for m in months]
        # === FIX END ===
        
        # Create DataFrame for visualization
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        trend_df = pd.DataFrame({
            'Month': [month_names[m-1] for m in months], # Now 'm' is definitely an int
            'Month_Num': months,
            'Predicted Intensity': intensities,
            'Current Month': [intensities[0] if i == 0 else None for i in range(len(months))]
        })
        
        # Create line chart
        fig_trend = px.line(
            trend_df, 
            x='Month', 
            y='Predicted Intensity',
            title="Monthly Fire Intensity Pattern",
            markers=True,
            line_shape='spline'
        )
        
        # Add shaded area for high-risk months
        high_risk_months = [6, 7, 8, 9]  # Jun-Sep
        for month_num in high_risk_months:
            if month_num in months:
                month_idx = months.index(month_num)
                fig_trend.add_vrect(
                    x0=month_idx-0.5, x1=month_idx+0.5,
                    fillcolor="red", opacity=0.1,
                    line_width=0,
                    annotation_text="High Risk" if month_num == 7 else "",
                    annotation_position="top left"
                )
        
        fig_trend.update_traces(
            line=dict(color='orange', width=3),
            marker=dict(size=8, color='red')
        )
        fig_trend.update_layout(
            xaxis_title="Month",
            yaxis_title="Fire Intensity (BI)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_ts2:
        st.subheader("📊 Monthly Stats")
        
        # Find current month
        current_month = datetime.datetime.now().month
        
        if current_month in trend_data:
            current_intensity = trend_data[current_month]
            avg_intensity = sum(trend_data.values()) / len(trend_data)
            
            st.metric(
                f"{month_names[current_month-1]} Intensity",
                f"{current_intensity:.1f}",
                f"{current_intensity - avg_intensity:+.1f}"
            )
            
            # Find peak month
            peak_month_num = max(trend_data, key=trend_data.get)
            peak_intensity = trend_data[peak_month_num]
            
            st.metric(
                f"Peak ({month_names[peak_month_num-1]})",
                f"{peak_intensity:.1f}"
            )
            
            # Risk months count
            high_months = sum(1 for intensity in intensities if intensity > 60)
            st.metric("High Risk Months", high_months)

    # --- NEW VISUAL 2: PCA CLUSTER VISUALIZATION ---
    st.divider()
    col_pca1, col_pca2 = st.columns([3, 1])
    
    with col_pca1:
        st.subheader("🔍 PCA Dimensionality Reduction")
        st.caption("Visualizing weather patterns in 2D space")
        
        pca_x = result['pca_x']
        pca_y = result['pca_y']
        
        # Create a scatter plot with simulated historical data
        np.random.seed(42)
        n_points = 100
        historical_x = np.random.normal(0, 1.5, n_points)
        historical_y = np.random.normal(0, 1.5, n_points)
        
        # Create DataFrame
        pca_df = pd.DataFrame({
            'PC1': list(historical_x) + [pca_x],
            'PC2': list(historical_y) + [pca_y],
            'Type': ['Historical Pattern'] * n_points + ['Current Condition']
        })
        
        # Create scatter plot
        fig_pca = px.scatter(
            pca_df, 
            x='PC1', 
            y='PC2', 
            color='Type',
            title="Principal Component Analysis (Weather Pattern Space)",
            color_discrete_map={
                'Historical Pattern': 'lightblue',
                'Current Condition': 'red'
            },
            opacity=0.7,
            size=[1] * n_points + [15]  # Larger marker for current condition
        )
        
        # Customize markers
        fig_pca.update_traces(
            marker=dict(
                symbol=['circle'] * n_points + ['star'],
                line=dict(width=1, color='DarkSlateGrey')
            ),
            selector=dict()
        )
        
        # Add annotations
        fig_pca.add_annotation(
            x=pca_x,
            y=pca_y,
            text="You Are Here",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black",
            ax=40,
            ay=-40
        )
        
        fig_pca.update_layout(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig_pca, use_container_width=True)
    
    with col_pca2:
        st.subheader("📈 PCA Insights")
        
        st.metric("PC1 Score", f"{pca_x:.2f}")
        st.metric("PC2 Score", f"{pca_y:.2f}")
        
        # Interpret PCA position
        if abs(pca_x) > 1.5:
            st.info("🔸 **Extreme PC1**: Unusual temperature/humidity pattern")
        else:
            st.info("✅ **Normal PC1**: Typical thermal conditions")
            
        if abs(pca_y) > 1.5:
            st.warning("🔸 **Extreme PC2**: Atypical wind/precipitation pattern")
        else:
            st.info("✅ **Normal PC2**: Standard atmospheric conditions")

# --- MLOps SECTION ---
st.divider()
st.subheader("⚙️ MLOps Pipeline Control")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.info("🔍 **System Health**")
    if st.button("Check API Health", use_container_width=True):
        try:
            res = requests.get(f"{API_URL}/health")
            health_data = res.json()
            
            if health_data['status'] == 'healthy':
                st.success("✅ All systems operational")
            else:
                st.warning(f"⚠️ {health_data['status']}: {health_data.get('detail', '')}")
            
            st.json(health_data)
        except:
            st.error("❌ API Offline")

with col_b:
    st.warning("🔄 **Model Management**")
    
    if st.button("View Model Info", use_container_width=True):
        try:
            st.info("Model: Random Forest Regressor v1.0")
            st.json({"type": "sklearn", "artifact": "regression_model.pkl"})
        except:
            st.error("Cannot fetch model info")
    
    if st.button("Get Seasonal Trends", use_container_width=True):
        # We reuse the logic if it's stored in session state, else fetch
        if 'prediction_result' in st.session_state:
             st.json(st.session_state.prediction_result.get('seasonal_trend', {}))
        else:
             st.warning("Run a prediction first to load seasonal data.")

with col_c:
    st.error("🚀 **Training Pipeline**")
    
    if st.button("Trigger Retraining", use_container_width=True):
        # In a real app, this would hit a /retrain endpoint
        with st.spinner("Orchestrating training pipeline..."):
            # Simulate training process
            import time
            time.sleep(2)
            
            st.success("✅ Retraining request sent to Prefect Orchestrator!")
            st.balloons()
        
        st.caption("Check terminal logs for 'flows/training_flow.py' execution")

# Footer
st.divider()
st.caption("""
**Wildfire Intelligence Platform v1.2.0** | 
Built with ❤️ using FastAPI, Scikit-learn, Gemini AI, and Streamlit
""")