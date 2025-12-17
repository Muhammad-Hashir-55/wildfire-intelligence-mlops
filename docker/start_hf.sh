#!/bin/bash

echo "🔥 [Hybrid-Startup] Launching FastAPI Backend..."
# The '&' pushes FastAPI to the background so the script continues
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

echo "⏳ Waiting 5 seconds for API to initialize..."
sleep 5

echo "🖥️ [Hybrid-Startup] Launching Streamlit Frontend..."
# Streamlit runs in the foreground on the specific port Hugging Face requires (7860)
streamlit run frontend/ui.py --server.port 7860 --server.address 0.0.0.0