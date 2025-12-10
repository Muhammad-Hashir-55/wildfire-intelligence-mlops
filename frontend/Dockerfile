FROM python:3.9-slim

WORKDIR /app

# Install Streamlit and requests
RUN pip install streamlit requests pandas plotly

# Copy the UI code
COPY ui.py .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]