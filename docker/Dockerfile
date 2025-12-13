# Use an official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# We keep this to ensure libraries like numpy/pandas work correctly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Expose the port
EXPOSE 8000

# COMMAND: Run FastAPI directly
# We removed "start.sh" so there are no more Windows formatting errors.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]