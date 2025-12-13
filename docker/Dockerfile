# Use an official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# --- NEW LINE: Fix Windows Line Endings (CRLF -> LF) ---
RUN sed -i 's/\r$//' docker/start.sh

# Make the start script executable
RUN chmod +x docker/start.sh

# Expose the port
EXPOSE 8000

# COMMAND: Run the script (Train -> Start Server)
CMD ["./docker/start.sh"]