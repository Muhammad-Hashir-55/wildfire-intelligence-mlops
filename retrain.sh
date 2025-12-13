#!/bin/bash

echo "🔥 [1/2] Triggering Training inside Docker Container..."
# This sends the command into the running container
docker-compose exec api python src/train.py

# Check if the previous command worked
if [ $? -eq 0 ]; then
    echo "✅ Training Successful! Models updated."
    
    echo "🔄 [2/2] Restarting API to load new models..."
    docker-compose restart api
    
    echo "🚀 System is Live with New Brains!"
else
    echo "❌ Training Failed. API was not restarted."
fi