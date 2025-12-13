#!/bin/bash

echo "🔥 [1/2] Triggering Prefect Orchestration Flow..."
# OLD COMMAND: docker-compose exec api python src/train.py
# NEW COMMAND: Runs the flow, which trains AND sends the Discord alert
docker-compose exec api python flows/training_flow.py

# Check if the flow succeeded
if [ $? -eq 0 ]; then
    echo "✅ Pipeline Success! Discord Notification Sent."
    
    echo "🔄 [2/2] Restarting API to load new models..."
    docker-compose restart api
    
    echo "🚀 System is Live with New Brains!"
else
    echo "❌ Flow Failed. Check logs above."
fi