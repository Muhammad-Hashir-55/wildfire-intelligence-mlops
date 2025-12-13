# #!/bin/bash

# # 1. Run Training
# echo "🔥 [Startup] Triggering Auto-Training Pipeline..."
# python src/train.py

# # Check if training succeeded
# if [ $? -eq 0 ]; then
#     echo "✅ [Startup] Training Complete. Models are ready."
# else
#     echo "❌ [Startup] Training Failed! Checking for existing models..."
#     # Optional: You could decide to exit here if strict
# fi

# # 2. Start the API
# echo "🚀 [Startup] Launching FastAPI..."
# exec uvicorn app.main:app --host 0.0.0.0 --port 8000

#!/bin/bash

# 1. Run Training (COMMENTED OUT FOR FAST STARTUP)
# echo "🔥 [Startup] Triggering Auto-Training Pipeline..."
# python src/train.py

# 2. Start the API (This runs immediately now)
echo "🚀 [Startup] Launching FastAPI..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000