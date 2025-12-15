from prefect import flow, task
import requests
import sys
import os

# Add the project root to python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your actual training logic
from src.train import train_all_tasks

# --- CONFIG ---
# Replace this with your actual Discord Webhook URL for the Bonus
# If you don't have one, leave it blank, the code handles it safely.
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1450136736238796947/Bl0JBjHUUeQbiqt7f_o-76zcrnVlu1-bHHHKZS8Y4r-dORJwZVJOy-5kyB2z6pPNWoKE" 

@task(name="Run ML Training")
def run_training_script():
    """Executes the Random Forest & Clustering training."""
    try:
        print("🔥 Orchestrator starting model training...")
        train_all_tasks() # Calls the function you wrote yesterday
        return "Success"
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        return "Failed"

@task(name="Send Notification")
def send_discord_alert(status: str):
    """Sends a notification to Discord/Slack upon completion."""
    if not DISCORD_WEBHOOK_URL or "YOUR_DISCORD" in DISCORD_WEBHOOK_URL:
        print(f"🔔 (Simulation) Notification: Training finished with status: {status}")
        return

    message = {
        "content": f"🚨 **Wildfire MLOps Alert** 🚨\nTraining Pipeline Completed.\nStatus: **{status}**\nModels are ready for deployment."
    }
    
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=message)
        print("✅ Discord Notification Sent!")
    except Exception as e:
        print(f"⚠️ Could not send notification: {e}")

@flow(name="Wildfire Model Retraining")
def training_flow():
    """The Master Flow that trains and notifies."""
    print("⚙️ Starting Retraining Flow...")
    
    # Step 1: Train
    status = run_training_script()
    
    # Step 2: Notify
    send_discord_alert(status)

if __name__ == "__main__":
    training_flow()