from prefect import flow, task
import requests
import sys
import os

# Add the project root to python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your actual training logic
from src.train import train_all_tasks

# --- CONFIG ---
# ⚠️ SECURITY NOTE: Ideally, use os.getenv('DISCORD_WEBHOOK_URL')
# But for your project demo, this hardcoded variable works.
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1450136736238796947/Bl0JBjHUUeQbiqt7f_o-76zcrnVlu1-bHHHKZS8Y4r-dORJwZVJOy-5kyB2z6pPNWoKE"

# 1. RETRY LOGIC (Satisfies Project Objective 3)
@task(name="Run ML Training", retries=3, retry_delay_seconds=10)
def run_training_script():
    """
    Executes the Random Forest & Clustering training.
    NO try/except here! We want it to fail so Prefect triggers retries.
    """
    print("🔥 Orchestrator starting model training...")
    train_all_tasks()  # If this crashes, Prefect will auto-retry 3 times
    return "Success"

@task(name="Send Notification")
def send_discord_alert(status: str):
    """Sends a notification to Discord/Slack upon completion."""
    if not DISCORD_WEBHOOK_URL or "YOUR_DISCORD" in DISCORD_WEBHOOK_URL:
        print(f"🔔 (Simulation) Notification: Training finished with status: {status}")
        return

    # Dynamic Message based on Status
    color_code = 5763719 if status == "Success" else 15548997  # Green or Red
    
    message = {
        "embeds": [{
            "title": f"🚨 Wildfire MLOps Pipeline: {status}",
            "description": "The automated retraining pipeline has finished execution.",
            "color": color_code,
            "fields": [
                {"name": "Status", "value": f"**{status}**", "inline": True},
                {"name": "Retries Used", "value": "Checked via Prefect", "inline": True}
            ],
            "footer": {"text": "Certified MLOps System"}
        }]
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
    
    final_status = "Failed" # Default to failed
    
    # 2. ERROR HANDLING (Satisfies Project Objective 3)
    try:
        # This will retry 3 times if it crashes. 
        # If it fails the 4th time, it raises the Exception to here.
        run_training_script() 
        final_status = "Success"
        
    except Exception as e:
        print(f"❌ CRITICAL: Pipeline failed after retries! Error: {e}")
        final_status = "Failed"
    
    # 3. NOTIFICATION (Satisfies Project Objective 3)
    # We run this regardless of success or failure
    send_discord_alert(final_status)

if __name__ == "__main__":
    training_flow()