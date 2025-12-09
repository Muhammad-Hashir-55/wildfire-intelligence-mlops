from prefect import flow, task
import os
import pandas as pd

# Define paths
RAW_DATA_PATH = "data/raw/california_wildfire.csv"
PROCESSED_DATA_PATH = "data/processed/Wildfire_Dataset.csv"

@task(name="Check Raw Data")
def check_raw_data_exists():
    """Simulates checking a data lake or Kaggle for new files."""
    if os.path.exists(RAW_DATA_PATH):
        print(f"✅ Raw data found at: {RAW_DATA_PATH}")
        return True
    else:
        print("❌ Raw data missing! Please download the dataset.")
        raise FileNotFoundError("Raw dataset not found.")

@task(name="Validate Schema")
def validate_schema():
    """Ensures the raw data has the columns we need for training."""
    # We read just the header to be fast
    df = pd.read_csv(RAW_DATA_PATH, nrows=5)
    
    # Updated columns based on your dataset
    required_columns = ['latitude', 'longitude', 'tmmn', 'tmmx', 'bi']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"❌ Data Validation Failed. Missing columns: {missing}")
    print("✅ Schema Validation Passed.")

@flow(name="Wildfire Data Ingestion")
def data_ingestion_flow():
    """The Main Control Flow for Data."""
    print("🌊 Starting Data Ingestion Pipeline...")
    exists = check_raw_data_exists()
    if exists:
        validate_schema()
        # In a real app, you would trigger the preprocessing script here
        print("🚀 Data is ready for the Training Pipeline.")

if __name__ == "__main__":
    data_ingestion_flow()