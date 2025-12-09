import pandas as pd
import os

# Define paths
RAW_DATA_PATH = "data/raw/california_wildfire.csv"
PROCESSED_DATA_PATH = "data/processed/Wildfire_Dataset.csv"
# California Bounding Box
LAT_MIN, LAT_MAX = 32.5, 42.0
LON_MIN, LON_MAX = -124.5, -114.0

def process_data():
    print("🔥 Starting Data Processing... (Chunking 9.5M rows)")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: File not found at {RAW_DATA_PATH}")
        return

    chunk_size = 100000 
    chunks = []
    
    # Read in chunks to handle the 1.3GB size
    for i, chunk in enumerate(pd.read_csv(RAW_DATA_PATH, chunksize=chunk_size)):
        if i % 10 == 0:
            print(f"   Processing chunk {i}...")

        # Filter for California Coordinates using new column names
        if 'latitude' in chunk.columns and 'longitude' in chunk.columns:
            cali_chunk = chunk[
                (chunk['latitude'] >= LAT_MIN) & 
                (chunk['latitude'] <= LAT_MAX) &
                (chunk['longitude'] >= LON_MIN) & 
                (chunk['longitude'] <= LON_MAX)
            ]
            
            # Simple clean: Drop rows where critical weather info is missing
            cali_chunk = cali_chunk.dropna(subset=['bi', 'tmmn', 'rmax', 'vs'])
            chunks.append(cali_chunk)
    
    if chunks:
        df_cali = pd.concat(chunks)
        print(f"✅ Filtered Data Shape: {df_cali.shape}")
        
        # Save to processed folder
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df_cali.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"💾 Saved processed data to: {PROCESSED_DATA_PATH}")
    else:
        print("⚠️ No data found for the specified region.")

if __name__ == "__main__":
    process_data()