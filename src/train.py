import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = "data/processed/california_wildfire.csv"
MODEL_DIR = "models"

def train_all_tasks():
    print("🚀 Loading Processed Data...")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure we have data
    if df.empty:
        print("❌ Error: Dataset is empty. Run preprocess.py first.")
        return

    # ==========================================
    # TASK 1: REGRESSION (Predict Fire Intensity)
    # Target: 'bi' (Burning Index)
    # Features: Weather metrics (Temp, Humidity, Wind, Rain)
    # ==========================================
    print("\n🔥 Training Task 1: Regression (Predict Burning Index)...")
    
    # Features: Temp Min/Max, Humidity Min/Max, Wind Speed, Precipitation, Energy Release Component
    reg_features = ['tmmn', 'tmmx', 'rmin', 'rmax', 'vs', 'pr', 'erc']
    target_reg = 'bi'
    
    X_reg = df[reg_features]
    y_reg = df[target_reg]
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    reg_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    reg_model.fit(X_train_r, y_train_r)
    
    rmse = np.sqrt(mean_squared_error(y_test_r, reg_model.predict(X_test_r)))
    print(f"✅ Regression RMSE: {rmse:.4f}")
    
    joblib.dump(reg_model, f"{MODEL_DIR}/regression_model.pkl")

    # ==========================================
    # TASK 2: CLASSIFICATION (Predict Risk Level)
    # Target: Custom 'Risk_Level' based on Burning Index
    # Logic: 0-40 Low, 40-80 Medium, >80 High
    # ==========================================
    print("\n⚠️ Training Task 2: Classification (Fire Risk Level)...")
    
    def get_risk_level(bi_val):
        if bi_val < 40: return 'Low'
        elif bi_val < 80: return 'Medium'
        else: return 'High'
        
    df['risk_level'] = df['bi'].apply(get_risk_level)
    
    # Encode Target (Low=0, Medium=1, High=2)
    le = LabelEncoder()
    y_clf = le.fit_transform(df['risk_level'])
    
    # Use same weather features for classification
    X_clf = df[reg_features]
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    clf_model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    clf_model.fit(X_train_c, y_train_c)
    
    acc = accuracy_score(y_test_c, clf_model.predict(X_test_c))
    print(f"✅ Classification Accuracy: {acc:.4f}")
    
    # Save Model + Encoder (needed to decode predictions later)
    joblib.dump(clf_model, f"{MODEL_DIR}/classification_model.pkl")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

    # ==========================================
    # TASK 3: CLUSTERING (Recovery Zones)
    # Group by Location (Lat/Lon) and Fire Intensity (bi)
    # ==========================================
    print("\n🌍 Training Task 3: Clustering (Recovery Zones)...")
    
    X_cluster = df[['latitude', 'longitude', 'bi']]
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    
    joblib.dump(kmeans, f"{MODEL_DIR}/clustering_model.pkl")
    print("✅ Clustering Model Saved.")
    print("\n🎉 All Systems Go! Models are ready in 'models/'")

if __name__ == "__main__":
    train_all_tasks()