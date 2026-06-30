# backend/ml_service.py
import pandas as pd
import numpy as np
import joblib
import shap
import os

# 1. Path ke model yang ada di folder models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/xgboost_engagement_model.pkl")

# 2. Load model global ke RAM saat server nyala
try:
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    print("✅ Otak AI (XGBoost & SHAP) berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model = None
    explainer = None

# Pastikan urutan fitur sama persis dengan yang ada di log training kamu (24 Fitur)
EXPECTED_FEATURES = [
    'Age', 'Gender', 'PlayTimeHours', 'InGamePurchases', 'GameDifficulty', 
    'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked', 
    'Location_Europe', 'Location_Other', 'Location_USA', 'GameGenre_RPG', 
    'GameGenre_Simulation', 'GameGenre_Sports', 'GameGenre_Strategy', 
    'GamingIntensity', 'SessionConsistency', 'AchievementRate', 'LevelSpeed', 
    'SessionsPerLevel', 'PlaytimeEfficiency', 'AgeLevelInteraction', 'EngagementDepth'
]

def preprocess_player_data(raw_data: dict) -> pd.DataFrame:
    """Melakukan Feature Engineering persis seperti di Jupyter Notebook-mu"""
    df = pd.DataFrame([raw_data])
    
    # --- 1. Encoding Manual ---
    if 'InGamePurchases' in df.columns and df['InGamePurchases'].dtype == object:
        df['InGamePurchases'] = df['InGamePurchases'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    diff_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    if 'GameDifficulty' in df.columns and df['GameDifficulty'].dtype == object:
        df['GameDifficulty'] = df['GameDifficulty'].map(diff_map).fillna(0)
        
    # Asumsi Gender: Male=1, Female=0 (Sesuaikan jika LabelEncoder-mu kebalik)
    gender_map = {'Male': 1, 'Female': 0} 
    if 'Gender' in df.columns and df['Gender'].dtype == object:
        df['Gender'] = df['Gender'].map(gender_map).fillna(0)

    # --- 2. Dummies Manual ---
    loc = df['Location'].iloc[0] if 'Location' in df.columns else ''
    df['Location_Europe'] = 1 if loc == 'Europe' else 0
    df['Location_Other'] = 1 if loc == 'Other' else 0
    df['Location_USA'] = 1 if loc == 'USA' else 0

    genre = df['GameGenre'].iloc[0] if 'GameGenre' in df.columns else ''
    df['GameGenre_RPG'] = 1 if genre == 'RPG' else 0
    df['GameGenre_Simulation'] = 1 if genre == 'Simulation' else 0
    df['GameGenre_Sports'] = 1 if genre == 'Sports' else 0
    df['GameGenre_Strategy'] = 1 if genre == 'Strategy' else 0

    # --- 3. Fitur Engineering Baru (Rumus dari kodemu) ---
    df['GamingIntensity'] = df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']
    df['SessionConsistency'] = df['SessionsPerWeek'] / (df['AvgSessionDurationMinutes'] + 1)
    df['AchievementRate'] = df['AchievementsUnlocked'] / (df['PlayerLevel'] + 1)
    df['LevelSpeed'] = df['PlayerLevel'] / (df['SessionsPerWeek'] + 1)
    df['SessionsPerLevel'] = df['SessionsPerWeek'] / (df['PlayerLevel'] + 1)
    df['PlaytimeEfficiency'] = df['AchievementsUnlocked'] / (df['GamingIntensity'] + 1)
    df['AgeLevelInteraction'] = df['Age'] * df['PlayerLevel']
    
    # Skoring EngagementDepth raw
    df['EngagementDepth'] = (df['AchievementsUnlocked'] * 0.4 + df['PlayerLevel'] * 0.3 + df['SessionsPerWeek'] * 0.3)
    
    # Pastikan format dan urutan sesuai XGBoost
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
            
    # 👇 TAMBAHKAN SATU BARIS INI UNTUK FIX ERROR DTYPES 👇
    # Paksa semua data menjadi angka numerik, jika ada error jadikan NaN, lalu isi dengan 0
    df_final = df[EXPECTED_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
            
    return df_final

def predict_and_explain(raw_data: dict):
    """Memprediksi dan membedah logika menggunakan SHAP"""
    if model is None:
        return {"error": "Model .pkl tidak ditemukan!"}
        
    # 1. Transformasi Data
    X_input = preprocess_player_data(raw_data)
    
    # 2. Prediksi XGBoost
    probs = model.predict_proba(X_input)[0]
    pred_idx = int(np.argmax(probs))
    classes = ['Low', 'Medium', 'High']
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])
    
    # 3. Hitung SHAP (Logika Explainable AI)
    shap_vals = explainer.shap_values(X_input)
    # XGBoost multiclass menghasilkan list array. Kita ambil yang sesuai prediksi targetnya.
    shap_vals_for_pred = shap_vals[pred_idx][0] 
    
    feature_impacts = []
    for i, feature in enumerate(EXPECTED_FEATURES):
        impact_val = float(shap_vals_for_pred[i])
        if impact_val != 0:
            feature_impacts.append({
                "feature": feature,
                "impact": round(impact_val, 4),
                "type": "positive" if impact_val > 0 else "negative"
            })
    
    # Sortir dan ambil top 5 alasan utama
    feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    top_features = feature_impacts[:5]
    
    return {
        "prediction": pred_label,
        "confidence": round(confidence * 100, 2), # Jadikan persen
        "top_shap_features": top_features
    }