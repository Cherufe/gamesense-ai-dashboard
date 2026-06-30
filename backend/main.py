# backend/main.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from backend.ml_service import predict_and_explain
from fastapi.middleware.cors import CORSMiddleware
from backend.ml_service import predict_and_explain
from backend.llm_service import generate_actionable_insight
import sqlite3
import os

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Gaming Engagement API",
    description="API untuk memprediksi dan menganalisis level engagement pemain",
    version="1.0.0"
)

# Pengaturan CORS (Sangat penting agar Frontend Vanilla JS bisa akses API ini)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Saat produksi, ganti "*" dengan URL Vercel/Frontend kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fungsi untuk mendapatkan path database yang dinamis dan aman
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "game_data.db")

# Fungsi helper untuk koneksi database
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Agar hasil query bisa diakses seperti dictionary
    return conn

# --- ENDPOINT API ---

# 1. Endpoint Root (Health Check)
@app.get("/")
def read_root():
    # Cari lokasi html di folder frontend
    html_path = os.path.join(BASE_DIR, r"D:\engagement_project\frontend\dashboard.html")

    # cek apakah file ada 
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {"staus": "error", "pesan": "File dashbord.html tidak di temukan di folder html!!"}

# 2. Endpoint Tes Database (Ambil total pemain & distribusi engagement)
@app.get("/api/stats/global")
def get_global_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query untuk menghitung total pemain
        cursor.execute("SELECT COUNT(*) as total_players FROM players")
        total = cursor.fetchone()["total_players"]
        
        # Query untuk menghitung porsi High, Medium, Low
        cursor.execute("SELECT EngagementLevel, COUNT(*) as count FROM players GROUP BY EngagementLevel")
        rows = cursor.fetchall()
        
        distribution = {}
        for r in rows:
            # Mengantisipasi jika ada data kosong
            level = r["EngagementLevel"] if r["EngagementLevel"] else "Unknown"
            distribution[level] = r["count"]
            
        conn.close()
        return {
            "status": "success",
            "data": {
                "total_players": total,
                "engagement_distribution": distribution
            }
        }
    except Exception as e:
        return {"status": "error", "pesan": str(e)}
    
# 3. Endpoint Inti: Analisis Profil Pemain (Prediksi + SHAP + Gemini)
@app.get("/api/player/{player_id}")
def analyze_player(player_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM players WHERE PlayerID = ?", (player_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {"status": "error", "pesan": "Pemain tidak ditemukan."}
            
        raw_data = dict(row)
        
        # 1. Masukkan ke Mesin AI XGBoost (Prediksi & SHAP)
        ai_result = predict_and_explain(raw_data)
        
        # Jika model gagal dimuat
        if "error" in ai_result:
            return {"status": "error", "pesan": ai_result["error"]}
            
        # 2. Masukkan ke Mesin AI Gemini (Rekomendasi Teks)
        # Kita ambil data hasil XGBoost untuk dijadikan bahan prompt Gemini
        pred_label = ai_result["prediction"]
        confidence = ai_result["confidence"]
        shap_features = ai_result["top_shap_features"]
        
        # Panggil fungsi dari llm_service.py
        ai_recommendation = generate_actionable_insight(pred_label, confidence, shap_features)
        
        # 3. Gabungkan semuanya ke dalam satu JSON yang rapi!
        return {
            "status": "success",
            "player_id": player_id,
            "raw_data": raw_data,
            "ai_analysis": ai_result,
            "ai_recommendation": ai_recommendation  # <--- TEKS DARI GEMINI MUNCUL DI SINI
        }
    except Exception as e:
        return {"status": "error", "pesan": str(e)}

# =====================================================================
# 👇 TAMBAHAN BARU UNTUK FITUR LIVE RADAR 👇
# =====================================================================

# 4. Endpoint Live AI Radar (Mengambil 5 data acak dari hasil prediksi)
@app.get("/api/live-radar")
def get_live_radar():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ambil 5 baris acak dari tabel live_predictions
        cursor.execute("SELECT * FROM live_predictions ORDER BY RANDOM() LIMIT 5")
        rows = cursor.fetchall()
        conn.close()
        
        # Ubah data menjadi format list dictionary agar mudah dibaca Frontend
        live_data = [dict(row) for row in rows]
        
        return {
            "status": "success",
            "data": live_data
        }
    except Exception as e:
        return {"status": "error", "pesan": str(e)}