# backend/setup_db.py
import pandas as pd
from sqlalchemy import create_engine
import os

print("--- Memulai Proses Migrasi CSV ke SQLite ---")

# 1. Tentukan path file CSV (sesuaikan jika path-mu berbeda)
csv_path = r"D:\engagement_project\models\online_gaming_behavior_insights - online_gaming_behavior_insights.csv"

# 👇 UPDATE: Paksa DB tersimpan di dalam folder backend/ agar tidak nyasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH_FIXED = os.path.join(BASE_DIR, "game_data.db")
db_path = f"sqlite:///{DB_PATH_FIXED}"
# 👆 AKHIR UPDATE

try:
    # 2. Baca CSV Pertama (Data Utama)
    df = pd.read_csv(csv_path)
    print(f"✅ Data CSV Utama berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    
    # 3. Buat koneksi ke SQLite
    engine = create_engine(db_path)
    
    # 4. Masukkan data ke dalam tabel bernama 'players'
    # index=False agar index pandas tidak ikut masuk jadi kolom
    df.to_sql('players', con=engine, if_exists='replace', index=False)
    
    print(f"✅ Berhasil! Data telah dipindahkan ke {DB_PATH_FIXED} dalam tabel 'players'.")

    # ==============================================================
    # 👇 TAMBAHAN BARU UNTUK FITUR LIVE RADAR (TANPA HAPUS KODE LAMA) 👇
    # ==============================================================
    print("\n--- Memulai Proses Migrasi Data Prediksi AI (Live Radar) ---")
    
    # Tentukan path file CSV hasil prediksi
    csv_path_live = r"D:\engagement_project\models\hasil_prediksi_engagement.csv"
    
    # Baca CSV Kedua (Hasil Prediksi)
    df_live = pd.read_csv(csv_path_live)
    print(f"✅ Data CSV Prediksi berhasil dimuat: {df_live.shape[0]} baris, {df_live.shape[1]} kolom.")
    
    # Masukkan data ke dalam tabel baru bernama 'live_predictions'
    df_live.to_sql('live_predictions', con=engine, if_exists='replace', index=False)
    
    print("✅ Berhasil! Data prediksi telah dimasukkan ke tabel 'live_predictions'.")
    # ==============================================================

except Exception as e:
    print(f"❌ Terjadi kesalahan: {e}")