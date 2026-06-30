# 🎮 GameSense AI: Gaming Analytics Command Center

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![XGBoost](https://img.shields.io/badge/Machine_Learning-XGBoost-orange.svg)
![Gemini AI](https://img.shields.io/badge/Generative_AI-Google_Gemini-8E75B2.svg)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white)

**GameSense AI** adalah sebuah dasbor analitik tingkat *enterprise* yang memadukan kekuatan **Predictive Machine Learning (XGBoost)**, **Explainable AI (SHAP)**, dan **Generative AI (Google Gemini)**. Sistem ini dirancang untuk membaca "DNA Pemain" secara *real-time*, memprediksi tingkat *engagement* (retensi), dan memberikan rekomendasi taktis secara instan untuk mencegah *churn*.

---

## ✨ Fitur Utama

1. **📊 Global Overview Dashboard**
   Pemantauan metrik utama (KPI) dengan visualisasi *Doughnut Chart* dinamis untuk melihat distribusi *engagement* pemain (High, Medium, Low) di seluruh ekosistem game.

2. **📡 Live AI Radar (Real-Time Monitoring)**
   Sistem radar yang me-*refresh* data secara otomatis setiap 5 detik. Menampilkan prediksi status *engagement* pemain secara *live* beserta *Confidence Score* (Tingkat Keyakinan AI) dan status akurasinya.

3. **🤖 Player Profiling & Explainable AI (XAI)**
   - **Player DNA (Radar Chart):** Membedah gaya bermain (Dedikasi, Keaktifan, Eksplorasi, Pengalaman, Kedalaman).
   - **SHAP Analysis:** Grafik transparansi yang menjelaskan *mengapa* AI memberikan prediksi tertentu (faktor apa yang paling memengaruhi psikologi pemain).
   - **Gemini Tactical Intervention:** LLM menganalisis data mentah dan memberikan Diagnosis Psikologi serta Rekomendasi Taktis (Action Plan) yang spesifik untuk setiap pemain.

---

## 🛠️ Arsitektur & Tech Stack

Proyek ini dibangun menggunakan arsitektur pemisahan *Frontend* dan *Backend* yang dihubungkan melalui REST API.

- **Backend / API Server:** Python, FastAPI, Uvicorn
- **Database:** SQLite (Relational Database untuk menyimpan *raw data* pemain dan histori AI)
- **Machine Learning Core:** XGBoost (Klasifikasi), SHAP (Model Interpretability), Pandas, Scikit-learn
- **LLM Integration:** `google-genai` (Google Gemini API)
- **Frontend / UI:** HTML5, Tailwind CSS (Dark/Gold Organic Theme), Chart.js (Radar, Doughnut, Bar charts)

---

## 🚀 Cara Instalasi & Menjalankan Aplikasi

Aplikasi ini dirancang *plug-and-play*. File *database* (`game_data.db`) dan model ML (`.pkl`) sudah disertakan agar sistem dapat langsung diuji coba.

### 1. Clone Repository

```bash
git clone https://github.com/UsernameKamu/gamesense-ai-dashboard.git
cd gamesense-ai-dashboard
```

### 2. Siapkan Virtual Environment (Opsional tapi Direkomendasikan)

```bash
python -m venv env_game

# Windows:
env_game\Scripts\activate

# Mac/Linux:
source env_game/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment Variables

Buat file bernama `.env` di dalam root folder proyek, lalu masukkan API Key Gemini Anda:

```
GEMINI_API_KEY=masukkan_api_key_google_gemini_anda_di_sini
```

### 5. Jalankan Server FastAPI

```bash
uvicorn backend.main:app --reload
```

### 6. Buka Dashboard

Buka browser Anda dan kunjungi URL berikut untuk mengakses antarmuka pengguna:

👉 **http://127.0.0.1:8000/**

---

## 🧠 Workflow Sistem AI

1. **Data Ingestion** — Backend membaca `PlayerID` yang dikirim dari Frontend lalu melakukan query ke database SQLite.
2. **Predictive Scoring** — Model XGBoost menerima *features* pemain dan mengembalikan prediksi (High, Medium, Low) beserta probabilitasnya.
3. **Feature Attribution** — SHAP menghitung kontribusi masing-masing *feature* terhadap hasil akhir prediksi.
4. **Cognitive Synthesis** — Data mentah + Hasil Prediksi + Nilai SHAP dibungkus menjadi *prompt* dinamis dan dikirim ke Google Gemini.
5. **Actionable Output** — Gemini merespons dengan format Markdown terstruktur, yang kemudian di-*parsing* oleh Frontend menjadi kartu Diagnosis dan Rekomendasi secara organik.

---

Developed with ☕ and Code by **Bryanka Jordaneo Vanky Heizer**

