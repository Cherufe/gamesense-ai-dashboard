# backend/llm_service.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Muat API Key dari file .env untuk lokal
load_dotenv()


def get_gemini_api_key():
    """
    Mengambil API key Gemini dari beberapa sumber:
    1. Environment variable / file .env untuk lokal
    2. Streamlit Secrets untuk deployment Streamlit Cloud
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


API_KEY = get_gemini_api_key()

if not API_KEY:
    print("⚠️ Peringatan: GEMINI_API_KEY tidak ditemukan di .env atau Streamlit Secrets!")

# Konfigurasi Gemini hanya jika API key tersedia
if API_KEY:
    genai.configure(api_key=API_KEY)

# Kita gunakan model Gemini Flash karena cepat untuk teks rekomendasi
gemini_model = genai.GenerativeModel("gemini-2.5-flash") if API_KEY else None

def generate_actionable_insight(prediction: str, confidence: float, shap_features: list) -> str:
    """
    Fungsi untuk mengubah data matematis menjadi rekomendasi bisnis yang mudah dibaca.
    """
    if not API_KEY or gemini_model is None:
        return "Rekomendasi AI tidak tersedia karena API Key belum dikonfigurasi."

    # 2. Merakit Prompt (Instruksi untuk AI)
    # Ekstrak nama fitur agar kalimatnya rapi
    reasons = [f"- {item['feature']} ({item['type']} impact)" for item in shap_features]
    reasons_text = "\n".join(reasons)

    prompt = f"""
    Kamu adalah seorang Ahli Analisis Game Online dan Pakar Retensi Pemain. 
    Saya memiliki seorang pemain dengan detail prediksi sebagai berikut:
    
    - Prediksi Tingkat Engagement: {prediction}
    - Tingkat Keyakinan Model: {confidence}%
    - Faktor Utama (SHAP) yang mempengaruhi pemain ini masuk ke kategori tersebut:
    {reasons_text}
    
    Berdasarkan data di atas, berikan analisis singkat (maksimal 2 kalimat) mengapa pemain ini berada di level '{prediction}'. 
    Kemudian, berikan 1 rekomendasi tindakan bisnis atau fitur in-game yang spesifik dan taktis (maksimal 2 kalimat) untuk meningkatkan atau menjaga engagement pemain ini.
    
    Gunakan bahasa Indonesia yang profesional namun asik layaknya tim pengembang game.
    """

    try:
        # 3. Minta jawaban dari Gemini
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gagal mendapatkan rekomendasi AI: {str(e)}"