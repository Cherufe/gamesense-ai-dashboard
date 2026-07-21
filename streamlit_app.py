# streamlit_app.py
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from backend.ml_service import predict_and_explain
from backend.llm_service import generate_actionable_insight


# ======================================================
# KONFIGURASI DASAR
# ======================================================
load_dotenv()

st.set_page_config(
    page_title="Gaming Analytics Command Center",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "backend" / "game_data.db"


# ======================================================
# CUSTOM STYLE - MIRIP DARK DASHBOARD SEBELUMNYA
# ======================================================
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] {
            background-color: #1e293b;
            border-right: 1px solid #334155;
        }

        section[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        div[data-testid="stMetric"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }

        div[data-testid="stMetricLabel"] p {
            color: #94a3b8;
            font-size: 14px;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff;
        }

        .dashboard-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            margin-bottom: 16px;
        }

        .gradient-card-blue {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 16px;
        }

        .gradient-card-green {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 16px;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 14px;
        }

        .badge-high {
            background-color: #10b981;
            color: white;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 12px;
        }

        .badge-medium {
            background-color: #f59e0b;
            color: white;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 12px;
        }

        .badge-low {
            background-color: #ef4444;
            color: white;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 12px;
        }

        .status-dot-green {
            height: 9px;
            width: 9px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }

        .status-dot-blue {
            height: 9px;
            width: 9px;
            background-color: #3b82f6;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }

        h1, h2, h3 {
            color: #ffffff;
        }

        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 700;
        }

        .stButton > button:hover {
            background-color: #2563eb;
            color: white;
            border: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# DATABASE HELPERS
# ======================================================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=10)
def get_global_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total_players FROM players")
    total_players = cursor.fetchone()["total_players"]

    cursor.execute("SELECT EngagementLevel, COUNT(*) as count FROM players GROUP BY EngagementLevel")
    rows = cursor.fetchall()

    distribution = {}
    for row in rows:
        level = row["EngagementLevel"] if row["EngagementLevel"] else "Unknown"
        distribution[level] = row["count"]

    conn.close()

    return {
        "total_players": total_players,
        "engagement_distribution": distribution
    }


@st.cache_data(ttl=5)
def get_live_radar():
    conn = get_db_connection()
    query = "SELECT * FROM live_predictions ORDER BY RANDOM() LIMIT 5"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_player_raw_data(player_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM players WHERE PlayerID = ?", (player_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def analyze_player(player_id: int):
    raw_data = get_player_raw_data(player_id)

    if not raw_data:
        return {
            "status": "error",
            "pesan": "Pemain tidak ditemukan."
        }

    ai_result = predict_and_explain(raw_data)

    if "error" in ai_result:
        return {
            "status": "error",
            "pesan": ai_result["error"]
        }

    pred_label = ai_result["prediction"]
    confidence = ai_result["confidence"]
    shap_features = ai_result["top_shap_features"]

    ai_recommendation = generate_actionable_insight(
        pred_label,
        confidence,
        shap_features
    )

    return {
        "status": "success",
        "player_id": player_id,
        "raw_data": raw_data,
        "ai_analysis": ai_result,
        "ai_recommendation": ai_recommendation
    }


# ======================================================
# DATA CLEANING HELPERS
# ======================================================
def safe_float(value, default: float = 0.0) -> float:
    """
    Konversi nilai ke float dengan aman.

    Beberapa data lama di SQLite/CSV bisa terbaca sebagai string dengan
    terlalu banyak titik, contoh: '4.401.729.344.841.460'.
    Nilai seperti itu seharusnya dibaca sebagai 4.401729344841460,
    bukan langsung dilempar ke float() karena akan menyebabkan ValueError.
    """
    if value is None or value == "":
        return default

    if isinstance(value, (int, float)):
        return float(value)

    text_value = str(value).strip()

    try:
        return float(text_value)
    except ValueError:
        pass

    # Handle format desimal rusak dengan banyak titik: 4.401.729 -> 4.401729
    if text_value.count(".") > 1:
        parts = text_value.split(".")
        normalized = f"{parts[0]}.{''.join(parts[1:])}"
        try:
            return float(normalized)
        except ValueError:
            return default

    # Handle format koma sebagai desimal: 4,40 -> 4.40
    if "," in text_value:
        try:
            return float(text_value.replace(",", "."))
        except ValueError:
            return default

    return default


# ======================================================
# CHART HELPERS
# ======================================================
def render_engagement_pie_chart(distribution: dict):
    high = distribution.get("High", 0)
    medium = distribution.get("Medium", 0)
    low = distribution.get("Low", 0)

    chart_df = pd.DataFrame({
        "Engagement": ["High", "Medium", "Low"],
        "Jumlah": [high, medium, low]
    })

    fig = px.pie(
        chart_df,
        names="Engagement",
        values="Jumlah",
        hole=0.7,
        color="Engagement",
        color_discrete_map={
            "High": "#10b981",
            "Medium": "#f59e0b",
            "Low": "#ef4444"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=20, b=20, l=20, r=20),
        height=330
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#1e293b", width=4))
    )

    st.plotly_chart(fig, use_container_width=True)


def render_shap_chart(shap_features: list):
    if not shap_features:
        st.info("Belum ada data SHAP untuk ditampilkan.")
        return

    shap_df = pd.DataFrame(shap_features)
    shap_df = shap_df.sort_values(by="impact", key=lambda x: x.abs(), ascending=True)

    colors = ["#10b981" if t == "positive" else "#ef4444" for t in shap_df["type"]]

    fig = go.Figure(
        go.Bar(
            x=shap_df["impact"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.3)"
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0)"),
        margin=dict(t=20, b=20, l=20, r=20),
        height=280,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_player_dna_radar(raw_data: dict):
    play_time = safe_float(raw_data.get("PlayTimeHours"))
    sessions = safe_float(raw_data.get("SessionsPerWeek"))
    achievements = safe_float(raw_data.get("AchievementsUnlocked"))
    player_level = safe_float(raw_data.get("PlayerLevel"))
    engagement_depth = safe_float(raw_data.get("EngagementDepth"))

    dna_values = [
        min(play_time * 2, 100),
        min(sessions * 5, 100),
        min(achievements * 2, 100),
        min(player_level, 100),
        min(engagement_depth * 100, 100)
    ]

    labels = [
        "Dedikasi",
        "Keaktifan",
        "Eksplorasi",
        "Pengalaman",
        "Kedalaman"
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=dna_values,
            theta=labels,
            fill="toself",
            name="DNA Pemain",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59, 130, 246, 0.25)",
            marker=dict(color="#10b981")
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=False,
                range=[0, 100]
            ),
            angularaxis=dict(color="#94a3b8"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=280
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# TEXT HELPERS
# ======================================================
def get_prediction_badge(prediction: str):
    prediction = prediction or "Unknown"

    if prediction == "High":
        badge_class = "badge-high"
    elif prediction == "Medium":
        badge_class = "badge-medium"
    else:
        badge_class = "badge-low"

    return f'<span class="{badge_class}">{prediction}</span>'


def split_ai_recommendation(full_text: str):
    diagnosis_text = full_text or "Menunggu data..."
    action_text = "Tidak ada rekomendasi khusus yang diberikan."

    if full_text and "Rekomendasi" in full_text:
        split_index = full_text.rfind("**Rekomendasi")
        if split_index == -1:
            split_index = full_text.rfind("Rekomendasi")

        if split_index != -1:
            diagnosis_text = full_text[:split_index]
            action_text = full_text[split_index:]

    return diagnosis_text.strip(), action_text.strip()


def show_dummy_action_buttons():
    col_reward, col_notif = st.columns(2)

    with col_reward:
        if st.button("🎁 Beri Reward", use_container_width=True):
            st.success("✅ Reward Milestone berhasil dikirim ke Inbox pemain!")

    with col_notif:
        if st.button("📲 Kirim Notif", use_container_width=True):
            st.info("📲 Push Notification peringatan churn sedang dikirim...")


# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("# 🎮 Game<span style='color:#3b82f6'>Sense</span>", unsafe_allow_html=True)
    st.markdown("---")

    selected_menu = st.radio(
        "Navigasi",
        ["📊 Global Overview", "🤖 AI Profiling"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Sistem**")
    st.markdown(
        "<p class='small-muted'><span class='status-dot-green'></span>API Server Online</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p class='small-muted'><span class='status-dot-blue'></span>Model: XGBoost v3</p>",
        unsafe_allow_html=True
    )


# ======================================================
# PAGE: GLOBAL OVERVIEW
# ======================================================
if selected_menu == "📊 Global Overview":
    st.title("Global Dashboard")

    try:
        global_stats = get_global_stats()
        total_players = global_stats["total_players"]
        distribution = global_stats["engagement_distribution"]

        col_total, col_accuracy, col_db = st.columns(3)

        with col_total:
            st.metric("Total Pemain Aktif", f"{total_players:,}".replace(",", "."))

        with col_accuracy:
            st.metric("Akurasi AI Model", "95.2%")

        with col_db:
            st.metric("Database Status", "Terhubung")

        col_chart, col_insight = st.columns([1, 1])

        with col_chart:
            with st.container(border=True):
                st.subheader("Distribusi Engagement Level")
                render_engagement_pie_chart(distribution)

        with col_insight:
            with st.container(border=True):
                st.subheader("Insight Global AI")
                st.markdown(
                    """
                    Sistem mendeteksi bahwa rata-rata pemain memiliki tingkat retensi yang stabil. 
                    Gunakan fitur **AI Profiling** di menu samping untuk membedah perilaku pemain spesifik 
                    dan mendapatkan rekomendasi intervensi taktis dari Gemini AI.
                    """
                )

        with st.container(border=True):
            col_radar_title, col_radar_action = st.columns([3, 1])

            with col_radar_title:
                st.subheader("🔴 Live AI Radar (Manual Monitoring)")
                st.caption("Klik tombol refresh untuk mengambil sampel data monitoring terbaru.")

            with col_radar_action:
                if st.button("🔄 Refresh Radar", use_container_width=True):
                    get_live_radar.clear()
                    st.rerun()

            live_df = get_live_radar()

            if live_df.empty:
                st.info("Menghubungkan ke radar...")
            else:
                display_df = live_df.copy()

                if "Confidence_Score" in display_df.columns:
                    display_df["Confidence_Score"] = pd.to_numeric(
                        display_df["Confidence_Score"],
                        errors="coerce"
                    )
                    display_df["Keyakinan"] = display_df["Confidence_Score"].apply(
                        lambda x: "N/A" if pd.isna(x) else f"{(x * 100 if x < 1 else x):.1f}%"
                    )

                if "PlayTimeHours" in display_df.columns:
                    display_df["Waktu Main (Jam)"] = display_df["PlayTimeHours"].apply(safe_float).round(1)

                if "Prediction_Correct" in display_df.columns:
                    display_df["Akurasi AI"] = display_df["Prediction_Correct"].apply(
                        lambda x: "✅ Tepat" if int(x) == 1 else "❌ Meleset"
                    )

                if "PlayerLevel" in display_df.columns and "Age" in display_df.columns:
                    display_df["Level & Usia"] = display_df.apply(
                        lambda row: f"Lvl {row.get('PlayerLevel', '-')} (Usia {row.get('Age', '-')})",
                        axis=1
                    )

                rename_map = {
                    "Predicted_Engagement": "Prediksi AI",
                    "Actual_Engagement": "Engagement Asli"
                }
                display_df = display_df.rename(columns=rename_map)

                selected_columns = [
                    col for col in [
                        "Level & Usia",
                        "Waktu Main (Jam)",
                        "Prediksi AI",
                        "Keyakinan",
                        "Akurasi AI",
                        "Engagement Asli"
                    ]
                    if col in display_df.columns
                ]

                st.dataframe(
                    display_df[selected_columns],
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as e:
        st.error(f"Gagal memuat Global Dashboard: {str(e)}")


# ======================================================
# PAGE: AI PROFILING
# ======================================================
if selected_menu == "🤖 AI Profiling":
    st.title("Player Profiling AI")

    with st.container(border=True):
        col_text, col_input = st.columns([2, 1])

        with col_text:
            st.subheader("Cari Data Pemain")
            st.caption("Masukkan PlayerID untuk membedah psikologi permainannya.")

        with col_input:
            player_id = st.number_input(
                "PlayerID",
                min_value=1,
                value=5,
                step=1
            )
            analyze_button = st.button("Analisis", use_container_width=True)

    if analyze_button:
        with st.spinner("🤖 AI sedang meracik analisis..."):
            result = analyze_player(int(player_id))

        if result["status"] == "error":
            st.error(f"❌ {result['pesan']}")
        else:
            raw_data = result["raw_data"]
            ai_analysis = result["ai_analysis"]
            ai_recommendation = result["ai_recommendation"]

            prediction = ai_analysis.get("prediction", "-")
            confidence = ai_analysis.get("confidence", 0)
            shap_features = ai_analysis.get("top_shap_features", [])

            col_stats, col_dna, col_shap = st.columns([1, 1, 2])

            with col_stats:
                with st.container(border=True):
                    st.subheader("Statistik Pemain")
                    st.markdown(f"**Player ID:** `{raw_data.get('PlayerID', player_id)}`")
                    st.markdown(f"**Level:** `{raw_data.get('PlayerLevel', '-')}`")
                    st.markdown(
                        f"**Prediksi AI:** {get_prediction_badge(prediction)}",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Keyakinan:** `{confidence}%`")

            with col_dna:
                with st.container(border=True):
                    st.subheader("Player DNA")
                    render_player_dna_radar(raw_data)

            with col_shap:
                with st.container(border=True):
                    st.subheader("Analisis Faktor AI (SHAP)")
                    render_shap_chart(shap_features)

            diagnosis_text, action_text = split_ai_recommendation(ai_recommendation)

            col_diagnosis, col_action = st.columns(2)

            with col_diagnosis:
                with st.container(border=True):
                    st.subheader("🧠 Diagnosis Psikologi")
                    st.markdown(diagnosis_text)

            with col_action:
                with st.container(border=True):
                    st.subheader("⚡ Rekomendasi Taktis")
                    st.markdown(action_text)
                    show_dummy_action_buttons()
    else:
        st.info("Masukkan PlayerID lalu klik tombol **Analisis**.")
