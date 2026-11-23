import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Konfigurasi Halaman Responsif
st.set_page_config(
    page_title="Air Quality Telemetry & AI Diagnostics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS Tema Gelap Industri + Highlight Tab & Tombol Aktif
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0b1120 !important;
        color: #f8fafc !important;
    }
    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stMarkdown {
        color: #f8fafc !important;
    }
    /* Tombol Menu Tidak Aktif (Default) */
    .stButton > button[kind="secondary"] {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        background-color: #273549 !important;
    }
    /* Sorotan Tab Menu Aktif */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(2, 132, 199, 0.45) !important;
    }
    /* Expander */
    [data-testid="stExpander"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .header-box {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%) !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .header-title { font-size: 22px; font-weight: 800; color: #f8fafc !important; margin: 0; line-height: 1.25; }
    .header-sub { font-size: 13px; color: #cbd5e1 !important; margin-top: 6px; }
    .status-badge {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.2) !important;
        color: #34d399 !important;
        border: 1px solid #059669 !important;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-card {
        background-color: #1e293b !important;
        border-radius: 10px;
        padding: 16px 18px;
        border-left: 6px solid #38bdf8;
        margin-bottom: 10px;
    }
    .card-label { font-size: 11.5px; color: #cbd5e1 !important; font-weight: 700; }
    .card-value { font-size: 24px; font-weight: 800; margin: 4px 0; }
    .card-sub { font-size: 12.5px; color: #f1f5f9 !important; }

    @media (max-width: 768px) {
        .header-title { font-size: 17px; }
        .header-sub { font-size: 11.5px; }
        .card-value { font-size: 20px; }
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Dataset & Pelatihan Model
@st.cache_resource
def load_and_train():
    train_df = pd.read_csv('Train_Data1_20pct.csv', sep=';')
    test_df = pd.read_csv('Test_Data1_20pct.csv', sep=';')

    fitur = [
        'pm25_concentration_ugm3', 'pm10_concentration_ugm3',
        'carbon_monoxide_co_ugm3', 'sulfur_dioxide_so2_ugm3',
        'nitrogen_dioxide_no2_ugm3', 'ozone_o3_ugm3',
        'ambient_temperature_celsius', 'relative_humidity_pct',
        'wind_speed_kmh', 'rainfall_intensity_mm'
    ]

    train_clean = train_df[fitur + ['Target_Encoded']].dropna()
    test_clean = test_df[fitur + ['Target_Encoded']].dropna().reset_index(drop=True)

    X_train, y_train = train_clean[fitur], train_clean['Target_Encoded'].astype(int)
    X_test, y_test = test_clean[fitur], test_clean['Target_Encoded'].astype(int)

    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    svm = SVC(kernel='rbf', C=1.0, random_state=42).fit(X_train, y_train)

    acc_rf = accuracy_score(y_test, rf.predict(X_test)) * 100
    acc_svm = accuracy_score(y_test, svm.predict(X_test)) * 100
    return X_train, X_test, y_test, fitur, rf, svm, acc_rf, acc_svm

X_train, X_test, y_test, fitur_sensor, rf_model, svm_model, acc_rf, acc_svm = load_and_train()

label_kategori = {
    0: 'Baik',
    1: 'Berbahaya',
    2: 'Sangat Tidak Sehat',
    3: 'Sedang',
    4: 'Tidak Sehat'
}

status_info = {
    'Baik': ('#10b981', 'AMAN — Kualitas udara sangat baik, tidak menimbulkan risiko kesehatan.'),
    'Sedang': ('#3b82f6', 'MODERAT — Masih dapat diterima, waspada bagi kelompok sensitif.'),
    'Tidak Sehat': ('#f59e0b', 'PERINGATAN — Mulai berdampak pada kesehatan pernapasan masyarakat.'),
    'Sangat Tidak Sehat': ('#ef4444', 'BAHAYA TINGGI — Segera aktifkan sistem filtrasi udara & batasi aktivitas luar.'),
    'Berbahaya': ('#c084fc', 'DARURAT LINGKUNGAN — Kondisi kritis, evakuasi / proteksi pernapasan wajib.')
}

for i, col in enumerate(fitur_sensor):
    if f"sl_{i}" not in st.session_state:
        st.session_state[f"sl_{i}"] = float(X_train[col].median())

if "sample_info" not in st.session_state:
    st.session_state["sample_info"] = "Mode: Nilai Median Sensor (Manual)"

if "active_mode" not in st.session_state:
    st.session_state["active_mode"] = "manual"

# CSS Dinamis untuk Mewarnai Tombol Skenario yang Sedang Aktif
mode_colors = {
    "bersih": ("#059669", "#10b981", "rgba(16, 185, 129, 0.45)"),
    "polusi": ("#d97706", "#f59e0b", "rgba(245, 158, 11, 0.45)"),
    "darurat": ("#dc2626", "#ef4444", "rgba(239, 68, 68, 0.45)"),
    "acak": ("#7c3aed", "#a855f7", "rgba(168, 85, 247, 0.45)"),
    "konversi": ("#0284c7", "#38bdf8", "rgba(56, 189, 248, 0.45)"),
    "manual": ("#334155", "#64748b", "rgba(100, 116, 139, 0.3)")
}
bg_act, brd_act, shd_act = mode_colors.get(st.session_state["active_mode"], mode_colors["manual"])

st.markdown(f"""
<style>
    .stButton > button[kind="primary"] {{
        background-color: {bg_act} !important;
        color: #ffffff !important;
        border: 2px solid {brd_act} !important;
        font-weight: 800 !important;
        box-shadow: 0 0 14px {shd_act} !important;
    }}
</style>
""", unsafe_allow_html=True)

def set_preset(values_list, info_msg="", mode_key="manual"):
    for i, val in enumerate(values_list):
        min_v = float(X_train[fitur_sensor[i]].min())
        max_v = float(X_train[fitur_sensor[i]].max())
        st.session_state[f"sl_{i}"] = float(np.clip(val, min_v, max_v))
    st.session_state["sample_info"] = info_msg
    st.session_state["active_mode"] = mode_key

def load_random_test_sample():
    idx = np.random.randint(0, len(X_test))
    row_vals = X_test.iloc[idx].tolist()
    actual_lbl = label_kategori[int(y_test.iloc[idx])]
    set_preset(row_vals, f"📌 Sampel Data Uji #{idx + 1} | Label Aktual Dataset: **{actual_lbl.upper()}**", "acak")

def on_manual_slider():
    st.session_state["active_mode"] = "manual"
    st.session_state["sample_info"] = "🎛️ Mode: Penyesuaian Slider Manual"

# 4. Header Responsif
st.markdown(f"""
<div class="header-box">
    <span class="status-badge">● SYSTEM ONLINE</span>
    <div class="header-title">🏭 AIR QUALITY TELEMETRY & AI DIAGNOSTICS</div>
    <div class="header-sub">
        Industrial Monitoring System (Z-Score) &nbsp;|&nbsp; RF Accuracy: <b>{acc_rf:.2f}%</b> • SVM Accuracy: <b>{acc_svm:.2f}%</b>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Tombol Skenario Cepat (Otomatis menyala berbeda saat aktif)
st.markdown("**⚡ Pilih Skenario Simulasi Cepat:**")
active_m = st.session_state["active_mode"]
b1, b2, b3, b4 = st.columns(4)
with b1:
    st.button(
        "🟢 Udara Bersih",
        type="primary" if active_m == "bersih" else "secondary",
        use_container_width=True,
        on_click=set_preset,
        args=([-1.15, -1.22, -1.10, -0.90, -1.32, -0.44, 0.0, 0.0, 0.5, 0.5], "🟢 Aktif: Skenario Preset Udara Bersih", "bersih")
    )
with b2:
    st.button(
        "🟠 Polusi Tinggi",
        type="primary" if active_m == "polusi" else "secondary",
        use_container_width=True,
        on_click=set_preset,
        args=([0.80, 0.85, 0.50, 0.20, 0.60, 0.10, 0.5, -0.3, -0.5, -0.37], "🟠 Aktif: Skenario Preset Polusi Tinggi", "polusi")
    )
with b3:
    st.button(
        "🔴 Darurat Asap",
        type="primary" if active_m == "darurat" else "secondary",
        use_container_width=True,
        on_click=set_preset,
        args=([5.50, 5.30, 3.20, 0.30, -0.50, 0.05, 1.0, -0.5, -1.0, -0.37], "🔴 Aktif: Skenario Preset Darurat Asap", "darurat")
    )
with b4:
    st.button(
        "🎲 Uji Data Uji Acak",
        type="primary" if active_m == "acak" else "secondary",
        use_container_width=True,
        on_click=load_random_test_sample
    )

if st.session_state["sample_info"]:
    st.caption(st.session_state["sample_info"])

# 6. Kontrol 10 Sensor Manual
label_map = ['PM2.5', 'PM10', 'Gas CO', 'Gas SO₂', 'Gas NO₂', 'Gas O₃', 'Suhu', 'Kelembapan', 'Angin', 'Hujan']
input_vals = []

with st.expander("🎛️ Atur Manual Nilai 10 Sensor (Klik untuk Buka/Tutup)", expanded=False):
    col_s1, col_s2 = st.columns(2)
    for i, col in enumerate(fitur_sensor):
        target_col = col_s1 if i < 5 else col_s2
        with target_col:
            val = st.slider(
                f"{label_map[i]} (Z-score)",
                min_value=float(X_train[col].min()),
                max_value=float(X_train[col].max()),
                step=0.05,
                key=f"sl_{i}",
                on_change=on_manual_slider
            )
            input_vals.append(val)

# 7. Prediksi Real-Time
df_in = pd.DataFrame([input_vals], columns=fitur_sensor)
pred_rf = label_kategori[int(rf_model.predict(df_in)[0])]
pred_svm = label_kategori[int(svm_model.predict(df_in)[0])]
probs_rf = rf_model.predict_proba(df_in)[0] * 100
max_prob = probs_rf.max()

col_rf, desc_rf = status_info[pred_rf]
col_svm, _ = status_info[pred_svm]

# 8. Kartu Hasil Diagnosis
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {col_rf};">
        <div class="card-label">PRIMARY ENGINE: RANDOM FOREST</div>
        <div class="card-value" style="color:{col_rf} !important;">{pred_rf.upper()}</div>
        <div class="card-sub">Confidence Score: <b>{max_prob:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {col_svm};">
        <div class="card-label">SECONDARY ENGINE: SVM (RBF KERNEL)</div>
        <div class="card-value" style="color:{col_svm} !important;">{pred_svm.upper()}</div>
        <div class="card-sub">Margin Classifier Status: <b>Active</b></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="background-color:#0f172a; border:1px solid {col_rf}; border-radius:8px; padding:12px 16px; margin-bottom:14px; color:#f8fafc;">
    <b>Rekomendasi Sistem:</b> {desc_rf}
</div>
""", unsafe_allow_html=True)

# 9. Tab Fitur Utama & Tambahan (Tab aktif disorot biru terang)
tab1, tab2, tab3 = st.tabs(["📊 Grafik Telemetri Real-Time", "🧮 Konversi Satuan Fisik → Z-Score", "📥 Ekspor Log Data"])

with tab1:
    g1, g2 = st.columns([1.3, 1])
    plot_cfg = {'displayModeBar': False}

    with g1:
        colors_bar = ['#ef4444' if (i < 6 and v > 2.5) else '#f59e0b' if (i < 6 and v > 0.4) else '#38bdf8' if i < 6 else '#2dd4bf' for i, v in enumerate(input_vals)]
        fig_s = go.Figure(go.Bar(
            x=label_map, y=input_vals, marker_color=colors_bar,
            text=[f"{v:.1f}" for v in input_vals], textposition='outside'
        ))
        fig_s.update_layout(
            title=dict(text="Profil Telemetri 10 Sensor (Z-Score)", font=dict(size=14, color='#f8fafc')),
            paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font_color='#e2e8f0',
            height=310, margin=dict(l=25, r=15, t=45, b=35),
            yaxis=dict(range=[-3.8, 11.2], fixedrange=True),
            xaxis=dict(fixedrange=True)
        )
        st.plotly_chart(fig_s, use_container_width=True, config=plot_cfg)

    with g2:
        urutan = [0, 3, 4, 2, 1]
        nama_k = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat TS', 'Berbahaya']
        nilai_p = [probs_rf[k] for k in urutan]
        warna_k = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#c084fc']
        fig_p = go.Figure(go.Bar(
            x=nilai_p, y=nama_k, orientation='h', marker_color=warna_k,
            text=[f"{p:.1f}%" for p in nilai_p], textposition='outside'
        ))
        fig_p.update_layout(
            title=dict(text="Probabilitas Random Forest (%)", font=dict(size=14, color='#f8fafc')),
            paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font_color='#e2e8f0',
            height=310, margin=dict(l=75, r=35, t=45, b=35),
            xaxis=dict(range=[0, 122], fixedrange=True),
            yaxis=dict(autorange='reversed', fixedrange=True)
        )
        st.plotly_chart(fig_p, use_container_width=True, config=plot_cfg)

with tab2:
    st.markdown("**Simulasi Kalibrasi Sensor Fisik Lapangan ke Skala Z-Score**")
    st.caption("Masukkan pembacaan asli sensor (satuan fisik), sistem akan menghitung rumus Z = (x - μ) / σ dan mengirimkannya ke dashboard.")
    ref_stats = [
        ("PM2.5 (µg/m³)", 65.0, 45.0, 0.0, 500.0),
        ("PM10 (µg/m³)", 110.0, 70.0, 0.0, 600.0),
        ("Gas CO (µg/m³)", 1200.0, 800.0, 0.0, 10000.0),
        ("Gas SO₂ (µg/m³)", 45.0, 35.0, 0.0, 500.0),
        ("Gas NO₂ (µg/m³)", 55.0, 40.0, 0.0, 400.0),
        ("Gas O₃ (µg/m³)", 70.0, 45.0, 0.0, 400.0),
        ("Suhu (°C)", 28.0, 4.5, 10.0, 45.0),
        ("Kelembapan (%)", 72.0, 15.0, 20.0, 100.0),
        ("Kec. Angin (km/h)", 12.0, 6.0, 0.0, 60.0),
        ("Curah Hujan (mm)", 2.5, 6.5, 0.0, 100.0),
    ]
    rc1, rc2 = st.columns(2)
    z_converted = []
    for i, (r_lbl, r_mean, r_std, r_min, r_max) in enumerate(ref_stats):
        target_rc = rc1 if i < 5 else rc2
        with target_rc:
            phys_val = st.number_input(r_lbl, min_value=r_min, max_value=r_max, value=r_mean, key=f"phys_{i}")
            z_val = (phys_val - r_mean) / r_std
            z_converted.append(z_val)

    def apply_converted():
        set_preset(z_converted, "🔄 Aktif: Hasil Konversi Sensor Fisik Lapangan", "konversi")

    st.button(
        "🚀 Terapkan Hasil Konversi ke Diagnosis AI",
        type="primary" if active_m == "konversi" else "secondary",
        on_click=apply_converted
    )

with tab3:
    st.markdown("**Unduh Log Pembacaan Sensor & Hasil Diagnosis AI**")
    log_df = df_in.copy()
    log_df.insert(0, 'Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log_df['Prediksi_Random_Forest'] = pred_rf
    log_df['Confidence_RF_Pct'] = round(max_prob, 2)
    log_df['Prediksi_SVM'] = pred_svm

    st.dataframe(log_df, use_container_width=True)
    csv_bytes = log_df.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button(
        label="📥 Download Laporan Log (.CSV)",
        data=csv_bytes,
        file_name=f"Log_Kualitas_Udara_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
