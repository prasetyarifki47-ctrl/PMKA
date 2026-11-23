import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Air Quality Telemetry & AI Diagnostics",
    page_icon="🌫️",
    layout="wide"
)

# CSS Tema Gelap Industri
st.markdown("""
<style>
    .stApp { background-color: #0b1120; color: #f8fafc; }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 18px;
        border-left: 6px solid #38bdf8;
        margin-bottom: 10px;
    }
    .status-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache Pelatihan Model agar Website Cepat
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
    test_clean = test_df[fitur + ['Target_Encoded']].dropna()

    X_train, y_train = train_clean[fitur], train_clean['Target_Encoded'].astype(int)
    X_test, y_test = test_clean[fitur], test_clean['Target_Encoded'].astype(int)

    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    svm = SVC(kernel='rbf', C=1.0, random_state=42).fit(X_train, y_train)

    acc_rf = accuracy_score(y_test, rf.predict(X_test)) * 100
    acc_svm = accuracy_score(y_test, svm.predict(X_test)) * 100
    return X_train, fitur, rf, svm, acc_rf, acc_svm

X_train, fitur_sensor, rf_model, svm_model, acc_rf, acc_svm = load_and_train()

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

# Header
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown("## 🏭 AIR QUALITY TELEMETRY & AI DIAGNOSTICS")
    st.caption(f"Industrial Environmental Monitoring System (Z-Score Telemetry) | RF Accuracy: {acc_rf:.2f}% • SVM Accuracy: {acc_svm:.2f}%")
with col_h2:
    st.markdown("<div style='text-align:right; margin-top:15px;'><span class='status-badge'>● SYSTEM ONLINE</span></div>", unsafe_allow_html=True)

st.divider()

# Preset Skenario Cepat di Sidebar
st.sidebar.header("⚡ Skenario Simulasi Cepat")
if st.sidebar.button("🟢 Udara Bersih", use_container_width=True):
    st.session_state.vals = [-1.15, -1.22, -1.10, -0.90, -1.32, -0.44, 0.0, 0.0, 0.5, 0.5]
if st.sidebar.button("🟠 Polusi Tinggi", use_container_width=True):
    st.session_state.vals = [0.80, 0.85, 0.50, 0.20, 0.60, 0.10, 0.5, -0.3, -0.5, -0.37]
if st.sidebar.button("🔴 Darurat Asap", use_container_width=True):
    st.session_state.vals = [5.50, 5.30, 3.20, 0.30, -0.50, 0.05, 1.0, -0.5, -1.0, -0.37]

if 'vals' not in st.session_state:
    st.session_state.vals = [float(X_train[c].median()) for c in fitur_sensor]

st.sidebar.divider()
st.sidebar.header("🎛️ Kontrol Input 10 Sensor")

label_map = ['PM2.5', 'PM10', 'Gas CO', 'Gas SO₂', 'Gas NO₂', 'Gas O₃', 'Suhu', 'Kelembapan', 'Angin', 'Hujan']
input_vals = []
for i, col in enumerate(fitur_sensor):
    val = st.sidebar.slider(
        f"{label_map[i]} (Z-score)",
        min_value=float(X_train[col].min()),
        max_value=float(X_train[col].max()),
        value=float(st.session_state.vals[i]),
        step=0.05
    )
    input_vals.append(val)

# Prediksi Real-Time
df_in = pd.DataFrame([input_vals], columns=fitur_sensor)
pred_rf = label_kategori[int(rf_model.predict(df_in)[0])]
pred_svm = label_kategori[int(svm_model.predict(df_in)[0])]
probs_rf = rf_model.predict_proba(df_in)[0] * 100
max_prob = probs_rf.max()

col_rf, desc_rf = status_info[pred_rf]
col_svm, _ = status_info[pred_svm]

# Kartu Hasil Diagnosis
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {col_rf};">
        <div style="font-size:12px; color:#94a3b8; font-weight:bold;">PRIMARY ENGINE: RANDOM FOREST</div>
        <div style="font-size:28px; color:{col_rf}; font-weight:800; margin:4px 0;">{pred_rf.upper()}</div>
        <div style="font-size:13px; color:#e2e8f0;">Confidence Score: <b>{max_prob:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {col_svm};">
        <div style="font-size:12px; color:#94a3b8; font-weight:bold;">SECONDARY ENGINE: SVM (RBF KERNEL)</div>
        <div style="font-size:28px; color:{col_svm}; font-weight:800; margin:4px 0;">{pred_svm.upper()}</div>
        <div style="font-size:13px; color:#e2e8f0;">Margin Classifier Status: <b>Active</b></div>
    </div>
    """, unsafe_allow_html=True)

st.info(f"**Rekomendasi Sistem:** {desc_rf}")

# Grafik Interaktif Plotly
g1, g2 = st.columns([1.4, 1])
with g1:
    colors_bar = ['#ef4444' if (i < 6 and v > 2.5) else '#f59e0b' if (i < 6 and v > 0.4) else '#38bdf8' if i < 6 else '#2dd4bf' for i, v in enumerate(input_vals)]
    fig_s = go.Figure(go.Bar(x=label_map, y=input_vals, marker_color=colors_bar, text=[f"{v:.2f}" for v in input_vals], textposition='outside'))
    fig_s.update_layout(title="Profil Telemetri 10 Sensor Real-Time (Z-Score)", paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font_color='#e2e8f0', height=340, yaxis=dict(range=[-3.8, 11]))
    st.plotly_chart(fig_s, use_container_width=True)

with g2:
    urutan = [0, 3, 4, 2, 1]
    nama_k = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    nilai_p = [probs_rf[k] for k in urutan]
    warna_k = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#c084fc']
    fig_p = go.Figure(go.Bar(x=nilai_p, y=nama_k, orientation='h', marker_color=warna_k, text=[f"{p:.1f}%" for p in nilai_p], textposition='outside'))
    fig_p.update_layout(title="Probabilitas Prediksi Random Forest (%)", paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font_color='#e2e8f0', height=340, xaxis=dict(range=[0, 118]), yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_p, use_container_width=True)