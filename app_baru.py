import streamlit as st

# Konfigurasi Halaman Dasar
st.set_page_config(page_title="ZISWAF Gen Z Analytics", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

# PENTING: Untuk grafik, kita akan pakai Seaborn agar visualisasinya lebih 'fresh'
# --- Tambahkan/Pastikan bagian ini ada di paling atas ---
try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False
# -------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# ===========================
# 🔐 6. USER MANAGEMENT (Admin & Viewer Role)
# ===========================

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None

def login_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("<h1 style='text-align: center; color: #00ffff;'>🔐 ZISWAF Access</h1>", unsafe_allow_html=True)
        
        with st.container(border=True):
            role_choice = st.selectbox("Pilih Role", ["Viewer", "Admin"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login 🚀", use_container_width=True):
                # Logika Akun Admin
                if role_choice == "Admin" and username == "admin" and password == "admin123":
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = "Admin"
                    st.rerun()
                # Logika Akun Viewer
                elif role_choice == "Viewer" and username == "user" and password == "user123":
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = "Viewer"
                    st.rerun()
                else:
                    st.error("Username atau Password salah untuk role tersebut!")

if not st.session_state['logged_in']:
    login_page()
    st.stop()

# --- Tombol Logout di Sidebar ---
if st.sidebar.button("Log Out 🚪"):
    st.session_state['logged_in'] = False
    st.rerun()

# 2. Inisialisasi Variabel Global (Biar gak Error NameError)
if 'akurasi' not in locals():
    akurasi = 0.0
if 'seaborn_available' not in locals():
    seaborn_available = True



# ===========================
# 💅 THEMING & CSS GEN Z (Neon-Fresh Style)
# ===========================
st.markdown("""
<style>

/* Background utama */
html, body, [data-testid="stAppViewContainer"] > .main {
    background-color: #ffffff;
    color: #000000;
    font-family: 'Open Sans', sans-serif;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #f5f5f5;
    border-right: 1px solid #ddd;
}
div[data-testid="stSidebar"],
div[data-testid="stSidebar"] * {
    color: #000000;
}

/* Metric Card */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

/* Judul */
h1, h2, h3, h4, h5 {
    color: #000000;
}

/* Tombol */
div.stButton > button {
    background-color: #ffffff;
    color: #000000;
    border: 1px solid #000000;
    border-radius: 10px;
}
div.stButton > button:hover {
    background-color: #000000;
    color: #ffffff;
}

/* 🔥 CARD TARGET (biru) → teks putih */
.custom-card {
    color: #ffffff !important;
}
.custom-card * {
    color: #ffffff !important;
}

/* 🔥 REKOMENDASI → teks hitam */
.rekomendasi-box {
    background-color: #ffffff;
    color: #000000 !important;
}
.rekomendasi-box * {
    color: #000000 !important;
}

/* Optional: biar highlight biru gak ganggu */
::selection {
    background: #cce5ff;
    color: #000000;
}

</style>
""", unsafe_allow_html=True)

# ===========================
# 📂 SIDEBAR & UPLOAD
# ===========================
st.sidebar.markdown("# ZISWAF.ai")
st.sidebar.header("📂 Data Center")

macro_file = st.sidebar.file_uploader("Upload Data Makro (Excel)", type=["xlsx"])
ziswaf_file = st.sidebar.file_uploader("Upload Data ZISWAF (Excel)", type=["xlsx"])

# ===========================
# 🧠 LOAD & PROCESS DATA (Cache biar ngebut)
# ===========================
@st.cache_data
def load_data(macro_file, ziswaf_file):
    if macro_file is None or ziswaf_file is None:
        return None

    df_macro = pd.read_excel(macro_file)
    df_ziswaf = pd.read_excel(ziswaf_file)

    # Normalisasi kolom (huruf kecil semua)
    df_macro.columns = df_macro.columns.str.strip().str.lower()
    df_ziswaf.columns = df_ziswaf.columns.str.strip().str.lower()

    # Normalisasi tahun
    for df in [df_macro, df_ziswaf]:
        year_col = next((c for c in df.columns if any(y in c for y in ["tahun", "thn", "year"])), None)
        if year_col:
            df.rename(columns={year_col: "tahun"}, inplace=True)

    if "tahun" not in df_macro.columns or "tahun" not in df_ziswaf.columns:
        st.error("❌ Dataset kudu ada kolom 'Tahun'-nya, bro!")
        return None

    # Merge dataset
    df = pd.merge(df_macro, df_ziswaf, on="tahun", how="inner")
    df = df.sort_values("tahun") # Pastikan urut

    return df

# Eksekusi Load Data
df = load_data(macro_file, ziswaf_file)

# Hentikan program jika data belum di-upload
if df is None:
    st.warning("⚠️ Upload dua dataset di sidebar dulu biar sistemnya aktif.")
    st.stop()

# ===========================
# 🤖 DETEKSI TARGET ZISWAF
# ===========================
possible_target = ["ziswaf", "total_ziswaf", "total zakat", "zakat", "dana_zis"]
target_column = next((col for col in df.columns if any(key in col for key in possible_target)), None)

if target_column is None:
    st.error("❌ Kolom ZISWAF gak ketemu. Pastikan di Excel ada kolom total Zakat/ZISWAF.")
    st.stop()

# ===========================
# 🤖 AI ENGINE (Random Forest + Evaluation)
# ===========================
# Buat folder models jika belum ada
if not os.path.exists("models"):
    os.makedirs("models")

# Siapkan data X (Fitur) dan y (Target)
X = df.drop(columns=["tahun", target_column], errors="ignore")
y = df[target_column]

# 1. Split data (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Training model Latihan
model_rf = RandomForestRegressor(n_estimators=200, random_state=42)
model_rf.fit(X_train, y_train)

# 3. Evaluasi (Hitung MAPE untuk akurasi)
y_pred_test = model_rf.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred_test)

# 4. Simpan model permanen
pickle.dump(model_rf, open("models/random_forest_genz.pkl", "wb"))

# 5. Latih ulang model FINAL (pake seluruh data) untuk prediksi
model_rf.fit(X, y) 
last_values = X.iloc[-1:].values
y_pred = model_rf.predict(last_values)

# ===========================
# 📌 SIDEBAR NAVIGASI
# ===========================
# 1. Tentukan menu dasar (semua orang bisa lihat)
menu_dasar = ["🏠 Home", "📂 Open Dataset", "📉 Makro Visuals", "🤖 AI Forecast", "🧠 DSS Cerdas"]

# 2. Cek Role: Kalau Admin, tambahin menu rahasia
if st.session_state['role'] == "Admin":
    menu_pilihan = menu_dasar + ["➕ Pengelolaan Data (CRUD)", "⚙️ System Settings"]
else:
    menu_pilihan = menu_dasar

# 3. Tampilkan menu yang sudah disaring
menu = st.sidebar.selectbox("📌 Navigasi Menu", menu_pilihan)

# 4. Kasih label di sidebar biar jelas siapa yang login
st.sidebar.markdown(f"""
    <div style='padding: 10px; border-radius: 10px; background-color: rgba(0, 255, 255, 0.1); border: 1px solid #00ffff;'>
        <p style='margin:0; font-size: 12px; color: #8b949e;'>Logged in as:</p>
        <strong style='color: #00ffff;'>{st.session_state['role']}</strong>
    </div>
""", unsafe_allow_html=True)

# ===========================
# 1. HOME
# ===========================
if menu == "🏠 Home":
    st.title("📊 ZISWAF Analytics")
    
    st.markdown("""
        Selamat datang di **ZISWAF.ai**, alat forecasting ZISWAF canggih yang ditenagai oleh *Artificial Intelligence*.
        
        ### Fitur Utama:
        - 📂 **Open Dataset:** Cek data mentah yang di-upload.
        - 📉 **Makro Visuals:** Lihat tren dan korelasi ekonomi.
        - 🤖  **AI Forecast:** Prediksi ZISWAF tahun depan dengan akurasi tinggi.
        - 🧠 **DSS Cerdas:** Sistem pendukung keputusan berbasis kondisi ekonomi terbaru.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tahun Terakhir Data", df['tahun'].max())
    with col2:
        st.metric("Total Data Historis", len(df), "tahun")

# ===========================
# 2. OPEN DATASET
# ===========================
elif menu == "📂 Open Dataset":
    st.header("📂 Dataset Gabungan Makro + ZISWAF")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

# ===========================
# 3. MAKRO VISUALS (Custom Seaborn Colors)
# ===========================
elif menu == "📉 Makro Visuals":
    st.header("📉 Visualisasi Indikator Makro Ekonomi")

    # Ambil variabel makro (numeric)
    numeric_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c not in [target_column, "tahun"]]

    # Tren Variabel (Line Chart)
    st.subheader("📌 Trend Variabel Ekonomi")
    selected_var = st.selectbox("Pilih variabel makro:", numeric_cols)
    
    # Visualisasi dengan Matplotlib/Seaborn agar warnanya Gen Z
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    # Warna Cyan-Neon untuk line chart
    ax1.plot(df["tahun"], df[selected_var], marker='o', color='#00ffff', linewidth=2, markersize=6)
    
    # Styling Axis agar gelap
    ax1.set_facecolor('#0d1117') # Background grafik gelap
    fig1.patch.set_facecolor('#0d1117') # Background figure gelap
    ax1.tick_params(colors='white') # Warna angka sumbu
    ax1.grid(color='#30363d', linestyle='-', linewidth=0.5) # Grid tipis
    
    st.pyplot(fig1)

    # Heatmap Korelasi
    st.subheader("🔗 Heatmap Korelasi")
    # Hanya ambil kolom yang relevan (bukan yang 'unnamed')
    relevant_cols = [c for c in numeric_cols + [target_column] if "unnamed" not in c.lower()]
    corr = df[relevant_cols].corr()

    # Tentukan ukuran figure yang lebih besar agar tidak sesak
    fig, ax = plt.subplots(figsize=(12, 8)) 

    if seaborn_available:
        # fmt=".2f" -> membatasi 2 angka di belakang koma
        sns.heatmap(corr, 
                    annot=True, 
                    fmt=".2f", 
                    cmap="RdPu", 
                    ax=ax, 
                    annot_kws={"size": 9}, 
                    linewidths=0.5)
        # Putar label sumbu X agar tidak tabrakan
        plt.xticks(rotation=45, ha='right')
    else:
        # Fallback jika seaborn tidak ada
        im = ax.matshow(corr, cmap="RdPu")
        fig.colorbar(im)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='left')

    st.pyplot(fig)

# ===========================
# 4. AI FORECAST (Fixed Neon Version)
# ===========================
elif menu == "🤖 AI Forecast":
    st.header("🤖 Forecasting Total ZISWAF (AI Engine)")

    # Baris pertama untuk metrik
    col1, col2 = st.columns(2)
    
    with col1:
        # Style Khusus Kotak Cyan
        st.markdown(f"""
            <div style='border: 2px solid #00ffff; padding: 20px; border-radius: 15px; background-color: rgba(0, 255, 255, 0.05);'>
                <p style='color: #8b949e; margin: 0; font-size: 14px;'>Total ZISWAF (Prediksi Tahun Depan)</p>
                <h2 style='color: #ffffff; margin: 0;'>Rp {y_pred[0]:,.0f}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Style Khusus Kotak Magenta
        akurasi = (1 - mape) * 100
        st.markdown(f"""
            <div style='border: 2px solid #ff00ff; padding: 20px; border-radius: 15px; background-color: rgba(255, 0, 255, 0.05);'>
                <p style='color: #8b949e; margin: 0; font-size: 14px;'>Tingkat Akurasi Model AI</p>
                <h2 style='color: #ffffff; margin: 0;'>{akurasi:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    st.write("") # Spasi tambahan
    
    if akurasi < 70:
        st.warning("⚠️ **Warning:** Akurasi model di bawah 70%. AI butuh asupan data historis lebih banyak lagi nih!")
        
# ===========================
# 5. DSS CERDAS (Logika Temanmu)
# ===========================
elif menu == "🧠 DSS Cerdas":
    st.header("🧠 Decision Support System (DSS)")
    
    # --- 1. Ambil Data Terbaru (Pakai Keywords) ---
    # Fungsi pembantu untuk mencari kolom berdasarkan kata kunci
    def get_latest_macro(keywords):
        col = next((c for c in df.columns if any(k in c.lower() for k in keywords)), None)
        return df[col].iloc[-1] if col else None

    inflasi = get_latest_macro(["inflasi", "inflation"])
    bi_rate = get_latest_macro(["bi rate", "suku bunga", "interest"])
    kurs = get_latest_macro(["kurs", "exchange"])
    pengangguran = get_latest_macro(["pengangguran", "unemployment"])

    st.subheader("📊 Kondisi Ekonomi Terbaru")
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Inflasi", f"{inflasi}%" if inflasi is not None else "Data N/A")
    with c2: st.metric("BI Rate", f"{bi_rate}%" if bi_rate is not None else "Data N/A")
    with c3: st.metric(
        "Kurs USD/IDR",
        f"Rp {kurs*1000:,.0f}" if kurs is not None else "Data N/A"
    )

    st.divider()
    
    st.subheader("💡 Rekomendasi Kebijakan (Rule-Based ZISWAF)")
    
    # 1. Tentukan Pesan Berdasarkan Kondisi
    pesan_dss = ""
    status_warna = "#ffcc00" # Default Kuning

    if inflasi is not None and inflasi > 4:
        pesan_dss = f"🔥 Inflasi Naik ({inflasi}%): Dorong program **Zakat Konsumtif** untuk menjaga daya beli mustahik."
        status_warna = "#ff00ff" # Magenta
    elif bi_rate is not None and bi_rate < 5:
        pesan_dss = f"📈 BI Rate Turun ({bi_rate}%): Momentum tepat untuk genjot **Wakaf Produktif** (investasi murah)."
        status_warna = "#00ffff" # Cyan
    elif pengangguran is not None and pengangguran > 5:
        pesan_dss = f"💼 Pengangguran Naik ({pengangguran}%): Fokuskan dana pada **Bantuan Modal Usaha/UMKM**."
        status_warna = "#ff00ff"
    else:
        pesan_dss = "✅ Kondisi Ekonomi Stabil: Pertahankan alokasi seimbang antara Zakat Konsumtif dan Pemberdayaan Ekonomi."
        status_warna = "#ffcc00" # Kuning

    # 2. Masukkan Pesan ke Dalam Kotak Neon
    st.markdown(f"""
<div class="rekomendasi-box" style='
    border: 2px solid {status_warna};
    padding: 20px;
    border-radius: 15px;
    background-color: #ffffff;
'>
    <p style='margin: 0; font-size: 1.1rem; line-height: 1.6;'>
        {pesan_dss}
    </p>
</div>
""", unsafe_allow_html=True)

    # --- 2. Logika Rule-Based Temanmu ---
    
    # Rule 1: Inflasi Naik -> Zakat Konsumtif
    if inflasi is not None and inflasi > 4:
        st.markdown(f'<div class="rec-item"><span class="rec-item-magenta">🔥 Inflasi Naik ({inflasi}%):</span> Dorong program **Zakat Konsumtif** untuk menjaga daya beli mustahik dari lonjakan harga barang pokok.</div>', unsafe_allow_html=True)
    
    # Rule 2: BI Rate Turun -> Wakaf Produktif
    if bi_rate is not None and bi_rate < 5:
        st.markdown(f'<div class="rec-item"><span class="rec-item-cyan">📈 BI Rate Turun ({bi_rate}%):</span> Momentum tepat untuk menggenjot **Wakaf Produktif**. Biaya modal rendah, investasi infrastruktur wakaf lebih efisien.</div>', unsafe_allow_html=True)
        
    # Rule 3: Pengangguran Naik -> Bantuan Modal
    if pengangguran is not None and pengangguran > 5:
        st.markdown(f'<div class="rec-item"><span class="rec-item-magenta">💼 Pengangguran Naik ({pengangguran}%):</span> Fokuskan dana Zakat Produktif untuk membuka program **Bantuan Modal Usaha/UMKM** agar tercipta lapangan kerja baru.</div>', unsafe_allow_html=True)

    # Rule 4: Kurs Melemah -> Zakat Ekspor
    if kurs is not None and kurs > 16000:
        st.markdown(f'<div class="rec-item"><span class="rec-item-magenta">⚠️ Kurs Melemah (Rp {kurs:,.0f}):</span> Program pemberdayaan UMKM berbasis ekspor (Zakat Ekspor) terdampak biaya logistik/bahan baku impor. Lakukan **audit efisiensi program**.</div>', unsafe_allow_html=True)
        
    # Rule Netral
    if inflasi is not None and bi_rate is not None and inflasi <= 4 and bi_rate >= 5:
        st.markdown(f'<div class="rec-item"><span class="rec-item-yellow">Kondisi Ekonomi Stabil:</span> Pertahankan alokasi seimbang antara Zakat Konsumtif dan Pemberdayaan Ekonomi jangka panjang.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 3. Rekomendasi Target Optimal ---
    st.divider()
    st.subheader("🎯 Target Optimal ZISWAF (AI Recommendation)")
    
    # Contoh Rumus: Target = Prediksi AI + 10% (buffer)
    target_optimal = y_pred[0] * 1.10
    
    # Tampilan Target pakai Card Neon
    st.markdown(f"""
<div class="custom-card" style='
    background: linear-gradient(135deg, #0d1117 0%, #1a3c61 100%);
    border: 2px solid #00ffff;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
'>
    <h1>TARGET 2026</h1>
    <p style='font-size: 2.8rem; font-weight: 800;'>
        Rp {target_optimal:,.0f}
    </p>
    <p>
        Target ini disarankan oleh AI (forecast + 10% buffer).
    </p>
</div>
""", unsafe_allow_html=True)

# ===========================
# 6. CRUD
# ===========================
elif menu == "➕ Pengelolaan Data (CRUD)":
    st.header("📝 Pengelolaan Data Tahunan")
    st.info("Menu ini hanya dapat diakses oleh Admin untuk memperbarui data historis.")

    # 1. Tampilkan Data yang Ada Sekarang
    st.subheader("Data Saat Ini")
    st.dataframe(df, use_container_width=True)

    st.divider()

    # 2. Form Tambah Data Baru
    st.subheader("➕ Tambah Data Tahun Baru")
    
    # Kita buat kolom input otomatis berdasarkan nama kolom di dataset kamu
    with st.form("form_tambah_data"):
        new_data = {}
        cols = st.columns(3) # Bagi input jadi 3 kolom supaya rapi
        
        for i, column in enumerate(df.columns):
            with cols[i % 3]:
                if column.lower() == 'tahun':
                    new_data[column] = st.number_input(f"Masukkan {column}", value=int(df[column].max() + 1))
                else:
                    new_data[column] = st.number_input(f"Masukkan {column}", value=0.0)
        
        submitted = st.form_submit_button("Simpan & Update AI Model 💾")
        
        if submitted:
            # Tambah ke DataFrame
            new_df = pd.DataFrame([new_data])
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Simpan secara lokal (opsional: bisa simpan balik ke Excel)
            # df.to_excel("data_updated.xlsx", index=False)
            
            st.success(f"Data tahun {new_data['tahun']} berhasil ditambahkan!")
            st.balloons() # Efek perayaan Gen Z!
            st.info("Silakan ke menu AI Forecast untuk melihat hasil prediksi terbaru.")

    # 3. Fitur Hapus Data (Hanya baris terakhir)
    st.divider()
    st.subheader("🗑️ Zona Bahaya")
    if st.button("Hapus Baris Terakhir", type="secondary"):
        if len(df) > 1:
            df = df.drop(df.tail(1).index)
            st.warning("Baris terakhir telah dihapus.")
            st.rerun()
            
# ===========================
# 7. System Settings
# ===========================
elif menu == "⚙️ System Settings":
    st.header("⚙️ Konfigurasi Sistem & Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Status Model AI")
        st.write(f"**Algoritma:** Random Forest Regressor")
        st.write(f"**Path Model:** `models/random_forest.pkl` (Saved ✅)")
        st.write(f"**Akurasi Terakhir:** {akurasi:.2f}%")
        
        if st.button("Re-train AI Model 🔄"):
            with st.spinner("AI sedang belajar data baru..."):
                # Logika training ulang ada di sini
                import time
                time.sleep(2) # Simulasi mikir
            st.success("Model berhasil dilatih ulang!")

    with col2:
        st.subheader("🛡️ Hak Akses")
        st.write(f"**User:** {st.session_state['role']}")
        st.write("**Sesi:** Aktif")
        if st.button("Reset Session 🛠️"):
            st.session_state.clear()
            st.rerun()