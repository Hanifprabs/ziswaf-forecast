import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os       
import pickle   

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_percentage_error 

# Optional seaborn
try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False

from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Dashboard ZISWAF + Makro", layout="wide")
# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stSidebar"] {
        background-color: #1e293b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# UPLOAD DATASET
# ===========================
st.sidebar.header("📂 Upload Dataset")

macro_file = st.sidebar.file_uploader("Upload Dataset Makro (Excel)", type=["xlsx"])
ziswaf_file = st.sidebar.file_uploader("Upload Dataset ZISWAF (Excel)", type=["xlsx"])

# ===========================
# LOAD DATA
# ===========================
@st.cache_data
def load_data(macro_file, ziswaf_file):
    if macro_file is None or ziswaf_file is None:
        return None

    df_macro = pd.read_excel(macro_file)
    df_ziswaf = pd.read_excel(ziswaf_file)

    # Normalisasi kolom
    df_macro.columns = df_macro.columns.str.strip().str.lower()
    df_ziswaf.columns = df_ziswaf.columns.str.strip().str.lower()

    # Normalisasi kolom tahun
    year_alias = ["tahun", "thn", "year", "yr"]
    for df in [df_macro, df_ziswaf]:
        for col in df.columns:
            if col in year_alias:
                df.rename(columns={col: "tahun"}, inplace=True)

    if "tahun" not in df_macro.columns:
        st.error("Dataset Makro tidak memiliki kolom 'Tahun'!")
        return None
    if "tahun" not in df_ziswaf.columns:
        st.error("Dataset ZISWAF tidak memiliki kolom 'Tahun'!")
        return None

    # Merge dataset
    df = pd.merge(df_macro, df_ziswaf, on="tahun", how="inner")

    return df

df = load_data(macro_file, ziswaf_file)

if df is None:
    st.warning("Silakan upload dua dataset untuk melanjutkan.")
    st.stop()

# ===========================
# DETEKSI KOLOM TARGET ZISWAF OTOMATIS
# ===========================
df.columns = df.columns.str.lower()

possible_target = ["ziswaf", "total_ziswaf", "total ziswaf", "total", "zakat"]
target_column = None

for col in df.columns:
    for key in possible_target:
        if key.replace(" ", "") in col.replace(" ", ""):
            target_column = col
            break
    if target_column:
        break

if target_column is None:
    st.error("❌ Kolom ZISWAF tidak ditemukan. Pastikan dataset punya kolom ZISWAF/Total ZISWAF.")
    st.stop()

st.sidebar.success(f"Kolom Target Terdeteksi: {target_column.upper()}")

# ===========================
# Siapkan data untuk model
# ===========================
X = df.drop(columns=["tahun", target_column], errors="ignore")
y = df[target_column]

# ===========================
# TRAIN MODEL & EVALUATION
# ===========================
# 1. Buat folder models jika belum ada
if not os.path.exists("models"):
    os.makedirs("models")

# 2. Split data: 80% untuk latihan, 20% untuk ujian (evaluasi)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training model menggunakan data latihan
model_rf = RandomForestRegressor(n_estimators=200, random_state=42)
model_rf.fit(X_train, y_train)

# 4. Evaluasi: Hitung seberapa akurat modelnya
y_pred_test = model_rf.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred_test)

# 5. Simpan model permanen ke folder models/
pickle.dump(model_rf, open("models/random_forest.pkl", "wb"))

# 6. Prediksi tahun berikutnya (menggunakan seluruh data untuk hasil akhir)
model_rf.fit(X, y) 
y_pred = model_rf.predict(X.iloc[-1:].values)

# ===========================
# SIDEBAR MENU
# ===========================
menu = st.sidebar.selectbox(
    "📌 Navigasi",
    ["Home", "Dataset", "Visualisasi Makro", "Forecasting", "DSS"]
)

# ===========================
# 1. HOME
# ===========================
if menu == "Home":
    st.title("📊 Dashboard Analisis Makro & Prediksi ZISWAF")
    st.write("""
        Dashboard ini menyediakan:
        - 📂 Visualisasi dan dataset indikator makro ekonomi  
        - 📉 Line chart & heatmap korelasi  
        - 🤖 Forecasting menggunakan Random Forest  
        - 🧠 Sistem Pendukung Keputusan (DSS)  
    """)

# ===========================
# 2. DATASET
# ===========================
elif menu == "Dataset":
    st.header("📂 Dataset Gabungan Makro + ZISWAF")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Dataset Gabungan",
                       csv,
                       "dataset_makro_ziswaf.csv",
                       "text/csv")

# ===========================
# 3. VISUALISASI
# ===========================
elif menu == "Visualisasi Makro":
    st.header("📈 Visualisasi Indikator Makro Ekonomi")

    # pilih variabel numeric
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Hapus kolom target dan kolom tahun dari dropdown
    numeric_cols = [
        col for col in numeric_cols 
        if col not in [target_column, "tahun"]
    ]

    st.subheader("📌 Trend Variabel")
    selected_var = st.selectbox("Pilih variabel:", numeric_cols)
    st.line_chart(df.set_index("tahun")[selected_var])

    st.subheader("🔗 Heatmap Korelasi")
    corr = df[numeric_cols + [target_column]].corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    if seaborn_available:
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    else:
        ax.matshow(corr)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticklabels(corr.columns)

    st.pyplot(fig)

# ===========================
# 4. FORECASTING
# ===========================
elif menu == "Forecasting":
    st.header("🤖 Forecasting Total ZISWAF (Random Forest)")

    st.subheader("📌 Fitur yang digunakan:")
    st.write(list(X.columns))

    st.subheader("🔮 Prediksi Tahun Berikutnya:")
    st.metric("Total ZISWAF (Prediksi)", f"{y_pred[0]:,.0f}")

    forecast_df = pd.DataFrame({"Prediksi ZISWAF": y_pred})
    csv_pred = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Prediksi",
                       csv_pred,
                       "prediksi_ziswaf.csv",
                       "text/csv")

# ===========================
# 5. DSS
# ===========================
elif menu == "DSS":
    st.header("🧠 Decision Support System (DSS) Cerdas")
    
    # --- 1. Ambil Data Terbaru ---
    # Kita asumsikan nama kolom di Excel kamu mengandung kata kunci ini
    def get_val(keywords):
        col = next((c for c in df.columns if any(k in c.lower() for k in keywords)), None)
        return df[col].iloc[-1] if col else None

    inflasi = get_val(["inflasi", "inflation"])
    bi_rate = get_val(["bi rate", "suku bunga", "interest"])
    kurs = get_val(["kurs", "exchange"])
    pengangguran = get_val(["pengangguran", "unemployment"])

    # --- 2. Risk Level Scoring Logic ---
    risk_score = 0
    if inflasi and inflasi > 4: risk_score += 1
    if bi_rate and bi_rate > 6: risk_score += 1
    if pengangguran and pengangguran > 5: risk_score += 1
    
    # Tentukan Level
    if risk_score >= 3:
        risk_status = "TINGGI 🔴"
        risk_color = "red"
    elif risk_score == 2:
        risk_status = "SEDANG 🟡"
        risk_color = "orange"
    else:
        risk_status = "RENDAH 🟢"
        risk_color = "green"

    # --- 3. Tampilan Dashboard DSS ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Status Ekonomi")
        st.markdown(f"### Risiko 2025: <span style='color:{risk_color}'>{risk_status}</span>", unsafe_allow_html=True)
        st.write(f"**Inflasi:** {inflasi}%" if inflasi else "")
        st.write(f"**BI Rate:** {bi_rate}%" if bi_rate else "")

    with col2:
        st.subheader("💡 Rekomendasi Kebijakan")
        
        # Rule 1: Inflasi
        if inflasi and inflasi > 4:
            st.error("🚨 **Inflasi Naik:** Dorong Zakat Konsumtif untuk menjaga daya beli mustahik.")
        
        # Rule 2: BI Rate
        if bi_rate and bi_rate < 5:
            st.success("📈 **BI Rate Turun:** Momentum tepat untuk percepat Wakaf Produktif (investasi murah).")
            
        # Rule 3: Pengangguran
        if pengangguran and pengangguran > 5:
            st.warning("💼 **Pengangguran Naik:** Segera buka program bantuan modal usaha/Zakat Produktif.")

    # --- 4. Rekomendasi Target ZISWAF ---
    st.divider()
    st.subheader("🎯 Target Optimal ZISWAF")
    target_optimal = y_pred[0] * 1.1 # Contoh: AI menyarankan naik 10% dari prediksi
    st.info(f"Berdasarkan forecast AI dan kondisi ekonomi, target penyaluran optimal adalah: **Rp {target_optimal:,.0f}**")