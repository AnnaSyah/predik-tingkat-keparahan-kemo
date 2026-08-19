import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import time
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Keparahan Kemoterapi",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card { background: linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e9ecef; text-align: center; }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #2b8a3e; }
    .metric-label { font-size: 1rem; color: #495057; font-weight: 500; }
    .badge-ringan { background-color: #d3f9d8; color: #2b8a3e; padding: 10px 20px; border-radius: 5px; font-weight: bold; display: inline-block; font-size: 1.2rem; }
    .badge-sedang { background-color: #fff3cd; color: #856404; padding: 10px 20px; border-radius: 5px; font-weight: bold; display: inline-block; font-size: 1.2rem; }
    .badge-berat { background-color: #f8d7da; color: #721c24; padding: 10px 20px; border-radius: 5px; font-weight: bold; display: inline-block; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    clean_path = os.path.join("data", "bersih", "dataset_clean.csv")
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
        # Sesuai instruksi: Isi nilai median jika ada NaN saat pembacaan di aplikasi
        fitur_numerik = ['usia_tahun', 'siklus_ke', 'hb', 'leukosit', 'neutrofil', 'trombosit', 'suhu']
        fitur_kategorik = ['jenis_kelamin', 'mual', 'muntah', 'fatigue', 'diare', 'konstipasi', 'mukositis', 'nyeri', 'dukungan_keluarga']
        
        for col in fitur_numerik:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in fitur_kategorik:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                
        df = df.dropna(subset=['target_severity'])
        return df
    return None

@st.cache_resource
def load_model():
    model_path = os.path.join("model_tersimpan", "model_terbaik.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2862/2862873.png", width=100)
    st.title("Sistem Cerdas Kemo")
    st.markdown("Aplikasi Pendukung Keputusan Klinis untuk memprediksi tingkat keparahan pasien anak yang menjalani kemoterapi.")
    
    st.divider()
    df = load_data()
    model_data = load_model()
    
    if df is not None:
        st.success(f"✅ Dataset termuat ({len(df)} pasien)")
    else:
        st.error("❌ Dataset belum tersedia")
        
    if model_data is not None:
        st.success(f"✅ Model termuat ({model_data['nama_model']})")
    else:
        st.error("❌ Model belum dilatih")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Analisis Data (EDA)", "⚙️ Komparasi Model", "🩺 Simulasi Pasien"])

with tab1:
    st.header("Exploratory Data Analysis")
    if df is not None:
        ringan = len(df[df['target_severity'] == 0])
        sedang = len(df[df['target_severity'] == 1])
        berat = len(df[df['target_severity'] == 2])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Pasien Valid</div></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #2b8a3e;">{ringan}</div><div class="metric-label">Keparahan Ringan</div></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #f59f00;">{sedang}</div><div class="metric-label">Keparahan Sedang</div></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #e03131;">{berat}</div><div class="metric-label">Keparahan Berat</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Distribusi Kelas Target")
            fig_bar = px.bar(
                x=['Ringan', 'Sedang', 'Berat'], y=[ringan, sedang, berat],
                color=['Ringan', 'Sedang', 'Berat'],
                color_discrete_map={'Ringan': '#40c057', 'Sedang': '#fab005', 'Berat': '#fa5252'},
                labels={'x': 'Tingkat Keparahan', 'y': 'Jumlah Pasien'}
            )
            fig_bar.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_chart2:
            st.subheader("Distribusi Nilai Laboratorium")
            lab_cols = ['hb', 'leukosit', 'neutrofil', 'trombosit', 'suhu']
            df_lab = df[lab_cols].melt(var_name='Parameter', value_name='Nilai')
            fig_box = px.box(df_lab, x='Parameter', y='Nilai', color='Parameter')
            fig_box.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, use_container_width=True)
            
        st.subheader("Rata-rata Tingkat Gejala CTCAE berdasarkan Keparahan")
        gejala_cols = ['mual', 'muntah', 'fatigue', 'diare', 'konstipasi', 'mukositis', 'nyeri']
        df_gejala = df.groupby('target_severity')[gejala_cols].mean().reset_index()
        df_gejala['target_severity'] = df_gejala['target_severity'].map({0: 'Ringan', 1: 'Sedang', 2: 'Berat'})
        df_gejala = df_gejala.set_index('target_severity')
        
        fig_heat = px.imshow(
            df_gejala.values, x=gejala_cols, y=df_gejala.index,
            color_continuous_scale='Reds', aspect="auto", text_auto='.2f'
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Silakan jalankan pembersihan data terlebih dahulu untuk melihat analisis.")

with tab2:
    st.header("Komparasi & Evaluasi Model")
    
    if st.button("🚀 Jalankan Validasi 5-Fold", type="primary"):
        with st.spinner("Menjalankan 5-Fold Cross Validation..."):
            import subprocess
            result = subprocess.run(['python', 'modul/mesin_validasi.py'], capture_output=True, text=True)
            st.code(result.stdout)
            if result.stderr:
                st.error(result.stderr)
            st.toast("Validasi 5-Fold berhasil diperbarui!", icon="✅")
            st.success("Evaluasi selesai dijalankan pada 100 data pasien.")
            time.sleep(2)
            load_model.clear()  # Membersihkan cache model lama di memori
            st.rerun()
            
    if model_data is not None and df is not None:
        st.subheader("Tabel Metrik Komparasi")
        evaluasi_df = pd.DataFrame(model_data['evaluasi'])
        st.dataframe(evaluasi_df, use_container_width=True)
        
        st.subheader("Visualisasi Evaluasi 5-Fold Cross Validation (Out-of-Fold)")
        
        fitur = model_data['fitur_numerik'] + model_data['fitur_kategorik']
        X = df[fitur].reset_index(drop=True)
        y = df['target_severity'].reset_index(drop=True)
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        from sklearn.utils.class_weight import compute_sample_weight
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        y_true_oof = np.zeros(len(X))
        y_pred_oof = np.zeros(len(X))
        y_proba_oof = np.zeros((len(X), 3))
        
        base_pipeline = clone(model_data['model'])
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_pipeline = clone(base_pipeline)
            
            if model_data['nama_model'] == 'XGBoost':
                s_weight = compute_sample_weight('balanced', y_train)
                fold_pipeline.fit(X_train, y_train, clf__sample_weight=s_weight)
            else:
                fold_pipeline.fit(X_train, y_train)
                
            y_pred_oof[val_idx] = fold_pipeline.predict(X_val)
            y_proba_oof[val_idx] = fold_pipeline.predict_proba(X_val)
            y_true_oof[val_idx] = y_val
            
        col_cm, col_roc = st.columns(2)
        
        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_true_oof, y_pred_oof)
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=['Ringan (Pred)', 'Sedang (Pred)', 'Berat (Pred)'],
                y=['Ringan (True)', 'Sedang (True)', 'Berat (True)'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_roc:
            st.markdown("**Kurva ROC Multiclass (OvR)**")
            y_test_bin = label_binarize(y_true_oof, classes=[0, 1, 2])
            n_classes = y_test_bin.shape[1]
            
            fig_roc = go.Figure()
            colors = ['#40c057', '#fab005', '#fa5252']
            class_names = ['Ringan', 'Sedang', 'Berat']
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_oof[:, i])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{class_names[i]} (AUC = {roc_auc:.2f})', line=dict(color=colors[i], width=2)))
                
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='gray'), showlegend=False))
            fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', plot_bgcolor='rgba(0,0,0,0)', hovermode='x')
            st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("Model belum dilatih. Klik tombol Jalankan Validasi di atas.")

with tab3:
    st.header("Simulasi Prediksi Pasien")
    if model_data is not None:
        with st.form("simulasi_form"):
            st.subheader("Data Demografi & Siklus")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1: usia_tahun = st.number_input("Usia (Tahun)", min_value=0.0, max_value=18.0, value=5.0, step=0.5)
            with col_d2: jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            with col_d3: siklus_ke = st.number_input("Siklus Ke-", min_value=1, max_value=20, value=1, step=1)
                
            st.subheader("Dukungan Keluarga")
            dukungan_keluarga = st.selectbox("Tingkat Dukungan", ["Tinggi", "Sedang", "Rendah"])
            
            st.subheader("Gejala Klinis (CTCAE v5.0)")
            st.markdown("Skala: 0 (Tidak ada) - 3 (Berat)")
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            with col_g1:
                mual = st.slider("Mual", 0, 3, 0)
                muntah = st.slider("Muntah", 0, 3, 0)
            with col_g2:
                fatigue = st.slider("Fatigue", 0, 3, 0)
                diare = st.slider("Diare", 0, 3, 0)
            with col_g3:
                konstipasi = st.slider("Konstipasi", 0, 3, 0)
                mukositis = st.slider("Mukositis", 0, 3, 0)
            with col_g4:
                nyeri = st.slider("Nyeri", 0, 3, 0)
                
            st.subheader("Hasil Laboratorium")
            col_l1, col_l2, col_l3, col_l4, col_l5 = st.columns(5)
            with col_l1: hb = st.number_input("Hemoglobin (g/dL)", value=11.0)
            with col_l2: leukosit = st.number_input("Leukosit (/uL)", value=8000.0)
            with col_l3: neutrofil = st.number_input("Neutrofil (%)", value=50.0)
            with col_l4: trombosit = st.number_input("Trombosit (/uL)", value=250000.0)
            with col_l5: suhu = st.number_input("Suhu (°C)", value=36.5)
                
            submitted = st.form_submit_button("Lakukan Prediksi Keparahan", type="primary", use_container_width=True)
            
        if submitted:
            jk_val = 0 if jenis_kelamin == "Laki-laki" else 1
            dukungan_val = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}[dukungan_keluarga]
            
            input_dict = {
                'usia_tahun': usia_tahun, 'siklus_ke': siklus_ke, 'hb': hb, 'leukosit': leukosit,
                'neutrofil': neutrofil, 'trombosit': trombosit, 'suhu': suhu, 'jenis_kelamin': jk_val,
                'mual': mual, 'muntah': muntah, 'fatigue': fatigue, 'diare': diare,
                'konstipasi': konstipasi, 'mukositis': mukositis, 'nyeri': nyeri, 'dukungan_keluarga': dukungan_val
            }
            
            fitur = model_data['fitur_numerik'] + model_data['fitur_kategorik']
            input_df = pd.DataFrame([input_dict])[fitur]
            
            model = model_data['model']
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            
            st.divider()
            st.subheader("Hasil Prediksi")
            
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                if pred == 0:
                    st.markdown('<div class="badge-ringan">Risiko: RINGAN</div>', unsafe_allow_html=True)
                    rek = "Pemantauan rutin rawat jalan. Lanjutkan edukasi gizi dan kebersihan."
                elif pred == 1:
                    st.markdown('<div class="badge-sedang">Risiko: SEDANG</div>', unsafe_allow_html=True)
                    rek = "Intervensi farmakologis untuk gejala. Pemantauan ketat tanda infeksi dan rehidrasi."
                else:
                    st.markdown('<div class="badge-berat">Risiko: BERAT</div>', unsafe_allow_html=True)
                    rek = "Pertimbangkan rawat inap darurat. Intervensi suportif agresif (transfusi/antibiotik empiris)."
                
                st.info(f"**Rekomendasi Keperawatan Preventif:**\n\n{rek}")
                st.caption("⚠️ Disclaimer: Hasil ini adalah alat bantu komputasi dan tidak menggantikan asesmen klinis tenaga medis profesional.")
                
            with col_r2:
                fig_prob = px.bar(
                    x=['Ringan', 'Sedang', 'Berat'], y=proba * 100, text=[f"{p*100:.1f}%" for p in proba],
                    color=['Ringan', 'Sedang', 'Berat'], color_discrete_map={'Ringan': '#40c057', 'Sedang': '#fab005', 'Berat': '#fa5252'},
                    labels={'x': 'Tingkat Keparahan', 'y': 'Probabilitas (%)'}, title='Probabilitas Prediksi per Kelas'
                )
                fig_prob.update_traces(textposition='outside')
                fig_prob.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', yaxis_range=[0, 110])
                st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("Model belum dilatih. Harap latih model di Tab Komparasi terlebih dahulu.")
