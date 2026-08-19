# Developer Technical Specification & Implementation Guide
## Chemo Severity Prediction Engine — ML Pipeline + Streamlit App

> **Scope:** Dokumen ini adalah blueprint teknis murni untuk implementasi kode. Tidak ada narasi administratif/akademis — langsung ke arsitektur, skema data, spesifikasi pipeline ML, dan komponen UI.

---

## 1. System Architecture & Tech Stack

### 1.1 Environment

| Item | Spesifikasi |
|---|---|
| Python | 3.10+ (rekomendasi 3.11) |
| Environment manager | `venv` (bawaan) atau `conda` |
| OS target dev | Cross-platform (Windows/Linux/Mac) — hindari path hardcoded, gunakan `pathlib` |
| IDE | VS Code (`.vscode/settings.json` untuk interpreter path) |

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 1.2 `requirements.txt`

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
xgboost==2.0.3
imbalanced-learn==0.12.2
streamlit==1.35.0
plotly==5.22.0
joblib==1.4.2
openpyxl==3.1.2
scipy==1.13.0
```

### 1.3 High-Level Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  data/raw/       │ --> │  src/preprocess  │ --> │  data/processed/   │
│  Master Tabel-2   │     │  .py              │     │  dataset_clean.csv │
└─────────────────┘     └──────────────────┘     └───────────────────┘
                                                            │
                                                            v
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  models/         │ <-- │  src/train.py    │ <-- │  src/cv_pipeline.py│
│  best_model.pkl  │     │  (5-Fold loop)    │     │  (sklearn Pipeline)│
└─────────────────┘     └──────────────────┘     └───────────────────┘
        │
        v
┌─────────────────────────────────────────────────────────┐
│  app.py (Streamlit)                                        │
│  Tab1: EDA | Tab2: Model Benchmarking | Tab3: Predictor     │
└─────────────────────────────────────────────────────────┘
```

### 1.4 State Management Architecture (`st.session_state`)

Semua hasil komputasi berat (training, metrik) **wajib** disimpan di `session_state` agar tidak retrain setiap kali user pindah tab / interact dengan widget lain (Streamlit re-run script dari atas setiap event).

| Key | Tipe | Diisi oleh | Dipakai oleh |
|---|---|---|---|
| `st.session_state.df_clean` | `pd.DataFrame` | Load awal (cached) | Tab 1 (EDA) |
| `st.session_state.cv_results` | `dict[str, list[dict]]` | Tab 2 tombol "Run CV" | Tab 2 (tabel & grafik) |
| `st.session_state.agg_confusion` | `dict[str, np.ndarray]` | Tab 2 | Tab 2 (heatmap) |
| `st.session_state.roc_data` | `dict[str, dict]` | Tab 2 | Tab 2 (ROC curve) |
| `st.session_state.best_model_name` | `str` | Tab 2 (auto: F1-Macro tertinggi) | Tab 3 (default selectbox) |
| `st.session_state.trained_pipelines` | `dict[str, Pipeline]` | Tab 2 (fit ulang di **full data** setelah CV selesai, untuk dipakai live predict) | Tab 3 |
| `st.session_state.cv_done` | `bool` | Tab 2 | Guard render Tab 2/3 |

```python
# Inisialisasi wajib di awal app.py
if "cv_done" not in st.session_state:
    st.session_state.cv_done = False
if "df_clean" not in st.session_state:
    st.session_state.df_clean = load_and_clean_data()  # cached via @st.cache_data
```

Gunakan `@st.cache_data` untuk `load_and_clean_data()` dan `@st.cache_resource` untuk objek model/pipeline agar tidak reload file Excel tiap rerun.

---

## 2. Concrete Data Schema & Preprocessing Specification

### 2.1 Mapping Kolom Raw (`Master Tabel-2.xlsx`) → DataFrame Bersih

> Sesuaikan nama kolom raw persis dengan header asli file Excel saat implementasi — kolom di bawah adalah asumsi kerja berdasarkan spesifikasi fitur yang diberikan. Cek `df.columns.tolist()` di awal notebook eksplorasi dan update mapping ini.

| Kolom Raw (asumsi) | Kolom Clean (output) | Tipe Akhir | Transformasi |
|---|---|---|---|
| `Usia` | `usia_bulan` | `int` | Regex parsing (lihat 2.2) |
| `Usia` | `usia_tahun` | `float` | `usia_bulan / 12`, round 1 desimal |
| `Jenis Kelamin` | `jenis_kelamin` | `int` (0/1) | L=0, P=1 |
| `Diagnosis Kanker` | `diagnosis_kanker` | `category` → one-hot | Grouping kategori langka jadi `"Lainnya"` (freq < 5) |
| `Lama Terdiagnosis` | `lama_terdiagnosis_bulan` | `int` | Parse ke satuan bulan (sama pola dgn usia) |
| `Siklus Kemoterapi` | `siklus_ke` | `int` | Regex `\d+` (lihat 2.3) |
| `Protokol Kemoterapi` | `protokol_kemoterapi` | `category` → one-hot | - |
| `Riwayat Rawat Inap` | `riwayat_rawat_inap` | `int` (0/1) | Ya=1, Tidak=0 |
| `Mual` | `mual` | `int` (0-3) | Mapping dict (2.4) |
| `Muntah` | `muntah` | `int` (0-3) | Mapping dict (2.4) |
| `Fatigue/Kelelahan` | `fatigue` | `int` (0-3) | Mapping dict (2.4) |
| `Diare` | `diare` | `int` (0-3) | Mapping dict (2.4) |
| `Konstipasi` | `konstipasi` | `int` (0-3) | Mapping dict (2.4) |
| `Luka Mulut/Mukositis` | `mukositis` | `int` (0-3) | Mapping dict (2.4) |
| `Skala Nyeri` | `nyeri` | `int` (0-3) | Mapping dict (2.4) |
| `Dukungan Keluarga` | `dukungan_keluarga` | `int` (0-2) | Rendah=0, Sedang=1, Tinggi=2 |
| `Hemoglobin` | `hb` | `float` | StandardScaler (fit di training fold) |
| `Leukosit` | `leukosit` | `float` | StandardScaler |
| `Neutrofil` | `neutrofil` | `float` | StandardScaler |
| `Trombosit` | `trombosit` | `float` | StandardScaler |
| `Suhu Tubuh` | `suhu` | `float` | StandardScaler |
| `Tingkat Keparahan` | `TARGET_SEVERITY` | `int` (0/1/2) | Ringan=0, Sedang=1, Berat=2 |

### 2.2 Regex Parsing — Usia

```python
import re

def parse_usia_ke_bulan(text: str) -> int:
    """
    '4 bulan'        -> 4
    '2 tahun 3 bulan' -> 27
    '16 tahun'        -> 192
    """
    text = str(text).lower().strip()
    tahun = re.search(r'(\d+)\s*tahun', text)
    bulan = re.search(r'(\d+)\s*bulan', text)
    total = 0
    if tahun:
        total += int(tahun.group(1)) * 12
    if bulan:
        total += int(bulan.group(1))
    if not tahun and not bulan:
        # fallback: angka polos dianggap tahun
        angka = re.search(r'\d+', text)
        if angka:
            total = int(angka.group()) * 12
        else:
            raise ValueError(f"Format usia tidak dikenali: {text}")
    return total
```

### 2.3 Regex Parsing — Siklus Kemoterapi

```python
def parse_siklus(text) -> int:
    """
    'ke 2'   -> 2
    '10 kali' -> 10
    """
    match = re.search(r'\d+', str(text))
    if match:
        return int(match.group())
    raise ValueError(f"Format siklus tidak dikenali: {text}")
```

### 2.4 Mapping Dictionary — Gejala Ordinal (CTCAE 0-3)

```python
CTCAE_MAP = {
    "tidak ada": 0, "none": 0,
    "ringan": 1, "mild": 1,
    "sedang": 2, "moderate": 2,
    "berat": 3, "severe": 3,
}

SYMPTOM_COLS = ["mual", "muntah", "fatigue", "diare", "konstipasi", "mukositis", "nyeri"]

def map_ctcae(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return int(value)  # sudah numerik
    return CTCAE_MAP.get(str(value).strip().lower(), np.nan)
```

### 2.5 Encoding Kategorik Lain

```python
FAMILY_SUPPORT_MAP = {"rendah": 0, "sedang": 1, "tinggi": 2}
GENDER_MAP = {"l": 0, "laki-laki": 0, "p": 1, "perempuan": 1}
TARGET_MAP = {"ringan": 0, "sedang": 1, "berat": 2}

df["dukungan_keluarga"] = df["Dukungan Keluarga"].str.lower().map(FAMILY_SUPPORT_MAP)
df["jenis_kelamin"] = df["Jenis Kelamin"].str.lower().map(GENDER_MAP)
df["TARGET_SEVERITY"] = df["Tingkat Keparahan"].str.lower().map(TARGET_MAP)
```

### 2.6 Fitur Kategorik Nominal (One-Hot)

```python
CATEGORICAL_NOMINAL = ["diagnosis_kanker", "protokol_kemoterapi"]

def group_rare_categories(series, min_freq=5):
    counts = series.value_counts()
    rare = counts[counts < min_freq].index
    return series.replace(rare, "Lainnya")

df["diagnosis_kanker"] = group_rare_categories(df["diagnosis_kanker"])
df["protokol_kemoterapi"] = group_rare_categories(df["protokol_kemoterapi"])
```

### 2.7 Final Feature List

```python
NUMERIC_CONTINUOUS = ["hb", "leukosit", "neutrofil", "trombosit", "suhu"]
NUMERIC_DISCRETE   = ["usia_bulan", "lama_terdiagnosis_bulan", "siklus_ke"]
ORDINAL_FEATURES   = SYMPTOM_COLS + ["dukungan_keluarga"]
BINARY_FEATURES    = ["jenis_kelamin", "riwayat_rawat_inap"]
NOMINAL_FEATURES   = CATEGORICAL_NOMINAL  # -> one-hot

TARGET_COL = "TARGET_SEVERITY"
```

**Aturan wajib:** `NUMERIC_CONTINUOUS` di-scale dengan `StandardScaler`. `ORDINAL_FEATURES` dan `BINARY_FEATURES` **tidak** perlu di-scale (biarkan numerik apa adanya, atau opsional scale bila memakai Logistic Regression). `NOMINAL_FEATURES` di-one-hot-encode via `OneHotEncoder(handle_unknown="ignore")`.

---

## 3. Machine Learning & 5-Fold Cross-Validation Pipeline Spec

### 3.1 `ColumnTransformer` + `Pipeline` (Anti Data-Leakage)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC_CONTINUOUS + NUMERIC_DISCRETE),
    ("ord", SimpleImputer(strategy="most_frequent"), ORDINAL_FEATURES + BINARY_FEATURES),
    ("nom", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), NOMINAL_FEATURES),
])

def build_pipeline(estimator):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])
```

> `preprocessor.fit()` **hanya** dipanggil di dalam `pipeline.fit(X_train, y_train)` per fold — tidak pernah di-fit pada seluruh dataset sebelum split.

### 3.2 Model Definitions (Imbalance-Aware)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

MODELS = {
    "Logistic Regression": build_pipeline(
        LogisticRegression(
            penalty="l2", C=0.5, solver="lbfgs",
            multi_class="multinomial", class_weight="balanced",
            max_iter=1000, random_state=42,
        )
    ),
    "Random Forest": build_pipeline(
        RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=4,
            max_features="sqrt", class_weight="balanced", random_state=42,
        )
    ),
    "XGBoost": build_pipeline(
        XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42,
        )
    ),
}

def compute_sample_weight(y_train):
    """XGBoost tidak punya class_weight bawaan -> gunakan sample_weight."""
    from sklearn.utils.class_weight import compute_sample_weight as csw
    return csw(class_weight="balanced", y=y_train)
```

### 3.3 Stratified 5-Fold CV Loop

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

def run_cross_validation(X, y, models=MODELS, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_results = {name: [] for name in models}
    agg_confusion = {name: np.zeros((3, 3)) for name in models}
    roc_data = {name: {"y_true": [], "y_proba": []} for name in models}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, pipeline in models.items():
            fit_kwargs = {}
            if name == "XGBoost":
                fit_kwargs["classifier__sample_weight"] = compute_sample_weight(y_train)

            pipeline.fit(X_train, y_train, **fit_kwargs)
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)

            cv_results[name].append({
                "fold": fold_idx,
                "accuracy": accuracy_score(y_val, y_pred),
                "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
                "roc_auc_ovr_macro": roc_auc_score(
                    y_val, y_proba, multi_class="ovr", average="macro"
                ),
            })
            agg_confusion[name] += confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
            roc_data[name]["y_true"].extend(y_val.tolist())
            roc_data[name]["y_proba"].extend(y_proba.tolist())

    return cv_results, agg_confusion, roc_data
```

### 3.4 Agregasi Metrik (Mean ± Std)

```python
def summarize_cv(cv_results: dict) -> pd.DataFrame:
    rows = []
    for name, folds in cv_results.items():
        df_folds = pd.DataFrame(folds)
        row = {"model": name}
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]:
            row[f"{metric}_mean"] = df_folds[metric].mean()
            row[f"{metric}_std"] = df_folds[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)
```

### 3.5 Serialisasi Model Terbaik

```python
import joblib

def save_best_model(models: dict, summary_df: pd.DataFrame, X, y, out_dir="models"):
    best_name = summary_df.sort_values("f1_macro_mean", ascending=False).iloc[0]["model"]
    best_pipeline = models[best_name]

    # Refit pada SELURUH data (bukan hanya satu fold) untuk model produksi/live predict
    fit_kwargs = {}
    if best_name == "XGBoost":
        fit_kwargs["classifier__sample_weight"] = compute_sample_weight(y)
    best_pipeline.fit(X, y, **fit_kwargs)

    joblib.dump(best_pipeline, f"{out_dir}/best_model.pkl")
    metadata = {
        "model_name": best_name,
        "feature_columns": X.columns.tolist(),
        "target_map": {0: "Ringan", 1: "Sedang", 2: "Berat"},
        "trained_on_n": len(X),
    }
    joblib.dump(metadata, f"{out_dir}/preprocessing_metadata.pkl")
    return best_name, best_pipeline
```

| Artifact | File | Isi |
|---|---|---|
| Model produksi | `models/best_model.pkl` | Pipeline lengkap (preprocessor + classifier), sudah di-fit ulang di seluruh data |
| Metadata | `models/preprocessing_metadata.pkl` | Nama model, urutan kolom fitur, target mapping |
| Semua model (opsional) | `models/all_pipelines.pkl` | Dict 3 pipeline untuk keperluan Tab 3 (compare live prediction antar model) |

---

## 4. Streamlit UI Component & Layout Specification

### 4.1 Global Layout

```python
st.set_page_config(page_title="Chemo Severity Predictor", layout="wide", page_icon="🩺")

with st.sidebar:
    st.title("🩺 Chemo Severity Predictor")
    st.caption("Prediksi dini keparahan efek samping kemoterapi anak — ML pipeline & decision support tool.")
    st.divider()
    st.markdown(f"**Dataset:** {len(st.session_state.df_clean)} responden")
    if st.session_state.cv_done:
        st.markdown(f"**Model aktif:** `{st.session_state.best_model_name}`")

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧪 Model Benchmarking", "⚡ Live Predictor"])
```

### 4.2 Tab 1 — Exploratory Data Analysis

| Komponen | Widget/Library | Detail |
|---|---|---|
| Ringkasan dataset | `st.columns(4)` + `st.metric` | Total N, %Ringan, %Sedang, %Berat |
| Distribusi target | `plotly.express.bar` / `pie` | Jumlah & proporsi 3 kelas, anotasi angka absolut |
| Distribusi gejala | `plotly.express.imshow` (heatmap) | Rata-rata skor CTCAE per gejala × per kelas target |
| Korelasi lab | `plotly.express.box` + heatmap Spearman | Boxplot Hb/Leukosit/Neutrofil/Trombosit/Suhu per kelas |
| Dukungan keluarga | `plotly.express.bar` (stacked, `barnorm="percent"`) | Proporsi Rendah/Sedang/Tinggi per kelas target |
| Filter | `st.multiselect`, `st.slider` | Filter usia & diagnosis untuk eksplorasi subgrup |

```python
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Responden", len(df))
    c2.metric("Ringan", f"{n_ringan} ({pct_ringan:.0f}%)")
    c3.metric("Sedang", f"{n_sedang} ({pct_sedang:.0f}%)")
    c4.metric("Berat", f"{n_berat} ({pct_berat:.0f}%)", delta="Minority class", delta_color="off")

    fig_target = px.bar(target_counts, x="Kelas", y="Jumlah", text="Jumlah", color="Kelas")
    st.plotly_chart(fig_target, use_container_width=True)
```

### 4.3 Tab 2 — Model Benchmarking & Evaluation

| Komponen | Widget/Library | Detail |
|---|---|---|
| Trigger training | `st.button("Run 5-Fold Cross-Validation")` | Memanggil `run_cross_validation()`, hasil disimpan ke `session_state` |
| Progress indicator | `st.spinner` / `st.progress` | Selama loop 5 fold × 3 model berjalan |
| Tabel metrik | `st.dataframe` dgn styling | Format `Mean ± Std`, highlight nilai tertinggi per kolom (`.style.highlight_max`) |
| Grafik komparasi | `plotly.graph_objects` grouped bar / radar | Accuracy, F1-Macro, Recall-Macro, ROC-AUC per model |
| Confusion Matrix | `plotly.express.imshow` | Selectbox pilih model → heatmap 3×3 dari `agg_confusion` |
| ROC Curve | `plotly.graph_objects.Scatter` | One-vs-Rest per kelas, per model, legend menampilkan AUC |
| Detail per-fold | `st.expander` + `st.dataframe` | Tabel 5 baris metrik mentah per model (untuk transparansi) |
| Export | `st.download_button` | CSV tabel metrik + PNG grafik (300dpi via `fig.write_image`) |

```python
with tab2:
    if st.button("🚀 Run 5-Fold Cross-Validation", type="primary"):
        with st.spinner("Training & validating 3 models across 5 folds..."):
            cv_results, agg_cm, roc_data = run_cross_validation(X, y)
            summary_df = summarize_cv(cv_results)
            best_name, pipelines = save_best_model(MODELS, summary_df, X, y)

        st.session_state.update({
            "cv_results": cv_results, "agg_confusion": agg_cm,
            "roc_data": roc_data, "cv_done": True,
            "best_model_name": best_name, "trained_pipelines": MODELS,
        })

    if st.session_state.cv_done:
        st.dataframe(format_mean_std_table(summary_df), use_container_width=True)
        model_choice = st.selectbox("Pilih model", list(MODELS.keys()))
        st.plotly_chart(plot_confusion_matrix(agg_cm[model_choice]), use_container_width=True)
        st.plotly_chart(plot_roc_curve(roc_data[model_choice]), use_container_width=True)
```

### 4.4 Tab 3 — Live Clinical Predictor

| Komponen | Widget | Detail |
|---|---|---|
| Form container | `st.form("patient_input")` | Semua input dikumpulkan, submit sekali via `st.form_submit_button` |
| Demografi | `number_input`, `radio`, `selectbox` | Usia (tahun/bulan toggle), jenis kelamin, diagnosis, protokol, siklus |
| Gejala harian | `st.slider(0, 3)` × 7 | Label deskriptif di tiap titik (`format_func`) |
| Dukungan keluarga | `st.selectbox` | Rendah / Sedang / Tinggi |
| Lab & vital | `st.number_input` × 5 | Rentang normal ditampilkan via `help=` |
| Pemilihan model | `st.selectbox` | Default: `st.session_state.best_model_name` |
| Output | `st.success` / `st.warning` / `st.error` | Badge warna sesuai kelas prediksi (hijau/kuning/merah) |
| Probabilitas | `plotly.express.bar` (horizontal) | Probabilitas 3 kelas dari `predict_proba` |
| Rekomendasi | `st.info` (teks statis, dict lookup) | Rekomendasi keperawatan per kelas — **bukan** output model |
| Disclaimer | `st.caption` (fixed, selalu tampil) | "Hasil bersifat estimatif, bukan pengganti asesmen klinis." |

```python
with tab3:
    with st.form("patient_input"):
        col1, col2 = st.columns(2)
        usia_th = col1.number_input("Usia (tahun)", 0, 18, 5)
        jk = col2.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        siklus = col1.number_input("Siklus kemoterapi ke-", 1, 30, 2)
        gejala = {}
        for g in SYMPTOM_COLS:
            gejala[g] = st.slider(g.capitalize(), 0, 3, 0,
                format_func=lambda x: ["Tidak Ada","Ringan","Sedang","Berat"][x])
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        X_new = build_input_row(usia_th, jk, siklus, gejala, ...)
        pipeline = st.session_state.trained_pipelines[model_choice]
        pred = pipeline.predict(X_new)[0]
        proba = pipeline.predict_proba(X_new)[0]

        label_map = {0: ("Ringan", st.success), 1: ("Sedang", st.warning), 2: ("Berat", st.error)}
        label, render_fn = label_map[pred]
        render_fn(f"Prediksi: **{label}** ({proba[pred]*100:.1f}% confidence)")

        st.plotly_chart(px.bar(x=proba, y=["Ringan","Sedang","Berat"], orientation="h"))
        st.info(RECOMMENDATION_MAP[pred])
        st.caption("⚠️ Hasil bersifat estimatif — bukan pengganti asesmen klinis tenaga kesehatan.")
```

### 4.5 Rekomendasi Statis (Dict Lookup, bukan Model Output)

```python
RECOMMENDATION_MAP = {
    0: "Edukasi mandiri, pemantauan gejala rutin per shift, dokumentasi ulang kunjungan berikutnya.",
    1: "Intervensi simtomatik terarah, pemantauan tanda vital ketat, evaluasi ulang dalam 24 jam.",
    2: "Eskalasi segera ke DPJP, pertimbangan intervensi medis aktif, pemantauan intensif.",
}
```

---

## 5. Project Directory Structure & Artifacts

### 5.1 Struktur Folder VS Code

```
chemo-severity-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   └── Master Tabel-2.xlsx
│   └── processed/
│       └── dataset_clean.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # parse_usia_ke_bulan, parse_siklus, mapping dicts
│   ├── pipeline.py           # ColumnTransformer + build_pipeline
│   ├── cv_engine.py          # run_cross_validation, summarize_cv, save_best_model
│   ├── viz.py                # fungsi plot Plotly (reusable di app.py)
│   └── constants.py          # SYMPTOM_COLS, feature lists, mapping dicts
├── models/
│   ├── best_model.pkl
│   ├── all_pipelines.pkl
│   └── preprocessing_metadata.pkl
├── notebooks/
│   ├── 01_eda_exploration.ipynb
│   └── 02_model_experiments.ipynb
├── results/
│   ├── metrics_5fold.xlsx
│   ├── confusion_matrices/
│   └── figures/
└── tests/
    └── test_preprocessing.py
```

### 5.2 Standar Penamaan File Output (untuk Google Drive Tim)

| Artifact | Naming Convention | Format |
|---|---|---|
| Tabel metrik CV | `metrics_5fold_YYYYMMDD.xlsx` | Excel, sheet per model + sheet summary |
| Grafik komparasi model | `fig_model_comparison_YYYYMMDD.png` | PNG, 300dpi, `fig.write_image(..., scale=3)` |
| Confusion matrix | `fig_confmatrix_<model>_YYYYMMDD.png` | PNG, 300dpi |
| ROC curve | `fig_roc_<model>_YYYYMMDD.png` | PNG, 300dpi |
| Notebook eksperimen | `02_model_experiments_vN.ipynb` | Jupyter, versi increment manual |
| Model final | `best_model_vN.pkl` | joblib, disertai `CHANGELOG.md` singkat |

---

## 6. Implementation Task Checklist (Sprint Action Plan)

### Step 1 — Setup & Preprocessing
- [ ] Inisialisasi repo, venv, `requirements.txt`
- [ ] Load `Master Tabel-2.xlsx`, cek `df.columns` aktual vs mapping asumsi (Bab 2.1)
- [ ] Implementasi `parse_usia_ke_bulan()` + unit test 5 kasus edge (format ambigu, kosong, dst.)
- [ ] Implementasi `parse_siklus()` + unit test
- [ ] Implementasi mapping ordinal gejala + validasi tidak ada `NaN` tak terduga
- [ ] Implementasi encoding dukungan keluarga, jenis kelamin, target
- [ ] Export `dataset_clean.csv` ke `data/processed/`
- [ ] Cek distribusi missing value & duplikasi per kolom

### Step 2 — ML Pipeline & Validation
- [ ] Bangun `ColumnTransformer` sesuai kelompok fitur (Bab 3.1)
- [ ] Definisikan 3 model dengan `class_weight`/`sample_weight` (Bab 3.2)
- [ ] Implementasi `run_cross_validation()` — validasi tidak ada leakage (fit hanya di training fold)
- [ ] Implementasi `summarize_cv()` — Mean ± Std per metrik
- [ ] Jalankan CV, cek F1-Macro & Recall kelas 'Berat' tidak 0 (indikasi imbalance belum tertangani)
- [ ] `save_best_model()` — refit di seluruh data, serialisasi `.pkl`
- [ ] Simpan `metrics_5fold.xlsx` ke `results/`

### Step 3 — Streamlit App Core
- [ ] Setup `app.py` skeleton — `st.set_page_config`, sidebar, 3 tabs
- [ ] Implementasi `session_state` initialization (Bab 1.4)
- [ ] Tab 1 (EDA): metric cards, bar chart target, heatmap gejala, boxplot lab
- [ ] Tab 2: tombol Run CV, tabel metrik, confusion matrix, ROC curve, expander detail fold
- [ ] Tab 3: `st.form` input pasien, inference, output badge + bar chart probabilitas

### Step 4 — UI Polish & Export
- [ ] Tambahkan `help=` text pada semua `number_input` lab (rentang normal klinis)
- [ ] Tambahkan disclaimer klinis fixed di Tab 3
- [ ] Tambahkan `st.download_button` untuk CSV metrik & PNG grafik
- [ ] Cache: pastikan `@st.cache_data`/`@st.cache_resource` terpasang di fungsi load & train
- [ ] Uji end-to-end: restart app, klik antar tab, pastikan tidak retrain otomatis
- [ ] Tulis `README.md` — cara install, cara run (`streamlit run app.py`)
- [ ] Final review kode dgn `pytest tests/` (preprocessing unit tests)
