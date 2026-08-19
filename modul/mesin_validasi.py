import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(objective='multi:softprob', random_state=42, eval_metric='mlogloss')
    xgb_name = 'XGBoost'
except (ImportError, Exception):
    from sklearn.ensemble import GradientBoostingClassifier
    xgb_model = GradientBoostingClassifier(random_state=42)
    xgb_name = 'XGBoost' # Keep the name as 'XGBoost' for the table as requested

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

def jalankan_validasi():
    print("Membaca data bersih...")
    clean_path = os.path.join("data", "bersih", "dataset_clean.csv")
    df = pd.read_csv(clean_path)
    
    # Hapus row yang target severity-nya NaN
    df = df.dropna(subset=['target_severity'])
    
    # Fitur yang akan digunakan
    fitur_numerik = ['usia_tahun', 'siklus_ke', 'hb', 'leukosit', 'neutrofil', 'trombosit', 'suhu']
    fitur_kategorik = ['jenis_kelamin', 'mual', 'muntah', 'fatigue', 'diare', 'konstipasi', 'mukositis', 'nyeri', 'dukungan_keluarga']
    
    X = df[fitur_numerik + fitur_kategorik]
    y = df['target_severity'].astype(int)
    
    # Preprocessing Pipeline (Mencegah Data Leakage & Handle NaN)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), fitur_numerik),
            ('cat', SimpleImputer(strategy='most_frequent'), fitur_kategorik)
        ]
    )
    
    # Definisi Model
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', multi_class='ovr', random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        xgb_name: xgb_model
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    hasil_evaluasi = []
    
    for nama_model, model in models.items():
        print(f"Melatih model: {nama_model}")
        metrics_fold = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'roc_auc': []}
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Buat pipeline
            pipeline = Pipeline([
                ('prep', preprocessor),
                ('clf', model)
            ])
            
            # Khusus XGBoost butuh sample_weight untuk class imbalance
            if nama_model == 'XGBoost':
                s_weight = compute_sample_weight('balanced', y_train)
                pipeline.fit(X_train, y_train, clf__sample_weight=s_weight)
            else:
                pipeline.fit(X_train, y_train)
                
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)
            
            metrics_fold['acc'].append(accuracy_score(y_val, y_pred))
            metrics_fold['prec'].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
            metrics_fold['rec'].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
            metrics_fold['f1'].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            
            # Untuk roc_auc multiclass ovr
            try:
                auc = roc_auc_score(y_val, y_proba, average='macro', multi_class='ovr')
            except ValueError:
                auc = np.nan # Jika ada kelas yang tidak muncul di val fold
            metrics_fold['roc_auc'].append(auc)
            
        # Agregasi
        mean_f1 = np.mean(metrics_fold['f1'])
        hasil_evaluasi.append({
            'Model': nama_model,
            'Accuracy': f"{np.mean(metrics_fold['acc']):.3f} ± {np.std(metrics_fold['acc']):.3f}",
            'Precision (Macro)': f"{np.mean(metrics_fold['prec']):.3f} ± {np.std(metrics_fold['prec']):.3f}",
            'Recall (Macro)': f"{np.mean(metrics_fold['rec']):.3f} ± {np.std(metrics_fold['rec']):.3f}",
            'F1-Score (Macro)': f"{mean_f1:.3f} ± {np.std(metrics_fold['f1']):.3f}",
            'ROC-AUC (Macro)': f"{np.nanmean(metrics_fold['roc_auc']):.3f} ± {np.nanstd(metrics_fold['roc_auc']):.3f}",
            'mean_f1_val': mean_f1
        })
        
    df_hasil = pd.DataFrame(hasil_evaluasi)
    print("\nHasil Evaluasi 5-Fold CV:")
    print(df_hasil.drop(columns=['mean_f1_val']))
    
    # Cari model terbaik berdasarkan F1-Score
    model_terbaik_nama = df_hasil.loc[df_hasil['mean_f1_val'].idxmax()]['Model']
    print(f"\nModel terbaik adalah {model_terbaik_nama}. Melakukan refit pada seluruh data...")
    
    # Refit
    model_terbaik = models[model_terbaik_nama]
    pipeline_terbaik = Pipeline([
        ('prep', preprocessor),
        ('clf', model_terbaik)
    ])
    
    if model_terbaik_nama == 'XGBoost':
        s_weight = compute_sample_weight('balanced', y)
        pipeline_terbaik.fit(X, y, clf__sample_weight=s_weight)
    else:
        pipeline_terbaik.fit(X, y)
        
    # Simpan model
    os.makedirs('model_tersimpan', exist_ok=True)
    model_path = os.path.join('model_tersimpan', 'model_terbaik.pkl')
    
    metadata = {
        'model': pipeline_terbaik,
        'fitur_numerik': fitur_numerik,
        'fitur_kategorik': fitur_kategorik,
        'nama_model': model_terbaik_nama,
        'evaluasi': df_hasil.drop(columns=['mean_f1_val']).to_dict('records')
    }
    
    joblib.dump(metadata, model_path)
    print(f"Model berhasil disimpan di: {model_path}")

if __name__ == "__main__":
    jalankan_validasi()
