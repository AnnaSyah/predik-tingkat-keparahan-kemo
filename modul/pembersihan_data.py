import pandas as pd
import numpy as np
import re
import os

def bersihkan_dataset_kemo():
    print("Memulai proses pembersihan data...")
    raw_path = os.path.join("data", "mentah", "Master Tabel-2.xlsx")
    clean_path = os.path.join("data", "bersih", "dataset_clean.csv")

    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)

    # Membaca excel tanpa header agar index kolom sesuai dengan angka 0-35
    df_raw = pd.read_excel(raw_path, sheet_name=0, header=None)

    # Mengambil data dari baris indeks 5 sebanyak 100 baris (pasien valid)
    df = df_raw.iloc[5:105].copy()
    
    # 1. Pilih kolom yang relevan
    kolom_relevan = [0, 4, 5, 6, 7, 8, 9, 10, 11] + list(range(12, 19)) + [25] + list(range(26, 31)) + [35]
    df = df[kolom_relevan].copy()
    
    # Memberi nama kolom untuk memudahkan
    nama_kolom = [
        'no', 'nama_anak', 'usia_str', 'jenis_kelamin', 'diagnosis', 'lama_terdiagnosis', 
        'siklus_kemoterapi_str', 'protokol_kemo', 'riwayat_ranap',
        'mual', 'muntah', 'fatigue', 'diare', 'konstipasi', 'mukositis', 'nyeri',
        'dukungan_keluarga', 
        'hb', 'leukosit', 'neutrofil', 'trombosit', 'suhu',
        'target_severity_str'
    ]
    df.columns = nama_kolom
    
    # 2. Parsing format usia
    def parse_usia_tahun(val):
        val = str(val).lower().replace(',', '.')
        
        # Cari angka untuk tahun dan bulan
        match_thn = re.search(r'(\d+(?:\.\d+)?)\s*tahun', val)
        match_bln = re.search(r'(\d+(?:\.\d+)?)\s*bulan', val)
        
        if match_thn or match_bln:
            thn = float(match_thn.group(1)) if match_thn else 0.0
            bln = float(match_bln.group(1)) if match_bln else 0.0
            return thn + (bln / 12.0)
            
        nums = re.findall(r'\d+(?:\.\d+)?', val)
        if nums:
            return float(nums[0])
            
        return np.nan
        
    df['usia_tahun'] = df['usia_str'].apply(parse_usia_tahun).astype('float64')

    # 3. Ekstraksi angka siklus kemo
    def parse_siklus(val):
        val = str(val).lower().strip()
        match = re.search(r'\d+', val)
        if match:
            return int(match.group(0))
        return np.nan
    df['siklus_ke'] = df['siklus_kemoterapi_str'].apply(parse_siklus).astype('float64')

    # 4. Mapping ordinal CTCAE
    gejala_cols = ['mual', 'muntah', 'fatigue', 'diare', 'konstipasi', 'mukositis', 'nyeri']
    map_gejala = {
        'tidak ada': 0,
        'ringan': 1,
        'sedang, berat': 2,
        'sedang': 2,
        'berat': 3
    }
    for col in gejala_cols:
        def parse_gejala(val):
            val = str(val).lower().strip()
            if val in map_gejala:
                return map_gejala[val]
            for k, v in map_gejala.items():
                if k in val:
                    return v
            return np.nan # Ubah default ke NaN agar bisa di-imputasi modus nanti
        df[col] = df[col].apply(parse_gejala)

    # 5. Mapping dukungan keluarga & jenis kelamin
    def parse_dukungan(val):
        val = str(val).lower()
        if 'rendah' in val: return 0
        elif 'sedang' in val: return 1
        elif 'tinggi' in val: return 2
        return np.nan
    df['dukungan_keluarga'] = df['dukungan_keluarga'].apply(parse_dukungan)

    def parse_jk(val):
        val = str(val).lower()
        if 'laki' in val: return 0
        elif 'perempuan' in val: return 1
        return np.nan
    df['jenis_kelamin'] = df['jenis_kelamin'].apply(parse_jk)

    # 6. Ekstrak data lab kontinu
    lab_cols = ['hb', 'leukosit', 'neutrofil', 'trombosit', 'suhu']
    for col in lab_cols:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 7. Mapping target keparahan
    def parse_target(val):
        val = str(val).lower()
        if 'ringan' in val: return 0
        elif 'sedang' in val: return 1
        elif 'berat' in val: return 2
        return np.nan
    df['target_severity'] = df['target_severity_str'].apply(parse_target)
    
    # Hapus baris yang target_severity-nya NaN agar tidak error saat training
    df = df.dropna(subset=['target_severity'])
    
    # Hapus kolom string asli
    kolom_hapus = ['usia_str', 'siklus_kemoterapi_str', 'target_severity_str']
    df = df.drop(columns=kolom_hapus)

    # 8. IMPUTASI PENUH AGAR BEBAS NaN
    fitur_numerik = ['usia_tahun', 'siklus_ke', 'hb', 'leukosit', 'neutrofil', 'trombosit', 'suhu']
    fitur_kategorik = ['jenis_kelamin', 'mual', 'muntah', 'fatigue', 'diare', 'konstipasi', 'mukositis', 'nyeri', 'dukungan_keluarga']
    
    for col in fitur_numerik:
        df[col] = df[col].fillna(df[col].median())
        
    for col in fitur_kategorik:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # Pastikan siklus dan gejala berupa integer
    df['siklus_ke'] = df['siklus_ke'].astype(int)
    for col in fitur_kategorik:
        df[col] = df[col].astype(int)

    # Simpan ke CSV
    df.to_csv(clean_path, index=False)
    print(f"Data berhasil dibersihkan (bebas NaN) dan disimpan di: {clean_path}")

if __name__ == "__main__":
    bersihkan_dataset_kemo()
