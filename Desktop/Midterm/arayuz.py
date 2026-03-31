

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.signal import correlate
import tempfile
import os

from ses_analizi import (
    make_frames,   # frame_signal yerine
    # compute_ste, compute_zcr, compute_f0_autocorrelation fonksiyonlarını aşağıda tanımlıysa import et
    # Eğer ses_analizi.py'de isimleri farklıysa, onları da buraya ekleyin
    # Diğer fonksiyonlar için de aynı şekilde
)

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Ses Analizi & Cinsiyet Sınıflandırma",
    page_icon="🎙️",
    layout="wide"
)


st.title("🎙️ Ses İşareti Analizi ve Cinsiyet Sınıflandırma")
st.markdown("**Otokorelasyon Tabanlı F0 Tespiti | Kural Tabanlı Sınıflandırıcı**")
st.markdown("---")

# Otokorelasyon yöntemi açıklaması ve formülü
st.markdown(r'''
### Otokorelasyon Yöntemi (Zaman Düzlemi)
Ses işaretinin periyodikliğini bulmak için otokorelasyon fonksiyonu kullanılır:

$$
R(\tau) = \sum_{n=0}^{N-1} x[n] \cdot x[n-\tau]
$$

Burada $R(\tau)$, gecikme ($\tau$) için otokorelasyon değerini, $x[n]$ ise ses sinyalini ifade eder. En yüksek tepe noktasının karşılık geldiği gecikme değeri temel frekansı (F0) verir:

$$
F_0 = \frac{f_s}{\tau_{peak}}
$$

Burada $f_s$ örnekleme frekansıdır, $\tau_{peak}$ ise ana tepe noktasıdır.

Bu projede F0 hesaplaması otokorelasyon yöntemiyle yapılmaktadır.
''')

# ============================================================
# SIDEBAR — EŞİK AYARLARI
# ============================================================
st.sidebar.header("⚙️ Sınıflandırma Eşikleri")
f0_th1 = st.sidebar.slider("Erkek–Kadın Eşiği (Hz)", 100, 220, 165, step=5)
f0_th2 = st.sidebar.slider("Kadın–Çocuk Eşiği (Hz)", 200, 350, 255, step=5)
st.sidebar.markdown(f"""
| Tahmin | Koşul |
|--------|-------|
| **Erkek** | F0 < {f0_th1} Hz |
| **Kadın** | {f0_th1} ≤ F0 < {f0_th2} Hz |
| **Çocuk** | F0 ≥ {f0_th2} Hz |
""")

# ============================================================
# TAB 1: TEK SES DOSYASI ANALİZİ
# ============================================================
tab1, tab2 = st.tabs(["🔍 Tek Dosya Analizi", "📊 Veri Seti Değerlendirmesi"])

with tab1:
    st.header("Tek Ses Dosyası Analizi")
    uploaded = st.file_uploader("WAV dosyası yükleyin", type=["wav"])
    
    if uploaded is not None:
        # Geçici dosyaya yaz
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        
        st.audio(uploaded)
        
        with st.spinner("Analiz yapılıyor..."):
            import librosa
            y, sr = librosa.load(tmp_path, sr=None)
            try:
                from ses_analizi import extract_features
                f0_mean, f0_std, zcr_mean, energy = extract_features(y, sr)
            except Exception as e:
                st.error(f"extract_features fonksiyonu çalıştırılamadı: {e}")
                f0_mean, f0_std, zcr_mean, energy = 0, 0, 0, 0
            # Kural tabanlı sınıflandırıcı
            if f0_mean < f0_th1:
                tahmin = 'E'
            elif f0_mean < f0_th2:
                tahmin = 'K'
            else:
                tahmin = 'C'
            features = {
                'f0_mean': f0_mean,
                'f0_std': f0_std,
                'zcr_mean': zcr_mean,
                'e_mean': energy,
                'f0_values': []
            }

        # Otokorelasyon ve FFT grafiği
        try:
            from ses_analizi import plot_autocorr_vs_fft
            fig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            plot_autocorr_vs_fft(y, sr, label="", save_path=fig_path)
            st.markdown("#### Otokorelasyon ve FFT (Büyüklük Yok)")
            st.image(fig_path, width=900)
            os.unlink(fig_path)
        except Exception as e:
            st.warning(f"Otokorelasyon/FFT grafiği çizilemedi: {e}")
        
        os.unlink(tmp_path)
        
        if features is None:
            st.error("❌ Ses dosyasından yeterli öznitelik çıkarılamadı. "
                     "Sessiz veya gürültülü kayıt olabilir.")
        else:
            f0_mean = features['f0_mean']
            f0_std  = features['f0_std']
            zcr     = features['zcr_mean']
            energy  = features['e_mean']
            from ses_analizi import rule_based_classify
            tahmin  = rule_based_classify(f0_mean)
            label_map  = {'Erkek': '👨 ERKEK', 'Kadın': '👩 KADIN', 'Çocuk': '👦 ÇOCUK', 'Bilinmiyor': '❓ TANIMSIZ'}
            color_map  = {'Erkek': 'blue', 'Kadın': 'violet', 'Çocuk': 'green', 'Bilinmiyor': 'gray'}
            
            st.markdown(f"### Tahmin Sonucu (Otokorelasyon ile F0)")
            st.markdown(
                f"<h2 style='color:{color_map[tahmin]};'>{label_map[tahmin]}</h2>",
                unsafe_allow_html=True
            )
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ortalama F0", f"{f0_mean:.1f} Hz")
            col2.metric("F0 Std. Sapma", f"{f0_std:.1f} Hz")
            col3.metric("Ortalama ZCR", f"{zcr:.4f}")
            col4.metric("Ortalama Enerji", f"{energy:.6f}")
            
            # F0 zaman serisi grafiği
            st.markdown("#### F0 Zaman Serisi (Voiced Çerçeveler)")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(features['f0_values'], color='steelblue', linewidth=1)
            ax.axhline(f0_mean, color='red', linestyle='--', label=f'Ortalama F0 = {f0_mean:.1f} Hz')
            ax.axhline(f0_th1, color='orange', linestyle=':', alpha=0.7, label=f'Erkek-Kadın sınırı ({f0_th1} Hz)')
            ax.axhline(f0_th2, color='green',  linestyle=':', alpha=0.7, label=f'Kadın-Çocuk sınırı ({f0_th2} Hz)')
            ax.set_xlabel("Çerçeve")
            ax.set_ylabel("F0 (Hz)")
            ax.set_title("Çerçeve Bazlı F0 Değerleri")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

# ============================================================
# TAB 2: VERİ SETİ DEĞERLENDİRMESİ
# ============================================================
with tab2:
    st.header("Tüm Veri Seti Değerlendirmesi")
    
    col_a, col_b = st.columns(2)
    with col_a:
        metadata_file = st.file_uploader("Metadata Excel dosyası (.xlsx)", type=["xlsx"])
    with col_b:
        dataset_path = st.text_input("Ses dosyaları klasörü", value="Dataset/")
    
    if metadata_file is not None:
        # Metadata dosyasını oku ve cinsiyet etiketlerini normalize et
        metadata = pd.read_excel(metadata_file)
        if 'Cinsiyet' in metadata.columns:
            metadata['Cinsiyet'] = metadata['Cinsiyet'].astype(str).str.strip().str.upper()
            metadata['Cinsiyet'] = metadata['Cinsiyet'].replace({'F': 'K', 'M': 'E'})
            metadata = metadata[metadata['Cinsiyet'].isin(['E', 'K', 'C'])]
        df_meta = pd.read_excel(metadata_file)
        df_meta['Cinsiyet'] = df_meta['Cinsiyet'].astype(str).str.strip().str.upper()
        df_meta['Cinsiyet'] = df_meta['Cinsiyet'].replace({'F': 'K', 'M': 'E'})
        df_meta = df_meta[df_meta['Cinsiyet'].isin(['E', 'K', 'C'])]
        
        st.info(f"Metadata yüklendi: **{len(df_meta)}** kayıt  "
                f"(E:{(df_meta['Cinsiyet']=='E').sum()}  "
                f"K:{(df_meta['Cinsiyet']=='K').sum()}  "
                f"C:{(df_meta['Cinsiyet']=='C').sum()})")
        
        if st.button("🚀 Analizi Başlat", type="primary"):
            progress = st.progress(0)
            status   = st.empty()
            results_list = []
            
            import glob as _glob
            
            def find_audio(filename, root):
                matches = _glob.glob(os.path.join(root, "**", filename), recursive=True)
                return matches[0] if matches else None
            
            total = len(df_meta)
            from ses_analizi import extract_features
            for i, (_, row) in enumerate(df_meta.iterrows()):
                progress.progress((i + 1) / total)
                status.text(f"İşleniyor: {i+1}/{total} — {row['Dosya_Adi']}")
                
                audio_path = find_audio(row['Dosya_Adi'], dataset_path)
                gercek = str(row['Cinsiyet']).strip().upper()
                
                if audio_path is None:
                    continue
                
                import librosa
                y, sr = librosa.load(audio_path, sr=None)
                try:
                    f0_mean, f0_std, zcr_mean, energy = extract_features(y, sr)
                except Exception as e:
                    f0_mean, f0_std, zcr_mean, energy = None, None, None, None
                if f0_mean is None:
                    tahmin = 'TANIMSIZ'
                    f0m = None
                else:
                    f0m = f0_mean
                    if f0m < f0_th1:
                        tahmin = 'E'
                    elif f0m < f0_th2:
                        tahmin = 'K'
                    else:
                        tahmin = 'C'
                
                results_list.append({
                    'Dosya'  : row['Dosya_Adi'],
                    'Gercek' : gercek,
                    'Gercek_ENG': row['Cinsiyet'],
                    'Tahmin' : tahmin,
                    'Tahmin_ENG': (
                        'M' if tahmin == 'E' else
                        'F' if tahmin == 'K' else
                        'C' if tahmin == 'C' else 'UNDEF'),
                    'F0_Mean': round(f0m, 2) if f0m else None,
                    'Dogru'  : gercek == tahmin,
                    'F0_Std' : round(f0_std, 2) if f0_std else None,
                    'F0_Range': round((np.nanmax([f0m, 0]) - np.nanmin([f0m, 0])), 2) if f0m else None
                })
            
            status.text("✅ Analiz tamamlandı!")
            results_df = pd.DataFrame(results_list)
            valid_df   = results_df[results_df['Tahmin'] != 'TANIMSIZ']
            
            if len(valid_df) == 0:
                st.error("Hiç geçerli tahmin yapılamadı. Klasör yolunu kontrol edin.")
            else:
                accuracy = valid_df['Dogru'].mean() * 100
                st.success(f"### 🎯 Genel Accuracy: %{accuracy:.1f}")
                
                # Sınıf bazlı tablo
                st.subheader("Sınıf Bazlı Sonuçlar")
                rows = []
                for sinif, label in [('E','Erkek'), ('K','Kadın'), ('C','Çocuk')]:
                    s = valid_df[valid_df['Gercek'] == sinif]
                    if len(s) == 0:
                        continue
                    # Sadece makul F0 aralığı (60-400 Hz) filtrele
                    f0s = s['F0_Mean'].dropna()
                    f0s = f0s[(f0s >= 60) & (f0s <= 400)]
                    f0_range = f0s.max() - f0s.min() if len(f0s) > 0 else None
                    rows.append({
                        'Sınıf'       : label,
                        'Örnek Sayısı': len(s),
                        'Ort. F0 (Hz)': round(f0s.mean(), 1) if len(f0s) > 0 else None,
                        'Std. Sapma'  : round(f0s.std(), 1) if len(f0s) > 0 else None,
                        'F0 Aralığı (Hz)': round(f0_range, 1) if f0_range is not None else None,
                        'Başarı (%)'  : round(s['Dogru'].mean() * 100, 1)
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                
                # Karışıklık matrisi
                st.subheader("Karışıklık Matrisi")
                from sklearn.metrics import confusion_matrix
                labels = [l for l in ['E', 'K', 'C'] if l in valid_df['Gercek'].values]
                cm = confusion_matrix(valid_df['Gercek'], valid_df['Tahmin'], labels=labels)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cm, cmap='Blues')
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                label_names = {'E': 'Erkek', 'K': 'Kadın', 'C': 'Çocuk'}
                ax.set_xticklabels([label_names[l] for l in labels])
                ax.set_yticklabels([label_names[l] for l in labels])
                ax.set_xlabel("Tahmin")
                ax.set_ylabel("Gerçek")
                ax.set_title("Karışıklık Matrisi")
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                                color='white' if cm[i][j] > cm.max() / 2 else 'black',
                                fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # CSV indir
                csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 Sonuçları CSV İndir", csv,
                                   "sonuclar.csv", "text/csv")
