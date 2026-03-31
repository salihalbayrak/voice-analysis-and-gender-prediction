
# ─────────────────────────────────────────────────────────────
# BAĞIMLILIKLAR
# ─────────────────────────────────────────────────────────────
import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI olmayan ortamlarda da çalışır (sunucu, Colab)
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)

warnings.filterwarnings("ignore")

# Librosa opsiyonel – varsa kullan, yoksa kendi okuyucumuzu kullanırız
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ─────────────────────────────────────────────────────────────
# AYARLAR  ← Sadece burayı düzenleyin
# ─────────────────────────────────────────────────────────────
METADATA_EXCEL  = "birlesik_metadata.xlsx"  # Metadata dosya yolu
DATASET_ROOT    = "Dataset"                 # .wav klasörü
OUTPUT_EXCEL    = "sonuclar.xlsx"           # Sonuç tablosu

FRAME_MS        = 25      # Pencere uzunluğu (ms)  — yönerge: 20-30 ms
HOP_MS          = 10      # Adım boyutu (ms)
F0_MIN          = 60.0    # Hz – arama aralığı alt sınır
F0_MAX          = 500.0   # Hz – arama aralığı üst sınır

STE_THRESH_FAC  = 0.10    # Enerji eşiği: maks enerjinin %10'u
ZCR_VOICED_THR  = 0.30    # ZCR > bu eşik → sessiz bölge kabul

# Kural tabanlı sınıflandırma eşikleri (Hz)
TH_ERKEK_KADIN  = 200.0   # F0 < 200        → Erkek
TH_KADIN_COCUK  = 300.0   # 200 ≤ F0 < 300  → Kadın  |  F0 ≥ 300 → Çocuk




def _read_wav_fallback(path: str):
    """
    Standart kütüphane kullanarak PCM WAV okur.
    Stereo ise mono'ya çevirir. float32 normalize edilmiş döndürür.
    """
    import wave
    with wave.open(path, "rb") as wf:
        sr         = wf.getframerate()
        n_frames   = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        raw        = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError(f"Desteklenmeyen örnek genişliği: {sampwidth} byte")

    samples = np.frombuffer(raw, dtype=dtype_map[sampwidth])
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    max_val = float(2 ** (8 * sampwidth - 1))
    signal  = samples.astype(np.float32) / max_val
    return signal, sr


def load_audio(path: str):
    """WAV yükleyici: librosa varsa onu, yoksa fallback kullanır."""
    if HAS_LIBROSA:
        signal, sr = librosa.load(path, sr=None, mono=True)
        return signal, sr
    try:
        import soundfile as sf
        signal, sr = sf.read(path, always_2d=False)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        return signal.astype(np.float32), sr
    except ImportError:
        pass
    return _read_wav_fallback(path)


# ══════════════════════════════════════════════════════════════
# BÖLÜM 1 – VERİ YÜKLEME VE ETİKET NORMALİZASYONU
# ══════════════════════════════════════════════════════════════

def normalize_label(cinsiyet: str, yas) -> str:
    """
    Farklı grupların kullandığı çeşitli cinsiyet etiketlerini
    tek bir forma (Erkek / Kadın / Çocuk) çevirir.

    Gözlemlenen etiketler: C, c, C , M, E, F, K, k
    Kural:
        C / c / C(boşluk)         → Çocuk
        M / E  + yaş < 18         → Çocuk
        M / E  + yaş >= 18        → Erkek
        F / K / k + yaş < 18      → Çocuk
        F / K / k + yaş >= 18     → Kadın
    """
    g   = str(cinsiyet).strip()
    age = float(yas) if pd.notna(yas) else 99.0

    if g.upper() == "C":
        return "Çocuk"
    elif g in ("M", "E"):
        return "Çocuk" if age < 18 else "Erkek"
    elif g in ("F", "K", "k"):
        return "Çocuk" if age < 18 else "Kadın"
    return "Bilinmiyor"


def load_metadata(path: str) -> pd.DataFrame:
    """
    birlesik_metadata.xlsx dosyasını okur.
    Sütun ekler:
        Sinif : Erkek / Kadın / Çocuk
        Grup  : G01, G02, ...  (dosya adından çıkarılır)
    """
    df = pd.read_excel(path)
    df["Sinif"] = df.apply(
        lambda r: normalize_label(r["Cinsiyet"], r["Yas"]), axis=1
    )
    df["Grup"] = df["Dosya_Adi"].str.extract(r"^(G\d+)", expand=False)
    return df


# ══════════════════════════════════════════════════════════════
# BÖLÜM 2 – ZAMAN DÜZLEMİ ÖN İŞLEME
#           Pencereleme, STE, ZCR, Sesli Bölge Maskesi
# ══════════════════════════════════════════════════════════════

def make_frames(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    Sinyali örtüşen çerçevelere böler.
    Döndürür: (frame_length, n_frames) şeklinde 2D dizi.
    librosa.util.frame() ile eşdeğerdir.
    """
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    strides  = (signal.strides[0], signal.strides[0] * hop_length)
    return np.lib.stride_tricks.as_strided(
        signal,
        shape   = (frame_length, n_frames),
        strides = strides
    )


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """
    Kısa Süreli Enerji (STE):
        E[m] = (1 / N) * sum(x[n]^2)   (m. çerçeve)

    Sesli bölgeleri sessizden ayırt etmek için kullanılır.
    Yüksek enerji → aktif konuşma.
    """
    return np.sum(frames ** 2, axis=0) / frames.shape[0]


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """
    Sıfır Geçiş Oranı (ZCR):
        ZCR[m] = (1/(N-1)) * sum(|sign(x[n]) - sign(x[n-1])| / 2)

    Ünsüzler ve gürültü → yüksek ZCR.
    Periyodik (ünlü) sesler → düşük ZCR.
    F0 tespiti yalnızca düşük ZCR'li (sesli) bölgelerde yapılır.
    """
    signs     = np.sign(frames)
    crossings = np.abs(np.diff(signs, axis=0))      # (N-1, n_frames)
    return np.sum(crossings, axis=0) / (2.0 * (frames.shape[0] - 1))


def voiced_mask(ste: np.ndarray, zcr: np.ndarray) -> np.ndarray:
    """
    Sesli (Voiced) çerçeve maskesi:
        Koşul 1: STE > STE_THRESH_FAC * max(STE)  → yeterli enerji
        Koşul 2: ZCR < ZCR_VOICED_THR              → periyodik dalga

    Bu maske sağlanmadan F0 hesabı yapılmaz.
    """
    max_ste = np.max(ste)
    if max_ste == 0:
        return np.zeros(len(ste), dtype=bool)
    ste_thr = STE_THRESH_FAC * max_ste
    n = min(len(ste), len(zcr))
    return (ste[:n] > ste_thr) & (zcr[:n] < ZCR_VOICED_THR)


# ══════════════════════════════════════════════════════════════
# BÖLÜM 3 – OTOKORELASYon TABANLI F0 TESPİTİ
# ══════════════════════════════════════════════════════════════

def autocorrelation_f0_frame(frame: np.ndarray, sr: int) -> float:
    """
    TEK BİR ses çerçevesi için Otokorelasyon Yöntemi ile F0 hesaplar.

    Formül (yönergeden):
        R(tau) = sum( x[n] * x[n - tau] )

    Uygulama adımları:
        1. Hanning penceresi uygula  → spektral sızıntıyı azalt
        2. np.correlate(..., full)   → tam çapraz korelasyon
        3. Yalnızca tau >= 0 tarafını al
        4. Arama aralığını F0_MIN–F0_MAX ile sınırla:
               lag_min = sr / F0_MAX
               lag_max = sr / F0_MIN
        5. Bu aralıktaki en yüksek tepeyi bul
        6. Parabolik interpolasyon ile hassas lag tahmini
        7. F0 = sr / best_lag

    Döndürür: F0 (Hz) veya np.nan
    """
    frame_w = frame * np.hanning(len(frame))

    # Tam otokorelasyon (uzunluk = 2*N - 1)
    corr = np.correlate(frame_w, frame_w, mode="full")
    corr = corr[len(corr) // 2:]           # tau = 0, 1, 2, ...

    # Lag arama aralığı
    lag_min = max(1, int(sr / F0_MAX))
    lag_max = min(len(corr) - 1, int(sr / F0_MIN))

    if lag_min >= lag_max:
        return np.nan

    search = corr[lag_min:lag_max]
    if np.max(search) <= 0:
        return np.nan

    best_idx = int(np.argmax(search))
    best_lag = float(lag_min + best_idx)

    # Parabolik interpolasyon (daha doğru lag tahmini)
    i = lag_min + best_idx
    if 1 <= i < len(corr) - 1:
        y0, y1, y2 = corr[i - 1], corr[i], corr[i + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if denom != 0:
            best_lag = i + (y2 - y0) / denom

    return float(sr / best_lag)


def extract_features(signal: np.ndarray, sr: int):
    """
    Tüm ses sinyalinden öznitelik çıkarımı:
        f0_mean    : Sesli çerçeveler üzerinden ortalama F0 (Hz)
        f0_std     : F0 standart sapması
        zcr_mean   : Ortalama ZCR (sıfır geçiş / saniye)
        energy_mean: Sesli bölgelerin ortalama enerjisi

    Döndürür: (f0_mean, f0_std, zcr_mean, energy_mean)
    """
    fl     = int(sr * FRAME_MS / 1000)
    hl     = int(sr * HOP_MS   / 1000)
    frames = make_frames(signal, fl, hl)
    ste    = short_time_energy(frames)
    zcr    = zero_crossing_rate(frames)
    mask   = voiced_mask(ste, zcr)
    n      = min(frames.shape[1], len(mask))

    # F0 – yalnızca sesli çerçevelerden
    f0_list = []
    for i in range(n):
        if mask[i]:
            f0 = autocorrelation_f0_frame(frames[:, i], sr)
            if not np.isnan(f0):
                f0_list.append(f0)

    f0_mean = float(np.mean(f0_list)) if f0_list else np.nan
    f0_std  = float(np.std(f0_list))  if f0_list else np.nan

    # ZCR – saniye başına normalize
    zcr_per_sec = zcr[:n] * sr / hl
    zcr_mean    = float(np.mean(zcr_per_sec))

    # Enerji – sesli bölgeler
    voiced_ste  = ste[:n][mask[:n]]
    energy_mean = float(np.mean(voiced_ste)) if len(voiced_ste) > 0 else np.nan

    return f0_mean, f0_std, zcr_mean, energy_mean


# ══════════════════════════════════════════════════════════════
# BÖLÜM 4 – KURAL TABANLI SINIFLANDIRICI
# ══════════════════════════════════════════════════════════════

def rule_based_classify(f0: float) -> str:
    """
    F0 değerine göre sınıf tahmini (Erkek / Kadın / Çocuk).

    Eşikler AYARLAR bölümünden değiştirilebilir:
        F0 < TH_ERKEK_KADIN (165 Hz)                → Erkek
        TH_ERKEK_KADIN <= F0 < TH_KADIN_COCUK (255) → Kadın
        F0 >= TH_KADIN_COCUK                         → Çocuk

    Literatür ortalamaları:
        Erkek  : ~85–180 Hz
        Kadın  : ~165–255 Hz
        Çocuk  : ~250–400 Hz
    """
    if np.isnan(f0):
        return "Bilinmiyor"
    if f0 < TH_ERKEK_KADIN:
        return "Erkek"
    elif f0 < TH_KADIN_COCUK:
        return "Kadın"
    elif f0 < 400:
        return "Çocuk"
    else:
        return "Tanımsız"


# ══════════════════════════════════════════════════════════════
# BÖLÜM 5 – GÖRSEL KARŞILAŞTIRMA: Otokorelasyon vs FFT
#           Yönerge Bölüm 3-A
# ══════════════════════════════════════════════════════════════

def plot_autocorr_vs_fft(signal: np.ndarray, sr: int,
                          label: str = "",
                          save_path: str = "autocorr_vs_fft.png"):
    """
    Yönerge Bölüm 3-A:
        'Otokorelasyon grafiğini ve FFT Spektrumunu yan yana çizin.
         Otokorelasyon ile bulduğunuz F0 değeri ile spektrumdaki ilk
         tepe noktasının yaklaşık aynı değere sahip olup olmadığını
         raporunuzda tartışın.'

    Sol  → R(tau) eğrisi, tepe noktasına kırmızı dikey çizgi
    Sağ  → |X(f)| spektrumu, F0 frekansına kırmızı dikey çizgi
    """
    fl     = int(sr * FRAME_MS / 1000)
    hl     = int(sr * HOP_MS   / 1000)
    frames = make_frames(signal, fl, hl)
    ste    = short_time_energy(frames)
    zcr    = zero_crossing_rate(frames)
    mask   = voiced_mask(ste, zcr)
    n      = min(frames.shape[1], len(mask))
    vi     = np.where(mask[:n])[0]

    if len(vi) == 0:
        print("  [UYARI] Sesli çerçeve bulunamadı, grafik üretilemedi.")
        return

    sel     = vi[len(vi) // 2]
    frame   = frames[:, sel]
    frame_w = frame * np.hanning(fl)

    # Otokorelasyon
    corr    = np.correlate(frame_w, frame_w, mode="full")
    corr    = corr[len(corr) // 2:]
    lags_ms = np.arange(len(corr)) / sr * 1000
    f0_auto = autocorrelation_f0_frame(frame, sr)

    # FFT
    N        = 2048
    spectrum = np.abs(np.fft.rfft(frame_w, n=N))
    freqs    = np.fft.rfftfreq(N, d=1.0 / sr)
    f_mask   = (freqs >= F0_MIN) & (freqs <= F0_MAX)
    f0_fft   = freqs[f_mask][np.argmax(spectrum[f_mask])] if np.any(f_mask) else np.nan

    # Çizim
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Otokorelasyon ve FFT", fontsize=11)

    # Sol: R(tau)
    ax     = axes[0]
    show_n = np.searchsorted(lags_ms, 35)
    ax.plot(lags_ms[1:show_n], corr[1:show_n], color="steelblue", lw=1.5)
    ax.set_xlabel("Gecikme tau (ms)", fontsize=11)
    ax.set_ylabel("R(tau)", fontsize=11)
    ax.set_title("Otokorelasyon", fontsize=11)
    ax.grid(alpha=0.3)

    # Sağ: |X(f)|
    ax     = axes[1]
    show_f = freqs <= 800
    ax.plot(freqs[show_f], spectrum[show_f], color="darkorange", lw=1.5)
    ax.set_xlabel("Frekans (Hz)", fontsize=11)
    ax.set_ylabel("Spektrum", fontsize=11)
    ax.set_title("FFT", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Kaydedildi: {save_path}")


# ══════════════════════════════════════════════════════════════
# BÖLÜM 6 – ANA İŞLEM DÖNGÜSÜ
# ══════════════════════════════════════════════════════════════

def process_dataset(metadata_excel: str, dataset_root: str) -> pd.DataFrame:
    """
    1. Metadata yükle
    2. Her .wav dosyasını glob ile bul (özyinelemeli)
    3. F0, ZCR, Enerji çıkar
    4. Kural tabanlı sınıflandır
    5. Sonuçları DataFrame olarak döndür ve Excel'e kaydet
    """
    df = load_metadata(metadata_excel)

    print(f"\n{'='*60}")
    print(f"  Toplam kayıt : {len(df)}")
    print(f"  Erkek        : {(df['Sinif']=='Erkek').sum()}")
    print(f"  Kadın        : {(df['Sinif']=='Kadın').sum()}")
    print(f"  Çocuk        : {(df['Sinif']=='Çocuk').sum()}")
    print(f"  Dataset kök  : {os.path.abspath(dataset_root)}")
    print(f"{'='*60}\n")

    rows        = []
    first_sig   = None
    first_sr    = None
    first_label = ""

    for _, row in df.iterrows():
        fname = row["Dosya_Adi"]
        sinif = row["Sinif"]

        # Dosyayı tüm alt klasörlerde ara
        matches = glob.glob(
            os.path.join(dataset_root, "**", fname), recursive=True
        )
        if not matches:
            direct = os.path.join(dataset_root, fname)
            if os.path.exists(direct):
                matches = [direct]

        if not matches:
            rows.append({**row.to_dict(),
                         "F0_mean": np.nan, "F0_std": np.nan,
                         "ZCR_mean": np.nan, "Enerji_mean": np.nan,
                         "Tahmin": "Bulunamadı", "Dogru_mu": False})
            continue

        fpath = matches[0]

        try:
            signal, sr = load_audio(fpath)

            if first_sig is None:
                first_sig   = signal
                first_sr    = sr
                first_label = f"{fname}  ({sinif})"

            f0_mean, f0_std, zcr_mean, enerji = extract_features(signal, sr)
            tahmin  = rule_based_classify(f0_mean)
            dogru   = (tahmin == sinif)

        except Exception as e:
            print(f"  [HATA] {fname}: {e}")
            f0_mean = f0_std = zcr_mean = enerji = np.nan
            tahmin, dogru = "Hata", False

        rows.append({**row.to_dict(),
                     "F0_mean"    : round(f0_mean, 2) if pd.notna(f0_mean) else np.nan,
                     "F0_std"     : round(f0_std,  2) if pd.notna(f0_std)  else np.nan,
                     "ZCR_mean"   : round(zcr_mean, 2) if pd.notna(zcr_mean) else np.nan,
                     "Enerji_mean": round(enerji, 6)   if pd.notna(enerji)   else np.nan,
                     "Tahmin"     : tahmin,
                     "Dogru_mu"   : dogru})

        ok     = "✓" if dogru else "✗"
        f0_str = f"{f0_mean:.1f} Hz" if pd.notna(f0_mean) else "—"
        print(f"  {ok}  {fname:<45}  "
              f"F0={f0_str:<10}  Gerçek={sinif:<7}  Tahmin={tahmin}")

    result_df = pd.DataFrame(rows)
    result_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n  -> Sonuçlar kaydedildi: {OUTPUT_EXCEL}")

    if first_sig is not None:
        plot_autocorr_vs_fft(first_sig, first_sr,
                             label=first_label,
                             save_path="autocorr_vs_fft.png")

    return result_df


# ══════════════════════════════════════════════════════════════
# BÖLÜM 7 – İSTATİSTİK TABLOSU, CONFUSION MATRIX, GRAFİKLER
#           Yönerge Bölüm 5
# ══════════════════════════════════════════════════════════════

def statistics_and_plots(result_df: pd.DataFrame):
    """
    Yönerge Bölüm 5 tablosu:
        Sınıf | Örnek Sayısı | Ortalama F0 | Std Sapma | Başarı (%)

    Ayrıca:
        - Sklearn classification report (Precision / Recall / F1)
        - Confusion Matrix (confusion_matrix.png)
        - F0 histogram + kutu grafiği (f0_dagilim.png)
        - Hata analizi Excel (hatalar.xlsx)
    """
    df = result_df[
        result_df["F0_mean"].notna() &
        result_df["Sinif"].isin(["Erkek", "Kadın", "Çocuk"]) &
        result_df["Tahmin"].isin(["Erkek", "Kadın", "Çocuk"])
    ].copy()

    if df.empty:
        print("\n[UYARI] Geçerli F0 değeri olan kayıt bulunamadı.")
        print("  Dataset/ klasörünüzün doğru yerde olduğunu kontrol edin.")
        return

    # Bölüm 5 Tablosu
    print(f"\n{'='*68}")
    print("  BÖLÜM 5 – İSTATİSTİKSEL TABLO")
    print(f"{'='*68}")
    hdr = f"  {'Sınıf':<10} {'Örnek':>7} {'Ort F0 (Hz)':>13} {'Std Sapma':>11} {'Başarı (%)':>12}"
    print(hdr)
    print("  " + "-" * 56)
    for sinif in ["Erkek", "Kadın", "Çocuk"]:
        grp = df[df["Sinif"] == sinif]
        if grp.empty:
            continue
        ort = grp["F0_mean"].mean()
        std = grp["F0_mean"].std()
        acc = 100.0 * grp["Dogru_mu"].mean()
        print(f"  {sinif:<10} {len(grp):>7} {ort:>13.1f} {std:>11.1f} {acc:>11.1f}%")
    genel = 100.0 * df["Dogru_mu"].mean()
    print("  " + "-" * 56)
    print(f"  {'GENEL DOĞRULUK':<48} {genel:>8.1f}%")
    print(f"{'='*68}\n")

    # Classification report
    y_true = df["Sinif"].tolist()
    y_pred = df["Tahmin"].tolist()
    labels = ["Erkek", "Kadın", "Çocuk"]
    print("  Detaylı Sınıflandırma Raporu (Precision / Recall / F1):")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Karışıklık Matrisi (Confusion Matrix)", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("  -> confusion_matrix.png kaydedildi")

    # F0 Dağılım Grafiği
    colors = {"Erkek": "steelblue", "Kadın": "tomato", "Çocuk": "mediumseagreen"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for sinif, clr in colors.items():
        vals = df[df["Sinif"] == sinif]["F0_mean"].dropna()
        if len(vals):
            ax.hist(vals, bins=25, alpha=0.6, color=clr,
                    edgecolor="white", label=sinif)
    ax.axvline(TH_ERKEK_KADIN, color="gray",  ls="--", lw=1.5,
               label=f"E/K esigi: {TH_ERKEK_KADIN} Hz")
    ax.axvline(TH_KADIN_COCUK, color="black", ls="--", lw=1.5,
               label=f"K/C esigi: {TH_KADIN_COCUK} Hz")
    ax.set_xlabel("F0 (Hz)", fontsize=11)
    ax.set_ylabel("Kayıt Sayısı", fontsize=11)
    ax.set_title("F0 Histogramı — Sınıf Bazlı", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    box_data = [df[df["Sinif"] == s]["F0_mean"].dropna().values
                for s in ["Erkek", "Kadın", "Çocuk"]]
    bp = ax2.boxplot(box_data, tick_labels=["Erkek", "Kadın", "Çocuk"],
                     patch_artist=True, notch=True)
    for patch, clr in zip(bp["boxes"], colors.values()):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax2.set_ylabel("F0 (Hz)", fontsize=11)
    ax2.set_title("F0 Kutu Grafiği", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("F0 Dağılımı – Erkek / Kadın / Çocuk", fontsize=13)
    plt.tight_layout()
    plt.savefig("f0_dagilim.png", dpi=150)
    plt.close()
    print("  -> f0_dagilim.png kaydedildi")

    # Hata Analizi
    hatalar = df[~df["Dogru_mu"]][
        ["Dosya_Adi", "Sinif", "F0_mean", "ZCR_mean",
         "Tahmin", "Duygu", "Yas", "gürültü seviyesi"]
    ]
    if not hatalar.empty:
        print(f"\n  Yanlış Tahmin Edilen Dosyalar ({len(hatalar)} adet):")
        print(hatalar.to_string(index=False))
        hatalar.to_excel("hatalar.xlsx", index=False)
        print("  -> hatalar.xlsx kaydedildi")
    else:
        print("\n  Tüm dosyalar doğru sınıflandırıldı!")


# ══════════════════════════════════════════════════════════════
# BÖLÜM 8 – TEK DOSYA TAHMİNİ  (Arayüz entegrasyonu için)
# ══════════════════════════════════════════════════════════════

def predict_single_file(wav_path: str) -> dict:
    """
    Tek bir .wav dosyasını analiz eder ve sınıfı tahmin eder.
    Streamlit / Tkinter / Flask / Jupyter arayüzlerinden çağrılabilir.

    Kullanım:
        result = predict_single_file("Dataset/G01_D01_C_11_Angry_C3.wav")
        print(result)
        # {'dosya': 'G01_D01_C_11_Angry_C3.wav',
        #  'F0_mean_Hz': 290.5, 'F0_std_Hz': 22.1,
        #  'ZCR_per_sec': 4100.0, 'Enerji_mean': 0.003,
        #  'Tahmin': 'Çocuk'}
    """
    if not os.path.exists(wav_path):
        return {"hata": f"Dosya bulunamadı: {wav_path}"}
    try:
        signal, sr                         = load_audio(wav_path)
        f0_mean, f0_std, zcr_mean, energy  = extract_features(signal, sr)
        tahmin                             = rule_based_classify(f0_mean)
    except Exception as e:
        return {"hata": str(e)}

    return {
        "dosya"       : os.path.basename(wav_path),
        "F0_mean_Hz"  : round(f0_mean, 2)  if pd.notna(f0_mean)  else None,
        "F0_std_Hz"   : round(f0_std, 2)   if pd.notna(f0_std)   else None,
        "ZCR_per_sec" : round(zcr_mean, 2) if pd.notna(zcr_mean) else None,
        "Enerji_mean" : energy,
        "Tahmin"      : tahmin,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SES ANALİZİ – Dönem İçi Proje")
    print(f"  Kütüphane: {'librosa ✓' if HAS_LIBROSA else 'numpy/scipy (fallback)'}")
    print("=" * 60)

    # Tek dosya testi (isteğe bağlı):
    # print(predict_single_file("Dataset/G01_D01_C_11_Angry_C3.wav"))

    # Tam veri seti
    result_df = process_dataset(METADATA_EXCEL, DATASET_ROOT)
    statistics_and_plots(result_df)

    print("\n  --- Tüm çıktılar oluşturuldu ---")
    for fname in [OUTPUT_EXCEL, "autocorr_vs_fft.png",
                  "confusion_matrix.png", "f0_dagilim.png"]:
        ok = "✓" if os.path.exists(fname) else "—"
        print(f"    {ok}  {fname}")
