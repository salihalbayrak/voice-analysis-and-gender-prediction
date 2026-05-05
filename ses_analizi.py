
# ─────────────────────────────────────────────────────────────
# BAĞIMLILIKLAR
# ─────────────────────────────────────────────────────────────
import os
import glob
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI olmayan ortamlarda da çalışır (sunucu, Colab)
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split

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
METADATA_EXCEL  = "birlesik_metadata.xlsx" # Birleşik metadata dosyası
DATASET_ROOT    = "Dataset"                 # .wav klasörü
OUTPUT_EXCEL    = "duygu_sonuclari.xlsx"    # Sonuç tablosu

FRAME_MS        = 25      # Pencere uzunluğu (ms)  — yönerge: 20-30 ms
HOP_MS          = 10      # Adım boyutu (ms)
F0_MIN          = 60.0    # Hz – arama aralığı alt sınır
F0_MAX          = 500.0   # Hz – arama aralığı üst sınır

STE_THRESH_FAC  = 0.10    # Enerji eşiği: maks enerjinin %10'u
ZCR_VOICED_THR  = 0.30    # ZCR > bu eşik → sessiz bölge kabul

EMOTION_LABELS = ["nötr", "mutlu", "öfkeli", "üzgün", "şaşkın"]
EMOTION_MAP = {
    "neutral": "nötr",
    "nötr": "nötr",
    "notr": "nötr",
    "happy": "mutlu",
    "mutlu": "mutlu",
    "furious": "öfkeli",
    "angry": "öfkeli",
    "öfke": "öfkeli",
    "öfkeli": "öfkeli",
    "ofkeli": "öfkeli",
    "sad": "üzgün",
    "üzgün": "üzgün",
    "uzgun": "üzgün",
    "shocked": "şaşkın",
    "surprised": "şaşkın",
    "surprise": "şaşkın",
    "şaşkın": "şaşkın",
    "saskin": "şaşkın",
    "saskin": "şaşkın",
    "sasirma": "şaşkın",
    "sarsirma": "şaşkın",
    "sackin": "şaşkın",
}

FEATURE_GROUP_PREFIXES = {
    "Pitch": "Pitch",
    "F0": "Pitch",
    "ZCR": "ZCR",
    "Energy": "Energy",
    "STE": "Energy",
    "MFCC": "MFCC",
    "Delta_MFCC": "Delta MFCC",
    "Delta2_MFCC": "Delta2 MFCC",
    "Mel": "Mel",
    "Chroma": "Chroma",
    "Spectral": "Spectral",
}




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

def normalize_label(raw_label: str, yas=None, filename: str | None = None) -> str:
    """
    Duygu etiketlerini veri setinde geçen farklı yazımlardan
    tek bir standart sınıf adına çevirir.
    """
    def canonicalize(text: object) -> str:
        normalized = unicodedata.normalize("NFKD", str(text or ""))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        replacements = {
            "ş": "s",
            "ğ": "g",
            "ı": "i",
            "ç": "c",
            "ö": "o",
            "ü": "u",
            "â": "a",
            "î": "i",
            "û": "u",
        }
        for source_char, target_char in replacements.items():
            normalized = normalized.replace(source_char, target_char)
        return normalized

    tokens = [canonicalize(raw_label)]
    if filename:
        tokens.append(canonicalize(filename))
    if yas is not None and pd.notna(yas):
        tokens.append(canonicalize(yas))
    blob = " ".join(tokens)

    for key, target in EMOTION_MAP.items():
        if key in blob:
            return target
    return "Bilinmiyor"


def _discover_metadata_files(source: str) -> list[str]:
    path = Path(source)
    if path.is_dir():
        candidates = []

        combined = path / "birlesik_metadata.xlsx"
        if combined.exists():
            return [str(combined)]

        parent_combined = path.parent / "birlesik_metadata.xlsx"
        if parent_combined.exists():
            return [str(parent_combined)]

        candidates.extend(str(p) for p in path.rglob("*.xlsx"))
        return sorted(dict.fromkeys(candidates))
    if path.is_file():
        return [str(path)]
    return []


def load_metadata(path: str) -> pd.DataFrame:
    """
    Tek bir metadata dosyasını veya metadata klasörünü okur.
    Sütun ekler:
        Sinif : nötr / mutlu / öfkeli / üzgün / şaşkın
        Grup  : G01, G02, ...  (dosya adından çıkarılır)
    """
    metadata_files = _discover_metadata_files(path)
    if not metadata_files:
        raise FileNotFoundError(f"Metadata bulunamadı: {path}")

    frames = []
    for file_path in metadata_files:
        frame = pd.read_excel(file_path)
        frame.columns = [str(col).strip() for col in frame.columns]
        if "Dosya_Adi" not in frame.columns:
            continue

        duygu_col = next(
            (col for col in frame.columns if col.lower() in {"duygu", "emotion", "label", "sinif"}),
            None,
        )
        if duygu_col is None:
            frame["Sinif"] = frame["Dosya_Adi"].apply(normalize_label)
        else:
            frame["Sinif"] = frame.apply(
                lambda row: normalize_label(row[duygu_col], row.get("Yas"), row["Dosya_Adi"]),
                axis=1,
            )

        frame["Grup"] = frame["Dosya_Adi"].astype(str).str.extract(r"^(G\d+)", expand=False)
        frame["Metadata_Kaynak"] = os.path.basename(file_path)
        frames.append(frame)

    if not frames:
        raise ValueError(f"Geçerli metadata okunamadı: {path}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["Sinif"].isin(EMOTION_LABELS)].copy()
    return merged


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
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))
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


def _safe_stat(values: np.ndarray, reducer) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(reducer(values))


def _matrix_stats(matrix: np.ndarray, prefix: str) -> dict:
    matrix = np.asarray(matrix)
    return {
        f"{prefix}_{idx + 1:02d}_Mean": float(np.mean(matrix[idx]))
        for idx in range(matrix.shape[0])
    } | {
        f"{prefix}_{idx + 1:02d}_Std": float(np.std(matrix[idx]))
        for idx in range(matrix.shape[0])
    }


def extract_emotion_feature_vector(signal: np.ndarray, sr: int) -> dict:
    """
    Duygu tanıma için temel zaman ve frekans düzlemi öznitelikleri çıkarır.
    """
    fl = int(sr * FRAME_MS / 1000)
    hl = int(sr * HOP_MS / 1000)
    frames = make_frames(signal, fl, hl)
    ste = short_time_energy(frames)
    zcr = zero_crossing_rate(frames)
    mask = voiced_mask(ste, zcr)
    n = min(frames.shape[1], len(mask))

    f0_list = []
    for i in range(n):
        if mask[i]:
            f0 = autocorrelation_f0_frame(frames[:, i], sr)
            if not np.isnan(f0):
                f0_list.append(f0)

    f0_arr = np.asarray(f0_list, dtype=float)
    voiced_ratio = float(np.mean(mask[:n])) if n else np.nan

    features = {
        "Pitch_Mean": _safe_stat(f0_arr, np.mean),
        "Pitch_Std": _safe_stat(f0_arr, np.std),
        "Pitch_Min": _safe_stat(f0_arr, np.min),
        "Pitch_Max": _safe_stat(f0_arr, np.max),
        "Pitch_Range": _safe_stat(f0_arr, lambda x: np.max(x) - np.min(x)),
        "Voiced_Ratio": voiced_ratio,
        "ZCR_Mean": float(np.mean(zcr[:n] * sr / hl)) if n else np.nan,
        "ZCR_Std": float(np.std(zcr[:n] * sr / hl)) if n else np.nan,
        "Energy_Mean": _safe_stat(ste[:n][mask[:n]], np.mean),
        "Energy_Std": _safe_stat(ste[:n][mask[:n]], np.std),
    }

    if HAS_LIBROSA:
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=20)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        features.update(_matrix_stats(mfcc, "MFCC"))
        features.update(_matrix_stats(mfcc_delta, "Delta_MFCC"))
        features.update(_matrix_stats(mfcc_delta2, "Delta2_MFCC"))
        features.update(_matrix_stats(mel_db, "Mel"))
        features.update(_matrix_stats(chroma, "Chroma"))
        features["Spectral_Centroid_Mean"] = float(np.mean(spectral_centroid))
        features["Spectral_Bandwidth_Mean"] = float(np.mean(spectral_bandwidth))
    else:
        for idx in range(1, 14):
            features[f"MFCC_{idx:02d}_Mean"] = np.nan
            features[f"MFCC_{idx:02d}_Std"] = np.nan
            features[f"Delta_MFCC_{idx:02d}_Mean"] = np.nan
            features[f"Delta_MFCC_{idx:02d}_Std"] = np.nan
            features[f"Delta2_MFCC_{idx:02d}_Mean"] = np.nan
            features[f"Delta2_MFCC_{idx:02d}_Std"] = np.nan
        for idx in range(1, 21):
            features[f"Mel_{idx:02d}_Mean"] = np.nan
            features[f"Mel_{idx:02d}_Std"] = np.nan
        for idx in range(1, 13):
            features[f"Chroma_{idx:02d}_Mean"] = np.nan
            features[f"Chroma_{idx:02d}_Std"] = np.nan
        features["Spectral_Centroid_Mean"] = np.nan
        features["Spectral_Bandwidth_Mean"] = np.nan

    return features


def extract_features(signal: np.ndarray, sr: int):
    """
    Geriye dönük uyumluluk için özet öznitelikleri döndürür.
    """
    features = extract_emotion_feature_vector(signal, sr)
    return (
        features["Pitch_Mean"],
        features["Pitch_Std"],
        features["ZCR_Mean"],
        features["Energy_Mean"],
    )


def feature_columns() -> list[str]:
    columns = [
        "Pitch_Mean", "Pitch_Std", "Pitch_Min", "Pitch_Max", "Pitch_Range",
        "Voiced_Ratio", "ZCR_Mean", "ZCR_Std", "Energy_Mean", "Energy_Std",
        "Spectral_Centroid_Mean", "Spectral_Bandwidth_Mean",
    ]
    columns.extend([f"MFCC_{idx:02d}_Mean" for idx in range(1, 14)])
    columns.extend([f"MFCC_{idx:02d}_Std" for idx in range(1, 14)])
    columns.extend([f"Delta_MFCC_{idx:02d}_Mean" for idx in range(1, 14)])
    columns.extend([f"Delta_MFCC_{idx:02d}_Std" for idx in range(1, 14)])
    columns.extend([f"Delta2_MFCC_{idx:02d}_Mean" for idx in range(1, 14)])
    columns.extend([f"Delta2_MFCC_{idx:02d}_Std" for idx in range(1, 14)])
    columns.extend([f"Mel_{idx:02d}_Mean" for idx in range(1, 21)])
    columns.extend([f"Mel_{idx:02d}_Std" for idx in range(1, 21)])
    columns.extend([f"Chroma_{idx:02d}_Mean" for idx in range(1, 13)])
    columns.extend([f"Chroma_{idx:02d}_Std" for idx in range(1, 13)])
    return columns


def build_feature_table(metadata_source: str, dataset_root: str) -> pd.DataFrame:
    df = load_metadata(metadata_source)
    rows = []

    for _, row in df.iterrows():
        fname = str(row["Dosya_Adi"])
        # Try exact match first, then a few tolerant variants to handle
        # trailing underscores or small filename differences in the dataset.
        matches = glob.glob(os.path.join(dataset_root, "**", fname), recursive=True)
        if not matches:
            # wildcard after the name (catches extra underscore suffixes)
            matches = glob.glob(os.path.join(dataset_root, "**", fname + "*"), recursive=True)
        if not matches:
            direct = os.path.join(dataset_root, fname)
            if os.path.exists(direct):
                matches = [direct]
        if not matches:
            # try stripped trailing underscores
            alt = fname.rstrip('_')
            if alt != fname:
                matches = glob.glob(os.path.join(dataset_root, "**", alt + "*"), recursive=True)
        if not matches:
            # try adding an underscore suffix
            matches = glob.glob(os.path.join(dataset_root, "**", fname + "_" + "*"), recursive=True)

        if not matches:
            rows.append({
                **row.to_dict(),
                **{col: np.nan for col in feature_columns()},
                "Tahmin": "Bulunamadı",
                "Dogru_mu": False,
                "Dosya_Bulundu": False,
            })
            continue

        try:
            signal, sr = load_audio(matches[0])
            features = extract_emotion_feature_vector(signal, sr)
            rows.append({
                **row.to_dict(),
                **features,
                "Tahmin": np.nan,
                "Dogru_mu": np.nan,
                "Dosya_Bulundu": True,
            })
        except Exception as exc:
            print(f"  [HATA] {fname}: {exc}")
            rows.append({
                **row.to_dict(),
                **{col: np.nan for col in feature_columns()},
                "Tahmin": "Hata",
                "Dogru_mu": False,
                "Dosya_Bulundu": True,
            })

    return pd.DataFrame(rows)


def train_emotion_model(feature_df: pd.DataFrame):
    cols = [col for col in feature_columns() if col in feature_df.columns]
    data = feature_df[feature_df["Sinif"].isin(EMOTION_LABELS)].copy()
    # Exclude records where the audio file was not found to avoid
    # training/evaluation on placeholder rows with NaN features.
    if "Dosya_Bulundu" in data.columns:
        data = data[data["Dosya_Bulundu"] == True]
    data = data[data[cols].notna().sum(axis=1) >= max(4, len(cols) // 3)]
    data = data.dropna(subset=["Sinif"])

    if data.empty:
        raise ValueError("Eğitim için yeterli geçerli örnek bulunamadı.")

    X = data[cols].fillna(data[cols].median(numeric_only=True))
    y = data["Sinif"]

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 and len(class_counts) > 1 else None
    test_size = 0.2 if len(data) >= 10 else 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    rf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(random_state=42)

    model = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("gb", gb),
        ],
        voting="soft",
        n_jobs=-1,
        weights=[2, 2, 1],
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_columns": cols,
        "model": model,
    }
    return metrics


def aggregate_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
    elif hasattr(model, "estimators_"):
        collected = []
        for estimator in model.estimators_:
            if hasattr(estimator, "feature_importances_"):
                collected.append(np.asarray(estimator.feature_importances_, dtype=float))
        if not collected:
            importances = pd.Series(np.zeros(len(feature_cols), dtype=float), index=feature_cols)
        else:
            averaged = np.mean(collected, axis=0)
            importances = pd.Series(averaged, index=feature_cols)
    else:
        importances = pd.Series(np.zeros(len(feature_cols), dtype=float), index=feature_cols)
    grouped = {}
    for feature_name, importance in importances.items():
        group = next(
            (label for prefix, label in FEATURE_GROUP_PREFIXES.items() if feature_name.startswith(prefix)),
            "Other",
        )
        grouped.setdefault(group, 0.0)
        grouped[group] += float(importance)
    result = pd.DataFrame(
        sorted(grouped.items(), key=lambda item: item[1], reverse=True),
        columns=["Feature_Group", "Importance"],
    )
    return result


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
    3. MFCC / Pitch / ZCR / Enerji / Spektral özellikler çıkar
    4. Klasik bir modelle duygu sınıflandırması yap
    5. Sonuçları DataFrame olarak döndür ve Excel'e kaydet
    """
    df = load_metadata(metadata_excel)

    print(f"\n{'='*60}")
    print(f"  Toplam kayıt : {len(df)}")
    for label in EMOTION_LABELS:
        print(f"  {label:<11}: {(df['Sinif']==label).sum()}")
    print(f"  Dataset kök  : {os.path.abspath(dataset_root)}")
    print(f"{'='*60}\n")

    result_df = build_feature_table(metadata_excel, dataset_root)
    feature_cols = [col for col in feature_columns() if col in result_df.columns]
    model_info = train_emotion_model(result_df)
    model = model_info["model"]
    valid_mask = result_df["Sinif"].isin(EMOTION_LABELS)
    if "Dosya_Bulundu" in result_df.columns:
        valid_mask = valid_mask & (result_df["Dosya_Bulundu"] == True)
    valid_df = result_df[valid_mask & result_df[feature_cols].notna().any(axis=1)].copy()
    valid_df[feature_cols] = valid_df[feature_cols].fillna(valid_df[feature_cols].median(numeric_only=True))
    valid_df["Tahmin"] = model.predict(valid_df[feature_cols])
    valid_df["Dogru_mu"] = valid_df["Tahmin"] == valid_df["Sinif"]
    result_df.loc[valid_df.index, "Tahmin"] = valid_df["Tahmin"]
    result_df.loc[valid_df.index, "Dogru_mu"] = valid_df["Dogru_mu"]

    result_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n  -> Sonuçlar kaydedildi: {OUTPUT_EXCEL}")

    if not valid_df.empty:
        cm = confusion_matrix(valid_df["Sinif"], valid_df["Tahmin"], labels=EMOTION_LABELS)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_LABELS)
        disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
        ax.set_title("Karışıklık Matrisi (5 Duygu)", fontsize=12)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.close()
        print("  -> confusion_matrix.png kaydedildi")

        importance_df = aggregate_feature_importance(model, feature_cols)
        importance_df.to_excel("feature_importance.xlsx", index=False)
        print("  -> feature_importance.xlsx kaydedildi")

    first_row = result_df[result_df["Dosya_Bulundu"] == True].head(1)
    if not first_row.empty:
        first_fname = first_row.iloc[0]["Dosya_Adi"]
        matches = glob.glob(os.path.join(dataset_root, "**", str(first_fname)), recursive=True)
        if matches:
            first_sig, first_sr = load_audio(matches[0])
            plot_autocorr_vs_fft(first_sig, first_sr,
                                 label=str(first_fname),
                                 save_path="autocorr_vs_fft.png")

    return result_df


# ══════════════════════════════════════════════════════════════
# BÖLÜM 7 – İSTATİSTİK TABLOSU, CONFUSION MATRIX, GRAFİKLER
#           Yönerge Bölüm 5
# ══════════════════════════════════════════════════════════════

def statistics_and_plots(result_df: pd.DataFrame):
    """
    Yönerge Bölüm 5 tablosu:
        Sınıf | Örnek Sayısı | Ortalama Pitch | Std Sapma | Başarı (%)

    Ayrıca:
        - Sklearn classification report (Precision / Recall / F1)
        - Confusion Matrix (confusion_matrix.png)
        - Pitch histogram + kutu grafiği (f0_dagilim.png)
        - Hata analizi Excel (hatalar.xlsx)
    """
    df = result_df[
        result_df["Pitch_Mean"].notna() &
        result_df["Sinif"].isin(EMOTION_LABELS) &
        result_df["Tahmin"].isin(EMOTION_LABELS)
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
    colors = {
        "nötr": "slategray",
        "mutlu": "goldenrod",
        "öfkeli": "firebrick",
        "üzgün": "royalblue",
        "şaşkın": "darkviolet",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for sinif, clr in colors.items():
        vals = df[df["Sinif"] == sinif]["Pitch_Mean"].dropna()
        if len(vals):
            ax.hist(vals, bins=25, alpha=0.6, color=clr,
                    edgecolor="white", label=sinif)
    ax.set_xlabel("Pitch / F0 (Hz)", fontsize=11)
    ax.set_ylabel("Kayıt Sayısı", fontsize=11)
    ax.set_title("Pitch Histogramı — Sınıf Bazlı", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    box_data = [df[df["Sinif"] == s]["Pitch_Mean"].dropna().values
                for s in EMOTION_LABELS]
    bp = ax2.boxplot(box_data, tick_labels=EMOTION_LABELS,
                     patch_artist=True, notch=True)
    for patch, clr in zip(bp["boxes"], colors.values()):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Pitch / F0 (Hz)", fontsize=11)
    ax2.set_title("Pitch Kutu Grafiği", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Pitch Dağılımı – 5 Duygu", fontsize=13)
    plt.tight_layout()
    plt.savefig("f0_dagilim.png", dpi=150)
    plt.close()
    print("  -> f0_dagilim.png kaydedildi")

    # Hata Analizi
    hata_cols = [col for col in ["Dosya_Adi", "Sinif", "Tahmin", "Pitch_Mean", "ZCR_Mean", "Energy_Mean", "Duygu", "Yas", "gürültü seviyesi"] if col in df.columns]
    hatalar = df[~df["Dogru_mu"]][hata_cols]
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

def predict_single_file(wav_path: str, model=None, feature_cols: list[str] | None = None) -> dict:
    """
    Tek bir .wav dosyasını analiz eder ve sınıfı tahmin eder.
    Streamlit / Tkinter / Flask / Jupyter arayüzlerinden çağrılabilir.

    Kullanım:
        result = predict_single_file("Dataset/G10_D02_M_25_happy_C2.wav")
        print(result)
        # {'dosya': 'G10_D02_M_25_happy_C2.wav',
        #  'Pitch_Mean_Hz': 214.5, 'Pitch_Std_Hz': 22.1,
        #  'ZCR_Per_Sec': 4100.0, 'Energy_Mean': 0.003,
        #  'Tahmin': 'mutlu'}
    """
    if not os.path.exists(wav_path):
        return {"hata": f"Dosya bulunamadı: {wav_path}"}
    try:
        signal, sr                         = load_audio(wav_path)
        features                           = extract_emotion_feature_vector(signal, sr)
        tahmin                             = None
        if model is not None and feature_cols:
            frame = pd.DataFrame([features])[feature_cols].fillna(0)
            tahmin = model.predict(frame)[0]
    except Exception as e:
        return {"hata": str(e)}

    return {
        "dosya"       : os.path.basename(wav_path),
        "Pitch_Mean_Hz": round(features["Pitch_Mean"], 2) if pd.notna(features["Pitch_Mean"]) else None,
        "Pitch_Std_Hz" : round(features["Pitch_Std"], 2) if pd.notna(features["Pitch_Std"]) else None,
        "ZCR_Per_Sec"  : round(features["ZCR_Mean"], 2) if pd.notna(features["ZCR_Mean"]) else None,
        "Energy_Mean"  : features["Energy_Mean"],
        "Tahmin"       : tahmin,
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
    # print(predict_single_file("Dataset/G10_D02_M_25_happy_C2.wav"))

    # Tam veri seti
    result_df = process_dataset(METADATA_EXCEL, DATASET_ROOT)
    statistics_and_plots(result_df)

    print("\n  --- Tüm çıktılar oluşturuldu ---")
    for fname in [OUTPUT_EXCEL, "autocorr_vs_fft.png",
                  "confusion_matrix.png", "f0_dagilim.png"]:
        ok = "✓" if os.path.exists(fname) else "—"
        print(f"    {ok}  {fname}")
