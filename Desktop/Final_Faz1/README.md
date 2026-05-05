# Duygu Analizi Ensemble

Ses dosyalarından MFCC, pitch/F0, sıfır geçiş oranı (ZCR), kısa süreli enerji (STE), mel ve spektral öznitelikler çıkararak 5 duygu sınıfı için sınıflandırma yapan bir proje. Eğitim tarafında Random Forest, ExtraTrees ve Gradient Boosting modellerinin yer aldığı bir `VotingClassifier` ensemble yapısı kullanılır.

## İçerik
- [Kurulum](#kurulum)
- [Proje Yapısı](#proje-yapısı)
- [Kullanım](#kullanım)
- [Çıktılar](#çıktılar)
- [Notlar](#notlar)
- [Lisans](#lisans)

## Kurulum

1. Python 3.10+ ile bir sanal ortam oluşturun ve etkinleştirin.
2. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

3. `birlesik_metadata.xlsx` dosyasının ve `Dataset/` klasörünün proje kökünde bulunduğundan emin olun.

## Proje Yapısı

```text
proje_klasörü/
├── arayuz.py
├── ses_analizi.py
├── README.md
├── requirements.txt
├── birlesik_metadata.xlsx
└── Dataset/
    └── ...
```

## Kullanım

### Streamlit Arayüzü

Arayüzü başlatmak için:

```bash
streamlit run arayuz.py
```

Arayüzde iki bölüm bulunur:

- Tek bir WAV dosyası yükleyip tahmin alma
- Metadata ve veri klasörü üzerinden toplu eğitim, değerlendirme ve raporlama

### Komut Satırı

Toplu analiz ve çıktı üretimi için:

```bash
python ses_analizi.py
```

Bu akış metadata dosyasını okur, `Dataset/` içindeki sesleri eşleştirir, öznitelikleri çıkarır, modeli eğitir ve sonuç dosyalarını üretir.

## Çıktılar

Çalıştırma sonunda aşağıdaki dosyalar üretilebilir:

- `duygu_sonuclari.xlsx`
- `feature_importance.xlsx`
- `hatalar.xlsx`
- `confusion_matrix.png`
- `f0_dagilim.png`
- `autocorr_vs_fft.png`

## Notlar

- Desteklenen sınıflar: `nötr`, `mutlu`, `öfkeli`, `üzgün`, `şaşkın`
- Kod, metadata içindeki dosya adlarını toleranslı biçimde eşleştirir; dosya adlarında son ek veya küçük yazım farkları olsa da çalışacak şekilde tasarlanmıştır.
- Arayüz tarafında model durumu bellekte tutulur; sayfayı yenilerseniz yeniden eğitim gerekebilir.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.
