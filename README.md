# Ses Analizi ve Cinsiyet Tahmini Projesi

Bu proje, ses dosyalarından temel frekans (F0), sıfır geçiş oranı (ZCR) ve kısa süreli enerji (STE) gibi öznitelikleri çıkararak, otokorelasyon tabanlı yöntemle cinsiyet (Erkek/Kadın/Çocuk) tahmini yapar. Ayrıca, Streamlit tabanlı bir arayüz ile tek dosya ve toplu veri seti analizi sunar.

## İçerik
- [Kurulum](#kurulum)
- [Klasör Yapısı](#klasör-yapısı)
- [Kullanım](#kullanım)
  - [Arayüz ile Analiz](#arayüz-ile-analiz)
  - [Komut Satırı ile Analiz](#komut-satırı-ile-analiz)
- [Fonksiyonlar ve Açıklamalar](#fonksiyonlar-ve-açıklamalar)
- [Çıktılar ve Raporlama](#çıktılar-ve-raporlama)
- [Sıkça Sorulan Sorular](#sıkça-sorulan-sorular)
- [Lisans](#lisans)

---

## Kurulum

1. **Gerekli Python Paketleri:**

```bash
pip install numpy scipy pandas matplotlib openpyxl scikit-learn soundfile librosa streamlit
```

2. **Proje Dosyalarını İndir:**

- Tüm dosyaları bu repodan indirin veya klonlayın.

3. **Klasör Yapısı:**

```
proje_klasörü/
├── arayuz.py
├── birlestir.py
├── ses_analizi.py
├── Dataset/
│   ├── ... (ses dosyaları ve alt klasörler)
│   └── ...
├── birlesik_metadata.xlsx
└── ...
```

---

## Kullanım

### Arayüz ile Analiz

1. **Streamlit Arayüzünü Başlat:**

```bash
streamlit run arayuz.py
```

2. **Tek Dosya Analizi:**
   - "Tek Dosya Analizi" sekmesinde bir WAV dosyası yükleyin.
   - Otokorelasyon ve FFT grafikleri, F0, ZCR, enerji ve cinsiyet tahmini ekranda gösterilir.

3. **Veri Seti Analizi:**
   - "Veri Seti Değerlendirmesi" sekmesinde bir metadata Excel dosyası (.xlsx) ve Dataset klasörünü seçin.
   - Tüm veri seti için toplu analiz, istatistiksel tablo, karışıklık matrisi ve başarı oranı ekranda gösterilir.

### Komut Satırı ile Analiz

```bash
python ses_analizi.py
```
- Tüm Dataset klasörü ve metadata dosyası üzerinden toplu analiz ve çıktı dosyaları oluşturulur.

---

## Fonksiyonlar ve Açıklamalar

- **Otokorelasyon ile F0 Tespiti:**
  - Zaman düzleminde otokorelasyon fonksiyonu ile temel frekans (F0) bulunur.
- **Kural Tabanlı Sınıflandırıcı:**
  - F0 değerine göre Erkek/Kadın/Çocuk sınıflandırması yapılır.
  - Eşikler arayüzden ayarlanabilir.
- **Öznitelik Çıkarımı:**
  - F0, ZCR, enerji gibi öznitelikler her dosya için hesaplanır.
- **İstatistiksel Tablo:**
  - Her sınıf için ortalama F0, standart sapma, F0 aralığı ve başarı oranı hesaplanır.
- **Karışıklık Matrisi:**
  - Sınıflandırma doğruluğu için karışıklık matrisi görselleştirilir.

---

## Çıktılar ve Raporlama

- **sonuclar.xlsx:** Her dosya için öznitelikler ve tahminler.
- **autocorr_vs_fft.png:** Otokorelasyon ve FFT grafiği.
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
├── README.md
├── requirements.txt
├── .gitignore
├── arayuz.py
├── ses_analizi.py
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
