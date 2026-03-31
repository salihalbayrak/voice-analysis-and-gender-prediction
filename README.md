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
- **confusion_matrix.png:** Karışıklık matrisi.
- **f0_dagilim.png:** F0 dağılımı ve kutu grafiği.
- **hatalar.xlsx:** Yanlış tahmin edilen dosyalar.

---

## Sıkça Sorulan Sorular

- **F0 aralığı neden çok yüksek çıkıyor?**
  - Outlier (uç değer) ve bozuk dosyalar filtrelenmelidir. Kodda 60–400 Hz arası değerler dikkate alınır.
- **Çocuk sınıfı için üst sınır var mı?**
  - Evet, 400 Hz üzeri değerler "Tanımsız" olarak atanır.
- **Arayüzde tablo neden boş?**
  - Metadata veya Dataset yolu yanlışsa ya da dosya formatı uyumsuzsa tablo boş kalabilir.

---

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.

---

Daha fazla bilgi ve güncellemeler için: [GitHub Proje Sayfası](https://github.com/salihalbayrak/voice-analysis-and-gender-prediction)
