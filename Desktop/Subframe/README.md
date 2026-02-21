# 📈 Sinüzoidal İşaretlerin Örneklenmesi ve Görselleştirilmesi

Bu proje, sinüzoidal işaretlerin bilgisayar ortamında örneklenerek nasıl
represent edildiğini incelemek amacıyla hazırlanmıştır. Ödev kapsamında
different frekanslardaki sinüzoidal işaretler üretilmiş, Nyquist örnekleme
theoremi dikkate alınarak örneklenmiş ve zaman domeninde görselleştirilmiştir.

## 🎯 Projenin Amaçları

- Sürekli zamanlı (analog) sinyallerin bilgisayarda doğrudan gösterilemeyeceğini kavramak
- Analog sinyallerin ancak örnekleme (sampling) ile dijital ortama aktarılabildiğini göstermek
- Örnekleme frekansının sinyal kalitesi üzerindeki etkisini incelemek
- Nyquist örnekleme teoremini uygulamalı olarak öğrenmek
- Birden fazla sinüzoidal işaretin toplanmasıyla oluşan karma sinyali gözlemlemek

## 📌 Temel Frekans (f₀) Seçimi
Her grup için temel frekans **f₀**, grup üyelerinin okul numaralarının son iki
hanesinin toplanmasıyla belirlenmiştir. Örnek:

- Öğrenci 1 → 24
- Öğrenci 2 → 29
- Öğrenci 3 → 47

Toplam: 24 + 29 + 47 = 100 → **f₀ = 100 Hz**

(Not: Önceki dokümanda 40 Hz yazılmıştı, bu matematiksel olarak yanlıştı.)

Bu yöntem her grubun farklı parametrelerle çalışmasını sağlar.

## 📐 Üretilen Sinyaller

Hesaplanan **f₀** kullanılarak üç farklı sinüzoidal işaret üretilmiştir:

```
f₁ = f₀
f₂ = f₀ / 2
f₃ = 10 f₀
```

Her sinyal için matematiksel ifade:

```
x(t) = sin(2π f t)
```

## ⚙️ Örnekleme Frekansı ve Nyquist Teoremi

Bilgisayar ortamında sinyaller ayrık zamanlı olarak temsil edilir. Bu nedenle
örnekleme gerekir. Nyquist örnekleme teoremine göre:

```
f_s ≥ 2 f_max
```

Bu çalışmada en yüksek frekans:

```
f_max = f₃ = 10 f₀
```

Dolayısıyla minimum örnekleme frekansı:

```
f_s ≥ 20 f₀
```

Grafiklerin daha düzgün elde edilmesi ve aliasing oluşmaması için örnekleme
frekansı güvenli tarafta seçilmiş, **f_s = 50 f₀** olarak belirlenmiştir. Bu seçim
Nyquist kriterini fazlasıyla sağlamaktadır ve özellikle yüksek frekanslı f₃
sinyalinde bozulmayı engeller.

## ⏱️ Zaman Penceresi Seçimi

Her sinyalin en az 3 tam periyodunun gözlemlenebilmesi için zaman ekseni
dinamik olarak ayarlanmıştır. Bir sinyalin periyodu:

```
T = 1 / f
```

Bu nedenle her sinyal için zaman aralığı:

```
t ∈ [0, 3T]
```

şeklinde belirlenmiştir.

## ➕ Sinyal Toplama

Üç sinüzoidal işaret toplanarak karma bir sinyal elde edilmiştir:

```
x_toplam(t) = x₁(t) + x₂(t) + x₃(t)
```

Bu işlem, gerçek hayatta karşılaşılan çok bileşenli sinyallerin nasıl oluştuğunu
göstermek amacıyla yapılmıştır.

---

# 📞 DTMF (Dual-Tone Multi-Frequency) Sinyal Üretimi

Telefon tuş takımında kullanılan DTMF sistemini sayısal sinyal işleme
prensipleriyle modelleyen bir uygulama geliştirilmiştir. Kullanıcı etkileşimli
bir arayüz üzerinden tuşa basıldığında, ilgili DTMF sinyali üretilir, zaman
domeninde görselleştirilir ve hoparlörden ses olarak çalınır.

## 🎯 DTMF Projesinin Amaçları

- İki sinüzoidal işaretin toplanmasıyla anlamlı bilgi üretildiğini göstermek
- Telefon tuş seslerinin matematiksel modelini uygulamak
- Nyquist örnekleme teoremine dijital ses üretiminde yer vermek
- Kullanıcı etkileşimli bir GUI geliştirmek
- Üretilen sinyali hem grafik hem ses olarak sunmak

## 📌 DTMF Nedir?

DTMF sisteminde her telefon tuşu, biri düşük diğeri yüksek frekans grubundan
seçilen iki sinüzoidal sinyalin toplamı ile temsil edilir:

```
x(t) = sin(2π f_low t) + sin(2π f_high t)
```

İki sinyalin toplamı maksimum ±2 genliğe ulaşabileceği için clipping oluşmaması
amaçlı şu şekilde ölçeklendirilir:

```
x(t) = 0.5 [ sin(2π f_low t) + sin(2π f_high t) ]
```

### 📊 DTMF Frekans Tablosu

|       | 1209 | 1336 | 1477 | 1633 |
|-------|------|------|------|------|
| 697   | 1    | 2    | 3    | A    |
| 770   | 4    | 5    | 6    | B    |
| 852   | 7    | 8    | 9    | C    |
| 941   | *    | 0    | #    | D    |

## ⚙️ Örnekleme Frekansı Seçimi

DTMF sisteminde en yüksek frekans 1633 Hz’dir. Nyquist teoremine göre:

```
f_s ≥ 2 × 1633 ≈ 3266 Hz
```

Bu nedenle uygulamada **f_s = 8000 Hz** seçilmiştir. Bu değer hem Nyquist
kriterini sağlamaktadır hem de telekomünikasyon sistemlerinde kullanılan
standart örnekleme frekansıdır.

## ⏱️ Sinyal Süresi

Her tuş basımı için sinyal süresi:

```
T = 0.25 saniye
```

Örnek sayısı:

```
N = f_s × T
```

Bu süre, DTMF tonunun net ve anlaşılır duyulması için yeterlidir.

## 🖥️ Uygulama Özellikleri

- Python + Tkinter ile telefon tuş takımı arayüzü
- Tuşa basılınca:
  - İlgili DTMF sinyali üretilir
  - Zaman domeninde grafik çizilir
  - Hoparlörden ses çıkarılır
  - (Opsiyonel) FFT ile frekans domeni analizi yapılır

## 🛠️ Kullanılan Teknolojiler

- Python 3
- NumPy
- Matplotlib
- sounddevice
- Tkinter

## ▶️ Kurulum

Komut satırında aşağıdaki komut çalıştırılmalıdır:

```bash
pip install numpy matplotlib sounddevice
```

*Bu README, proje detaylarını açıklamak ve çalışma yönergeleri sunmak
amacıyla hazırlanmıştır.*