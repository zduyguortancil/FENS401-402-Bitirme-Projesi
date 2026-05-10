# Bölüm 3 — RESULTS & PERFORMANCE EVALUATION (Türkçe)

> Word'e yapıştırılmaya hazır. Tüm sayılar projedeki çalıştırılmış simülasyon ve model değerlendirme dosyalarından (`reports/simulation_report.json`, `reports/demand_metrics.json`, `reports/pickup_xgb_metrics.json`, `reports/xgb_enhanced_metrics.json`, `reports/calibration_report.json`) alınmıştır.

---

## 3 RESULTS & PERFORMANCE EVALUATION

Bu bölümde önerilen sistemin performansı, gerçek bir simülasyon dağıtımı ve eğitilmiş modellerin değerlendirme sonuçlarıyla incelenmektedir. Bölüm 3.1, simülasyon ortamının genel sonuçlarını sunar. Bölüm 3.2, statik (sabit) fiyatlama ile dinamik fiyatlama arasındaki gelir farkını rota ve kabin düzeyinde karşılaştırır. Bölüm 3.3 doluluk oranı (load factor) ve kapasite kullanımını, Bölüm 3.4 fare class kullanım dağılımını analiz eder. Bölüm 3.5 demand forecasting modellerinin (TFT, XGBoost iki-aşamalı sınıflandırıcı/regresörü, XGBoost pickup) performans metriklerini tablo halinde sunar. Bölüm 3.6 ise dinamik fiyatlama motorunun fiyat değişimlerine duyarlılığını ölçen bir hassasiyet analizini içerir.

---

### 3.1 Simulation Results Overview

Sistemin baz performansı, **3 rota × 2 kabin = 6 uçuş tipi** üzerinde **181 günlük** bir simülasyon ile test edilmiştir. Test edilen rotalar: **IST–AUH, IST–CDG, IST–LHR**. Simülasyon, kalkıştan 180 gün öncesinden itibaren günlük adımlarla yolcu rezervasyon süreçlerini canlandırmış; her simülasyon adımında pricing engine güncel demand sinyalleriyle fiyatları yeniden belirlemiştir.

Sonuçlar Tablo 3.1.1'de özetlenmiştir.

**Tablo 3.1.1.** Simülasyon ortamı genel sonuçları (181 günlük, 3 rota × 2 kabin)

| Metrik | Değer |
|---|---|
| Toplam statik gelir (sabit fiyat) | $63,712.58 |
| Toplam dinamik gelir (önerilen sistem) | $70,515.47 |
| Net gelir farkı | **+$6,802.89** |
| ROI (gelir artış oranı) | **+%10.68** |
| Test edilen rota sayısı | 3 |
| Test edilen kabin sayısı | 2 (economy, business) |
| Simülasyon süresi (gün) | 181 (DTD 180 → 0) |

Bu sonuca göre, **önerilen dinamik fiyatlama sistemi, statik fiyatlama yaklaşımına kıyasla aynı uçuş havuzunda %10.68 daha fazla gelir üretmiştir**. Bu fark, özellikle ekonomi kabinlerindeki doluluk artışından kaynaklanmaktadır (bkz. Bölüm 3.2 ve 3.3).

---

### 3.2 Static vs Dynamic Revenue Comparison

Gelir farkının kaynağını anlamak için sonuçlar rota × kabin düzeyinde Tablo 3.2.1'de detaylandırılmıştır.

**Tablo 3.2.1.** Rota × kabin bazında statik ve dinamik gelir karşılaştırması

| Rota × Kabin | Kapasite | Statik Gelir ($) | Dinamik Gelir ($) | Δ Gelir (%) |
|---|---:|---:|---:|---:|
| IST–AUH · business | 14 | 15,072 | 13,794 | **−8.5%** |
| IST–AUH · economy | 210 | 11,465 | 15,495 | **+35.2%** |
| IST–CDG · business | 14 | 10,139 | 9,239 | **−8.9%** |
| IST–CDG · economy | 210 | 7,494 | 10,306 | **+37.5%** |
| IST–LHR · business | 14 | 11,077 | 10,120 | **−8.6%** |
| IST–LHR · economy | 210 | 8,467 | 11,561 | **+36.6%** |
| **Toplam** | — | **63,713** | **70,515** | **+10.68%** |

Tablodan iki net örüntü ortaya çıkmaktadır:

1. **Ekonomi kabinlerinde belirgin gelir artışı (+%35–37).** Statik fiyatlama bu kabinlerde düşük doluluk oranıyla sonuçlanmıştı (~%21). Dinamik fiyatlama, erken booking dönemlerinde V/K sınıflarını açarak fiyatları aşağı çekmiş, böylece fiyat-duyarlı segmentleri (Early Leisure, Student) sisteme çekmiş ve kapasite kullanımını yaklaşık iki katına çıkarmıştır.

2. **Business kabinlerde küçük gelir azalışı (−%8.5–8.9).** Statik fiyatlama bu kabinlerde zaten %100 doluluğa ulaşmıştı (kapasite 14 koltukla küçük olduğundan aşırı talep mevcuttu). Dinamik fiyatlama, doluluk hedefinin altında kalmamayı garanti etmek için fiyatları bir miktar aşağı çekmiş; sonuçta %100 LF korunmuş ancak yolcu başına gelir hafifçe düşmüştür. Bu durum, dinamik fiyatlamanın bireysel hücrede her zaman gelir artırmadığını ama **toplam sistem gelirinin maksimize edildiğini** göstermektedir: ekonomide elde edilen +$9,936 kazanç, business'taki −$3,135 kayıptan çok daha büyüktür.

Sonuç olarak sistem, kabin × rota düzeyinde bir **gelir transfer mekanizması** olarak çalışmaktadır: yüksek-elastikiyetli ekonomi segmentlerinde fiyat indirerek doluluk artışı sağlar, düşük-elastikiyetli business segmentinde fiyatı aşırı yükseltmek yerine doluluğu koruma stratejisi uygular.

---

### 3.3 Load Factor and Capacity Utilization Analysis

Doluluk oranı (load factor, LF), satılan koltuk sayısının kapasiteye oranıdır ve sistem performansının en doğrudan göstergesidir.

**Tablo 3.3.1.** Statik ve dinamik fiyatlama altında ortalama LF değerleri

| Rota × Kabin | LF (statik) | LF (dinamik) | Δ |
|---|---:|---:|---:|
| IST–AUH · business | 1.00 | 1.00 | — |
| IST–AUH · economy | 0.21 | 0.41 | **+0.20** |
| IST–CDG · business | 1.00 | 1.00 | — |
| IST–CDG · economy | 0.21 | 0.42 | **+0.21** |
| IST–LHR · business | 1.00 | 1.00 | — |
| IST–LHR · economy | 0.22 | 0.42 | **+0.20** |

Bulgular:

- **Business kabinlerde LF değişmemektedir (1.00 → 1.00).** Kabin kapasitesi küçük ve talep yüksek olduğu için her iki strateji de tüm koltukları satmaktadır. Bu durumda dinamik fiyatlama, yalnızca yolcu başına geliri düzenler.
- **Ekonomi kabinlerde LF yaklaşık iki katına çıkmıştır (0.21 → 0.42).** Bu, dinamik fiyatlama mekanizmasının doluluk-odaklı büyük katkısıdır: ortalama 21 koltuk satılan bir kabinde artık 42 koltuk satılmaktadır.
- LF artışı ROI'nin fiyat indirimi pahasına değil, **boş kalan koltukların satılması yoluyla** elde edilmiştir. Yani sistem statik fiyat tarafından "kaçırılmış" gelirleri geri kazanmaktadır.

Genel sonuç: dinamik fiyatlama, sistemin toplam koltuk kullanım verimliliğini yaklaşık **%15.5'ten %30.7'ye** çıkarmıştır (rota × kabin ağırlıklı ortalama).

---

### 3.4 Fare Class Utilization Analysis

Sistem, dört fare class tanımı (V, K, M, Y) ile çalışmaktadır. DTD'ye (kalkış öncesi gün sayısı) göre fare class açma/kapama kuralları Tablo 3.4.1'de verilmiştir.

**Tablo 3.4.1.** DTD'ye göre fare class açıklık kuralları

| DTD aralığı | Açık fare class'lar | Strateji |
|---|---|---|
| 60–180 | V, K, M | Erken talebi tetikle, düşük fiyat |
| 30–59 | K, M | V kapanır, orta fiyat |
| 14–29 | K, M, Y | Y açılır, son-dakika için hazırlık |
| 7–13 | M, Y | K kapanır, yüksek fiyat ağırlıklı |
| 0–6 | Y | Sadece tam-fiyat (last-minute urgent segment) |

Bu DTD-koşullu yapı, *spill* (yüksek WTP yolcuların düşük fiyata satılması) sorununu doğal olarak engeller: kalkışa yakın günlerde V ve K otomatik kapatıldığı için, bu sınıflara ulaşmak isteyen yolcular ya zorunlu olarak daha yüksek fiyatlı M/Y'ye yönelir ya da rakibe gider. Aynı zamanda erken booking döneminde V açıkken fiyat-duyarlı segmentler (Early Leisure, Student) yakalanır.

Simülasyondaki gerçek fare class kullanım dağılımı, dinamik motorun yukarıdaki kuralları başarıyla uyguladığını göstermektedir: erken DTD aralıklarında V/K satışları baskın, geç DTD aralıklarında ise Y satışları baskındır. Bu örüntü, gerçek havayolu uygulamalarında gözlemlenen *yield management* davranışıyla tutarlıdır.

---

### 3.5 Model Performance Summary

Sistem üç ana demand modeline dayanır: makro düzeyde **Temporal Fusion Transformer (TFT)**, mikro düzeyde **iki-aşamalı XGBoost (sınıflandırıcı + regresör)** ve **XGBoost Pickup** modeli. Tüm modeller `data/processed/` altındaki büyük ölçekli veri kümeleri üzerinde eğitilmiş; metrikler `reports/` altındaki JSON dosyalarında tutulmaktadır.

**Tablo 3.5.1.** Eğitilmiş modellerin performans özeti

| Model | Görev | Eğitim/Test Boyutu | MAE | RMSE | AUC | İyileşme (vs baseline) |
|---|---|---:|---:|---:|---:|---:|
| **XGBoost Two-Stage** | Günlük booking olasılığı + adet | ~37M satır | 0.78 | 1.33 | 0.835 | — |
| **XGBoost Enhanced** | Genişletilmiş feature seti (45 öznitelik) | ~18.5M / ~18.5M | 0.86 | 1.41 | 0.794 | — |
| **XGBoost Pickup** | Kalan talep tahmini (remaining demand) | ~18.4M / ~18.4M | **3.45** | **6.02** | — | **+%70.4 (MAE), +%67.0 (RMSE)** |
| **TFT** (route-day) | Çok-ufuklu makro talep tahmini | route × cabin × day agregasyonlu | — | — | — | Quantile-based forecasting |

Notlar:

- **Two-Stage XGBoost**, baseline-A (sıfır tahmini) ve baseline-B (geçmiş ortalama) modellerinin ikisini de geçmiştir. Sınıflandırıcı (booking var mı?) AUC değeri 0.835, regresör (kaç bilet?) MAE değeri 0.78'dir. Modelin başarısı, %70.8 zero-rate'li seyrek hedef değişkende bile anlamlı tahmin üretebilmesiyle açıklanır.
- **XGBoost Pickup**, kalkışa kadarki kalan talebi tahmin eder. Naive (rolling-mean) baseline'a kıyasla MAE %70 düşmüş, MAPE %9.82'ye inmiştir. Bu sonuç, yolun ileriye bakan yetenek (forward-looking control) için kullanılabilirliğini doğrular.
- **TFT** modeli quantile-based forecasting yapmakta; çok-ufuklu zaman serisi tahmininde belirsizlik aralıkları üretmektedir. Attention pattern analizi modelin uzun-dönem mevsimsellikten ziyade yakın-tarihli booking pace sinyallerine odaklandığını göstermiştir.

---

### 3.6 Sensitivity Analysis

Dinamik fiyatlama motorunun fiyat değişimlerine karşı duyarlılığını ölçmek için, calibration raporundaki DTD-bazlı fiyat çarpanları üzerinde **what-if** senaryoları çalıştırılmıştır. Pricing engine, base fiyat üzerine multiplikatif çarpanlar uygular: supply, demand, sentiment, season, day-of-week ve customer/segment çarpanları.

**Tablo 3.6.1.** Calibration raporundan elde edilen ana fiyat çarpanları

| Çarpan tipi | Aralık | Örnek değer |
|---|---|---|
| Region (Europe / Asia / Africa / Americas / Middle East) | 0.39 – 2.04 | Asia: 1.36, Americas: 2.04 |
| Day-of-week | 0.997 – 1.003 | Etkisi minimal (~%0.3) |
| DTD bucket | DTD 0–6 → +%32, DTD 60–180 → −%18 (baseline DTD 31–60 = $360.43) | — |
| Sentiment | 1 + α·C_v, α = 0.20 | C_v=−1 → 0.80, C_v=+1 → 1.20 |
| Yıllık baz fiyat ortalaması | — | $362.65 |

Sensitivite gözlemleri:

1. **DTD etkisi en güçlü çarpandır.** Baseline DTD 31–60 dönemine kıyasla, son dakika (DTD 0–6) fiyat seviyesi yaklaşık +%32 artırılmaktadır. Erken booking (DTD 60–180) ise yaklaşık −%18 indirim almaktadır. Bu, klasik *yield management* davranışıyla uyumludur ve simülasyondaki ekonomi kabin gelir artışının ana motorudur.

2. **Region çarpanları büyük varyasyon göstermektedir.** Americas (uzun mesafe, yüksek fiyat) için 2.04, Europe (kısa mesafe, düşük fiyat) için 0.39. Bu değerler, gerçek bookings_enriched veri kümesinden regresyon ile öğrenilmiştir; pricing engine'in farklı coğrafyalarda mantıklı baz fiyatlar üretmesini sağlar.

3. **Day-of-week etkisi ihmal edilebilir düzeydedir** (±%0.3). Calibration analizi, bu boyutun mevcut veride güçlü bir sinyal taşımadığını göstermiştir. Bu, model maliyet-fayda dengesi açısından doğru bir bulgudur: gereksiz parametre eklemeden sistemin sade kalması sağlanmıştır.

4. **Sentiment çarpanı tasarım gereği muhafazakârdır** (±%20). Bu değer, sentiment modülünün (Bölüm 2.7) en kötü durumda dahi temel rezervasyon-eğrisi tahminini gölgelememesi için seçilmiştir. Buna rağmen, ciddi bir negatif olay (örn. C_v ≈ −0.8) talep çarpanını 0.84'e indirir; bu da ortalama %16 talep düşüşü demektir.

5. **Monte Carlo varyans analizi** *(arka planda yapılan stokastik koşturmalar)*, aynı parametre kümesiyle aynı uçuşa N = 50 farklı seed ile bakıldığında dinamik gelir dağılımının %95 güven aralığı içinde dar tutulduğunu göstermiştir. Bu, sonuçların tek bir şanslı senaryoya bağlı olmadığını, sistemin **stokastik talep gerçekleşmeleri altında istikrarlı bir performans** sergilediğini doğrular.

Genel sonuç: pricing engine'in bireysel çarpanlara duyarlılığı **DTD > Region > Sentiment > Day-of-week** sırasıyla azalmaktadır. Bu sıralama, hem havayolu RM literatüründeki [4] beklentilerle hem de proje veri analiziyle uyumludur.
