# scripts/

Veri hazirlama, model egitimi ve Kaggle scriptleri.

## data_prep/
Ham veriden islenmis veri ureten scriptler.

| # | Dosya | Girdi | Cikti | Aciklama |
|---|-------|-------|-------|----------|
| 1 | build_demand_training.py | raw/*.parquet | processed/demand_training.parquet, flight_metadata.parquet | 37M satirlik DTD bazli egitim verisi + ucus metadata |
| 2 | add_event_tags.py | processed/demand_training.parquet | (demand_training'e tag ekler) | 15 olay etiketi (ramazan, bayram, tatil, vb.) |
| 3 | build_tft_route_daily.py | raw/ (OneDrive/veriler/) | processed/tft_route_daily.parquet | TFT icin rota bazli gunluk agregat (138K satir) |
| 4 | build_pickup_master.py | processed/demand_training + flight_metadata + tft_route_daily | processed/pickup_master.parquet | XGBoost pickup model master tablosu (36.8M satir, 54 kolon) |
| 5 | build_tft_dataset.py | raw/*.parquet | processed/tft_dataset.parquet | TFT icin eski ucus bazli dataset (kullanilmiyor) |
| 6 | build_demand_functions.py | processed/demand_training.parquet | reports/demand_functions_report.json | Talep fonksiyonlari analizi |
| 7 | build_passenger_clusters.py | raw/bookings_enriched.parquet | processed/passenger_clusters.parquet | Yolcu kumeleme (K-means) |
| 8 | flight_snapshot.py | - | raw/flight_snapshot*.parquet | Ucus snapshot olusturma |
| 9 | check_lf.py | - | - | Load factor kontrolu |

## training/
Model egitim scriptleri.

| Dosya | Model | Sonuc | Durum |
|-------|-------|-------|-------|
| train_pickup_xgb.py | XGBoost Pickup | MAE=3.45, MAPE=%9.8, +%70.4 vs baseline | AKTIF — lokal calistirilir (16GB RAM yeterli) |
| train_tft_model.py | TFT (local) | - | Kaggle'da calistirildi, lokal versiyonu referans |
| train_demand_model.py | XGBoost (eski) | MAE=0.780, AUC=0.835 | ESKI — baseline ile ayni, kullanilmiyor |
| train_xgb_enhanced.py | XGBoost (enhanced) | MAE=0.855 | BASARISIZ — baseline'dan kotu |

## kaggle/
Kaggle notebook olarak calistirilmak uzere yazilmis scriptler.

| Dosya | Sonuc | Aciklama |
|-------|-------|----------|
| kaggle_tft_route.py | MAPE=%5.9 | TFT rota bazli egitim (BASARILI, aktif model) |
| kaggle_tft_notebook.py | - | TFT ucus bazli deneme (basarisiz, entity sorunu) |
| kaggle_xgb_enhanced.py | MAE=0.855 | Enhanced XGBoost (basarisiz) |

## Calistirma Sirasi (sifirdan)
```bash
# 1. Egitim verisi olustur
python scripts/data_prep/build_demand_training.py

# 2. Rota gunluk agregat
python scripts/data_prep/build_tft_route_daily.py

# 3. Pickup master tablo
python scripts/data_prep/build_pickup_master.py

# 4. XGBoost pickup egitimi
python scripts/training/train_pickup_xgb.py

# 5. TFT egitimi (Kaggle'da)
# kaggle_tft_route.py'yi Kaggle notebook olarak calistir
```
