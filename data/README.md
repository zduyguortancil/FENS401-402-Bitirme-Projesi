# data/

Projenin tum verileri bu klasorde bulunur.

## raw/
Ham kaynak veriler. Degistirilmez, sadece okunur.

| Dosya | Boyut | Aciklama |
|-------|-------|----------|
| bookings_enriched.parquet | 341 MB | Booking bazli veri (fare, channel, DTD, event tags) |
| flight_snapshot.parquet | 453 MB | Ucus snapshot v1 (eski format) |
| flight_snapshot_v2.parquet | 205 MB | Ucus snapshot v2 (ticket + ancillary revenue) |

## processed/
Scriptler tarafindan uretilmis, modelleme icin hazir veriler.

| Dosya | Boyut | Satir | Ureten Script | Aciklama |
|-------|-------|-------|---------------|----------|
| demand_training.parquet | 107 MB | 36.9M | build_demand_training.py | DTD bazli egitim verisi (flight x DTD, 51 kolon) |
| flight_metadata.parquet | 1.4 MB | 204K | build_demand_training.py | Ucus metadata (rota, mesafe, sure, kapasite) |
| pickup_master.parquet | ~200 MB | 36.8M | build_pickup_master.py | XGBoost pickup model master tablosu (54 kolon) |
| tft_route_daily.parquet | 6.8 MB | 138K | build_tft_route_daily.py | Rota bazli gunluk agregat (TFT girdi, 42 kolon) |
| tft_dataset.parquet | 22 MB | - | build_tft_dataset.py | TFT icin eski dataset (kullanilmiyor) |
| tft_predictions.parquet | 3.4 MB | 691K | kaggle_tft_route.py | TFT model ciktisi (actual/predicted) |
| passenger_clusters.parquet | 5.1 MB | - | build_passenger_clusters.py | Yolcu kumeleme sonuclari (K-means) |
| demand_predictions_test_sample.parquet | 8 MB | - | train_demand_model.py | Eski XGBoost test tahminleri |
| _tmp_test.parquet | - | - | train_pickup_xgb.py | Gecici dosya (silinebilir) |

## models/
Egitilmis model dosyalari ve feature listeleri.

### Aktif Modeller
| Dosya | Aciklama |
|-------|----------|
| pickup_xgb.json | XGBoost pickup model (MAE=3.45, MAPE=%9.8) |
| pickup_feature_list.json | Pickup model feature listesi (49 feature) |
| tft_route_daily_model.pt | TFT rota bazli model (MAPE=%5.9) |

### Eski Modeller (referans, kullanilmiyor)
| Dosya | Aciklama |
|-------|----------|
| xgb_demand_classifier.pkl | Eski XGBoost classifier (sale yes/no) |
| xgb_demand_regressor.pkl | Eski XGBoost regressor |
| xgb_enhanced_classifier.pkl | Enhanced XGBoost classifier (basarisiz) |
| xgb_enhanced_regressor.pkl | Enhanced XGBoost regressor (basarisiz) |
| feature_list.json | Eski XGBoost feature listesi |
| enhanced_feature_list.json | Enhanced XGBoost feature listesi |
