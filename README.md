# SeatWise

**End-to-end airline revenue management system** combining machine learning demand forecasting, dynamic pricing, fare class optimization, and real-time booking simulation.

Built for the IST (Istanbul) hub network: 50 routes, 51 destinations, 5 regions, 2 cabin classes, ~146K daily observations.

## System Architecture

```
                    +─────────────────────────────────+
                    |        DASHBOARD (Flask)         |
                    |  Simulation UI · Booking · APIs  |
                    +────────────┬────────────────────+
                                 │
          +──────────────────────┼──────────────────────+
          │                      │                      │
  +───────▼───────+    +────────▼────────+    +────────▼────────+
  │   SIMULATION  │    │  PRICING ENGINE │    │    NETWORK      │
  │    ENGINE     │    │                 │    │   OPTIMIZER     │
  │               │    │  base × supply  │    │                 │
  │  Bot agents   │    │  × demand       │    │  EMSR-b         │
  │  Cancel/NoShow│    │  × sentiment    │    │  Bid price      │
  │  Overbooking  │    │  × customer     │    │  O&D proration  │
  +───────▲───────+    +────────▲────────+    +────────▲────────+
          │                      │                      │
          +──────────────────────┼──────────────────────+
                                 │
                    +────────────▼────────────────────+
                    |       FORECAST BRIDGE            |
                    |  Connects 3 ML models to sim    |
                    +────────────┬────────────────────+
                                 │
          +──────────────────────┼──────────────────────+
          │                      │                      │
  +───────▼───────+    +────────▼────────+    +────────▼────────+
  │   TFT         │    │  Two-Stage XGB  │    │  Pickup XGB     │
  │               │    │  (Hurdle Model) │    │                 │
  │  Route-daily  │    │  Daily booking  │    │  Remaining pax  │
  │  demand       │    │  prediction     │    │  prediction     │
  │  MAE: 14.03   │    │  AUC: 0.835    │    │  MAE: 3.45      │
  +───────────────+    +─────────────────+    │  SHAP explain.  │
                                              +─────────────────+
                    +─────────────────────────────────+
                    |     SENTIMENT INTELLIGENCE       |
                    |  GDELT + Google News + DeBERTa   |
                    |  51 cities · 14-day recency      |
                    +─────────────────────────────────+
```

## Key Features

### Demand Forecasting
- **Temporal Fusion Transformer (TFT):** Route-daily demand prediction with interpretable attention. 200 entities, 30-day horizon, quantile loss. Variable Selection Network reveals which features drive predictions.
- **Two-Stage XGBoost (Hurdle Model):** Handles 70.8% zero-inflated daily sales data. Stage 1 classifies sale probability (AUC 0.835), Stage 2 predicts quantity (MAE 0.78).
- **Pickup XGBoost:** Predicts remaining passengers at each DTD point. 49 features, 70.4% improvement over naive baseline. SHAP TreeExplainer provides per-flight feature importance.

### Dynamic Pricing
- **4-factor pricing formula:** `price = base * supply * demand * sentiment * customer`
- **Data-calibrated coefficients:** Base price learned via linear regression (R2=0.979), LF curve and route factors from statistical analysis. Season/DOW/special period factors from industry standards.
- **Fare class management:** 4 classes (V/K/M/Y) with DTD rules, LF thresholds, quota limits, and EMSR-b dynamic protection.

### Revenue Optimization
- **EMSR-b fare class optimization:** Forecast-informed Expected Marginal Seat Revenue controls V/K class availability. Protection levels computed using inverse normal CDF with demand estimates from TFT/Pickup.
- **O&D network optimization:** Bid price control for connecting vs local passengers. Distance-based fare proration with 15% connecting discount. Displacement tracking.
- **Overbooking:** Sell limit at 108% capacity, calibrated against segment-specific no-show rates (3-20%).

### Simulation
- **Real-time booking simulation** with configurable speed (1x to 14400x).
- **6 passenger segments** (A-F) with distinct WTP ranges, booking windows, and no-show rates.
- **Cancellation model:** Fare-class based daily cancel probabilities with refund logic.
- **Explainability panel:** TFT attention weights, pricing decomposition, and EMSR-b status displayed in the simulation UI.

### Sentiment Intelligence
- **51 destination cities** monitored via GDELT API and Google News RSS.
- **DeBERTa NLI classification** for positive/negative/neutral sentiment.
- **14-day recency filter** ensures only fresh news affects pricing.
- Dual impact: sentiment affects both demand volume and price level.

## Model Performance

| Model | MAE | Other Metric | Data |
|-------|-----|-------------|------|
| TFT | 14.03 | Correlation: 0.991 | 200 entities x 730 days |
| Pickup XGBoost | 3.45 | WAPE: 9.82%, Improvement: 70.4% | 49 features |
| Two-Stage XGBoost | 0.78 | AUC: 0.835 | 31 features, hurdle model |

## Project Structure

```
seatwise/
├── dashboard/
│   ├── app.py                    # Flask application - all API endpoints
│   ├── pricing_engine.py         # Dynamic pricing (4 multipliers + fare class)
│   ├── simulation_engine.py      # Booking simulation + overbooking + cancellation
│   ├── forecast_bridge.py        # ML model bridge (TFT + Two-Stage + Pickup)
│   ├── network_optimizer.py      # O&D optimization, EMSR-b, bid price
│   ├── sentiment/                # Sentiment analysis module
│   └── templates/                # HTML templates (simulation, booking, dashboard)
├── scripts/
│   ├── calibrate_from_data.py    # Learn pricing coefficients from data
│   ├── extract_tft_attention.py  # Extract TFT interpretation weights
│   ├── generate_reports.py       # PDF report generator
│   ├── data_prep/                # Data preparation pipeline
│   ├── training/                 # Model training scripts
│   └── kaggle/                   # Kaggle GPU training notebooks
├── reports/                      # JSON metric reports + calibration
├── docs_ts/                      # Time series documentation
└── data/                         # Data files (not in repo, see Setup)
```

## Setup

### Requirements

```
Python 3.11+
flask, duckdb, xgboost, pandas, numpy, joblib, shap
pytorch-forecasting, lightning, torch
transformers, feedparser, scipy
reportlab (for PDF reports)
```

### Data

Data files are not included in the repository due to size. Place the following:

```
data/raw/           -> flight_snapshot_v2.parquet, bookings_enriched.parquet
data/processed/     -> tft_route_daily.parquet, pickup_master.parquet, flight_metadata.parquet, ...
data/models/        -> pickup_xgb.json, xgb_demand_*.pkl, tft_full_checkpoint.ckpt, ...
```

### Run

```bash
cd dashboard
python app.py
# Dashboard:  http://localhost:5005
# Simulation: http://localhost:5005/simulation
# Booking:    http://localhost:5005/booking
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Database | DuckDB + Apache Parquet |
| ML - TFT | PyTorch, pytorch-forecasting |
| ML - XGBoost | xgboost, joblib |
| ML - SHAP | shap (TreeExplainer) |
| NLP | HuggingFace transformers, DeBERTa-v3 |
| Optimization | scipy.stats (EMSR-b) |
| Frontend | Vanilla JS, CSS, Chart.js |
| Data Collection | GDELT API, Google News RSS |

