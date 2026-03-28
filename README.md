# SeatWise — Airline Revenue Management System

AI-powered revenue management platform for airline pricing, demand forecasting, and booking simulation.

## Overview

SeatWise is built for the IST hub network: 51 destinations, 100 routes, 2 cabin classes. The system combines three forecasting models with a dynamic pricing engine and real-time booking simulation.

## Architecture

```
TFT (route-daily forecast) ──┐
                              ├── ForecastBridge ──> Simulation Engine
Two-Stage XGBoost (daily) ───┤                       Pricing Engine
                              │
Pickup XGBoost (remaining) ──┘

Sentiment (51 cities) ──────────> Demand + Price adjustment
```

## Modules

| Module | Purpose | Key Metric |
|--------|---------|------------|
| TFT Forecast | Route-level 30-day demand prediction | MAE 14.03, Corr 0.991 |
| Pickup XGBoost | Remaining passenger prediction | MAE 3.45, 70.4% improvement |
| Two-Stage XGBoost | Daily booking prediction (hurdle model) | MAE 0.78, AUC 0.835 |
| Pricing Engine | 4-factor dynamic pricing + fare class management | — |
| Simulation | Live booking sim with 6 passenger segments | — |
| Sentiment | Real-time news analysis, 51 destinations | 14-day filter |

## Project Structure

```
seatwise/
├── dashboard/
│   ├── app.py                  # Flask application (all APIs)
│   ├── pricing_engine.py       # Dynamic pricing (4 multipliers)
│   ├── simulation_engine.py    # Booking simulation
│   ├── forecast_bridge.py      # Model-to-simulation bridge
│   ├── sentiment_app.py        # Standalone sentiment app
│   ├── sentiment/              # Sentiment module
│   ├── templates/              # HTML templates
│   └── static/                 # Static assets
├── scripts/
│   ├── data_prep/              # Data preparation scripts
│   ├── training/               # Model training scripts
│   └── kaggle/                 # Kaggle notebook (TFT)
├── data/                       # Data files (not in repo, see Setup)
├── reports/                    # JSON metric reports
└── archive/                    # Legacy scripts
```

## Setup

### 1. Install dependencies

```bash
pip install flask duckdb xgboost pandas numpy joblib pytorch-forecasting lightning
```

### 2. Data files

Download the data folder from Google Drive and place:

```
data/raw/           <- flight_snapshot_v2.parquet, bookings_enriched.parquet
data/processed/     <- tft_route_daily.parquet, pickup_master.parquet, etc.
data/models/        <- pickup_xgb.json, xgb_demand_*.pkl, tft_*.pt
```

### 3. Environment

```bash
echo "NEWSAPI_KEY=your_key_here" > .env
```

### 4. Run

```bash
cd dashboard
python app.py
# Open http://localhost:5005
```

## Tech Stack

Python, Flask, DuckDB, XGBoost, PyTorch, pytorch-forecasting, Pandas, SQLite, Chart.js
