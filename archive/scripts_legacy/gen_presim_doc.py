"""Generate pre-simulation documentation."""
import duckdb, json, os, numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESKTOP = os.path.expanduser("~/OneDrive/Desktop")

con = duckdb.connect()
p = lambda f: os.path.join(PROJECT, f).replace("\\", "/")

# Data stats
snap_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p('data/raw/flight_snapshot_v2.parquet')}')").fetchone()[0]
meta = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT flight_id) FROM read_parquet('{p('data/processed/flight_metadata.parquet')}')").fetchone()
tft_rd = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT entity_id), MIN(dep_date), MAX(dep_date) FROM read_parquet('{p('data/processed/tft_route_daily.parquet')}')").fetchone()
pickup = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT flight_id) FROM read_parquet('{p('data/processed/pickup_master.parquet')}')").fetchone()
tft_pred = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT entity_id) FROM read_parquet('{p('data/processed/tft_predictions_indexed.parquet')}')").fetchone()
routes = con.execute(f"SELECT COUNT(DISTINCT departure_airport || '-' || arrival_airport) FROM read_parquet('{p('data/processed/flight_metadata.parquet')}')").fetchone()[0]
regions = [r[0] for r in con.execute(f"SELECT DISTINCT region FROM read_parquet('{p('data/processed/flight_metadata.parquet')}') ORDER BY region").fetchall()]

# TFT metrics
tft_df = con.execute(f"SELECT actual, predicted FROM read_parquet('{p('data/processed/tft_predictions_indexed.parquet')}')").fetchdf()
a, pr = tft_df['actual'].values, tft_df['predicted'].values
tft_m = {"mae": np.mean(np.abs(a-pr)), "rmse": np.sqrt(np.mean((a-pr)**2)),
         "corr": np.corrcoef(a,pr)[0,1], "wape": np.sum(np.abs(a-pr))/np.sum(np.abs(a))*100}
mask = a > 0
tft_m["mape"] = np.mean(np.abs((a[mask]-pr[mask])/a[mask]))*100

region_df = con.execute(f"""SELECT r.region, AVG(ABS(p.actual-p.predicted)) as mae, CORR(p.actual,p.predicted) as corr, COUNT(*) as n
    FROM read_parquet('{p('data/processed/tft_predictions_indexed.parquet')}') p
    JOIN read_parquet('{p('data/processed/tft_route_daily.parquet')}') r ON p.entity_id=r.entity_id AND p.dep_date=r.dep_date
    GROUP BY r.region ORDER BY mae""").fetchdf()
cabin_df = con.execute(f"""SELECT r.cabin_class, AVG(ABS(p.actual-p.predicted)) as mae, CORR(p.actual,p.predicted) as corr
    FROM read_parquet('{p('data/processed/tft_predictions_indexed.parquet')}') p
    JOIN read_parquet('{p('data/processed/tft_route_daily.parquet')}') r ON p.entity_id=r.entity_id AND p.dep_date=r.dep_date
    GROUP BY r.cabin_class ORDER BY mae""").fetchdf()
con.close()

# JSON reports
def load_json(f):
    with open(os.path.join(PROJECT, f), encoding='utf-8') as fh:
        return json.load(fh)

demand = load_json('reports/demand_functions_report.json')
cal = load_json('reports/calibration_report.json')
pk = load_json('reports/pickup_xgb_metrics.json')
dm = load_json('reports/demand_metrics.json')
ti = load_json('reports/tft_interpretation.json')

# Build doc
L = []
w = L.append

w("=" * 90)
w("SEATWISE — PRE-SIMULATION TECHNICAL DOCUMENTATION")
w("Data, Models, Calibration, Sentiment — Complete Reference")
w("=" * 90)
w("")

# 1. DATA
w("=" * 90)
w("1. DATA LAYER")
w("=" * 90)
w("")
w("1.1 Overview")
w(f"  Panel Data: {tft_rd[1]} entities x ~730 days")
w(f"  Routes: {routes} ({routes//2} bidirectional pairs, IST hub)")
w(f"  Regions: {', '.join(regions)}")
w(f"  Cabins: Economy (~300 seats) + Business (~49 seats)")
w(f"  Date Range: {str(tft_rd[2])[:10]} to {str(tft_rd[3])[:10]}")
w(f"  Total Snapshot Records: {snap_count:,}")
w(f"  Total Metadata Records: {meta[0]:,} ({meta[1]:,} unique flights)")
w(f"  TFT Route-Daily: {tft_rd[0]:,} rows")
w(f"  Pickup Master: {pickup[0]:,} rows")
w("")

w("1.2 Passenger Segments")
for seg_id, seg in demand.get('segments', {}).items():
    bw = seg.get('booking_window', {})
    wtp = seg.get('wtp_multiplier', {})
    w(f"  Segment {seg_id}: {seg.get('name', seg_id)}")
    w(f"    Share: {seg.get('base_share_pct',0):.0f}% | DTD: {bw.get('min_dtd',0)}-{bw.get('max_dtd',180)} peak={bw.get('peak_dtd',30)} | WTP: {wtp.get('min',0):.2f}-{wtp.get('max',0):.2f}")
w("")

# 2. FORECASTING
w("=" * 90)
w("2. FORECASTING MODELS")
w("=" * 90)
w("")
w("2.1 Temporal Fusion Transformer (TFT)")
w("  Role: Route-daily total passenger demand prediction")
w("  Architecture: Encoder-Decoder + Variable Selection Network + Multi-Head Attention")
w("  Training: 200 entities, QuantileLoss(0.1, 0.5, 0.9), GroupNormalizer")
w("  Encoder: 60 days | Horizon: 30 days | Split: 60/20/20")
w(f"  MAE: {tft_m['mae']:.2f} | RMSE: {tft_m['rmse']:.2f} | MAPE: {tft_m['mape']:.2f}% | WAPE: {tft_m['wape']:.2f}% | Corr: {tft_m['corr']:.4f}")
w(f"  Predictions: {tft_pred[0]:,} ({tft_pred[1]} entities)")
w("")
w("  By Region:")
for _, r in region_df.iterrows():
    w(f"    {r['region']:15s} MAE={r['mae']:6.2f}  Corr={r['corr']:.4f}  n={int(r['n']):,}")
w("  By Cabin:")
for _, r in cabin_df.iterrows():
    w(f"    {r['cabin_class']:15s} MAE={r['mae']:6.2f}  Corr={r['corr']:.4f}")
w("")
w("  Variable Selection Network:")
w("    Static:  " + " | ".join(f"{x['feature']}:{x['importance']*100:.1f}%" for x in ti.get('static_variable_importance',[])[:5]))
w("    Encoder: " + " | ".join(f"{x['feature']}:{x['importance']*100:.1f}%" for x in ti.get('encoder_variable_importance',[])[:5]))
w("    Decoder: " + " | ".join(f"{x['feature']}:{x['importance']*100:.1f}%" for x in ti.get('decoder_variable_importance',[])[:5]))
w("")
w("  Unconstraining: if tft_total/capacity > 0.90 -> tft_total *= 1.15")
w("  S-Curve: cum_fraction = 1 - (dtd/180)^1.5")
w("")

w("2.2 Two-Stage XGBoost (Hurdle Model)")
w("  Role: Daily booking prediction | Handles 70.8% zero-inflated data")
w("  Formula: daily_demand = P(sale>0) x E[pax|sale>0]")
w(f"  MAE: {dm['two_stage_model']['mae']:.4f} | AUC: {dm['two_stage_model']['auc_sale_classifier']:.4f} | Features: 31")
w(f"  Train: {dm['rows_train_2025']:,} | Test: {dm['rows_test_2026']:,}")
w("  Simulation role: p_sale modulates TFT daily target")
w("")

w("2.3 XGBoost Pickup")
w("  Role: Remaining passenger prediction -> pricing engine supply multiplier")
w(f"  MAE: {pk['mae']:.4f} | WAPE: {pk['wape']}% | Improvement: {pk['improvement_mae_pct']}% | Features: {pk['n_features']}")
w(f"  Train: {pk['train_rows']:,} | Test: {pk['test_rows']:,}")
w("  SHAP: TreeExplainer, top features = remaining_seats > dtd > route_n_flights")
w("")

# 3. PRICING
w("=" * 90)
w("3. PRICING ENGINE")
w("=" * 90)
w("")
w("  Formula: price(fc) = base x fc_mult x supply x demand x sentiment x customer")
w("")
bp = cal['base_price']
w("  Base Price (data-calibrated):")
for cab in bp:
    w(f"    {cab}: {bp[cab]['formula']} (R2={bp[cab]['r_squared']}, n={bp[cab]['n_samples']:,})")
w("")
w("  Fare Classes: V(0.50, 15%, LF40%) | K(0.75, 25%, LF60%) | M(1.00, 35%, LF85%) | Y(1.50, 100%, always)")
w("")
w("  Supply: Pickup-driven expected_final_lf -> 1.00/1.15/1.40/1.80 | DTD boost: <=3:1.30 <=7:1.15 <=14:1.05")
w("  Demand: season x special x dow, dampened x0.7 | Jul=1.30, Kurban=1.60, Fri=1.15")
w("  Sentiment: 1.0 + S_city x 0.15 (+-15%)")
w("  Customer: segment WTP + behavioral factors, clamped 0.90-1.20")
w("")

# 4. SENTIMENT
w("=" * 90)
w("4. SENTIMENT INTELLIGENCE")
w("=" * 90)
w("")
w("  Pipeline: GDELT API + Google News RSS -> keyword classifier (416 kw, 8 cat) -> scoring -> city composite")
w("  Scoring: score = event_impact (GDELT ArtList mode does not return tone)")
w("  Composite: S = sum(score_i x e^(-0.1 x h_i)) / sum(e^(-0.1 x h_i)) | 14-day filter")
w("  Alert: HIGH (S<-0.3 or security_threat) | MEDIUM (S<-0.1) | LOW")
w("  Price effect: +-15% | Demand effect: +-30%")
w("")
w("  Categories: security_threat(-0.8, 78kw) | weather_disaster(-0.7, 43kw) | health_crisis(-0.7, 35kw)")
w("              strike_protest(-0.6, 36kw) | political_instability(-0.6, 31kw) | flight_disruption(-0.5, 56kw)")
w("              tourism_growth(+0.5, 71kw) | positive_travel(+0.4, 66kw)")
w("")
w("  Decay: 1h->0.905 | 24h->0.091 | 7d->0.001 | 14d->filtered")
w("  False positives: box office bomb, strike gold, fire sale, crash course, killer app, bomb cyclone")
w("")

# 5. CALIBRATION
w("=" * 90)
w("5. DATA CALIBRATION")
w("=" * 90)
w("")
w("  Market-driven (from data): base_price, route_factors, LF_curve, DTD_curve")
w("  Policy-driven (industry): season, DOW, special periods")
w("")
dtd = cal.get('dtd_factors',{}).get('factors',{})
w("  DTD Curve: " + " | ".join(f"{k}:{dtd[k]}" for k in ['0-3','4-7','8-14','15-30','31-60','61-90','91-120','121+'] if k in dtd))
lf = cal.get('lf_curve',{}).get('factors',{})
w("  LF Curve:  " + " | ".join(f"{k}:{lf[k]}" for k in ['LF<30','LF30-50','LF50-70','LF70-85','LF85-95','LF95+'] if k in lf))
w("")

# 6. NETWORK
w("=" * 90)
w("6. O&D NETWORK OPTIMIZATION")
w("=" * 90)
w("")
w("  EMSR-b: protection(high) = mean_d + std_d x Phi^(-1)(1 - price_ratio)")
w("  Forecast-informed: demand_base = min(TFT_total, capacity x 1.1)")
w("  Controls V and K only | Re-opens if booking pace < 70% of expected")
w("  Bid price: current_prices[cheapest_open] | Active at LF > 70%")
w("  Proration: leg_contribution = total_fare x dest_dist / total_dist | 15% connecting discount")
w("")

# Save
out = "\n".join(L)
path = os.path.join(DESKTOP, "SeatWise_Pre_Simulation_Documentation.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write(out)
print(f"Saved: {path}")
print(f"Lines: {len(L)}")
