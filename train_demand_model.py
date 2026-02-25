"""
Sprint 4 Step 2 - Two-Stage XGBoost Demand Model
Stage 1: Classifier  (y > 0: sale or not)
Stage 2: Regressor   (y | y > 0: how many pax)
Final:   y_pred = p_sale * max(y_pos_pred, 0)
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

# ---- XGBoost import/install ----
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "xgboost"])
    from xgboost import XGBClassifier, XGBRegressor

BASE_DIR     = Path(r"C:\Users\ahmet\OneDrive\Desktop\ptir")
DATA_PATH    = BASE_DIR / "demand_training.parquet"
OUT_CLF      = BASE_DIR / "xgb_demand_classifier.pkl"
OUT_REG      = BASE_DIR / "xgb_demand_regressor.pkl"
OUT_PRED     = BASE_DIR / "demand_predictions_test.parquet"
OUT_METRICS  = BASE_DIR / "demand_metrics.json"
OUT_FEATURES = BASE_DIR / "feature_list.json"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Not found: {DATA_PATH}")

    print("[1/8] Reading training data...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"       Shape: {df.shape}", flush=True)

    # ---- Keep only 2025/2026 ----
    df = df[df["dep_year"].isin([2025, 2026])]
    print(f"       After year filter: {len(df):,}", flush=True)

    # ---- Basic cleanup ----
    df["cabin_class"] = df["cabin_class"].astype(str).str.lower().str.strip()
    if "region" in df.columns:
        df["region"] = df["region"].astype(str)

    # Target
    y = df["y_pax_sold_today"].astype(float).values
    y_bin = (y > 0).astype(np.int8)

    # Train/Test masks
    train_mask = (df["dep_year"] == 2025).values
    test_mask  = (df["dep_year"] == 2026).values

    # ---- Baselines (compute BEFORE one-hot to save memory) ----
    print("[2/8] Computing baselines...", flush=True)

    # Baseline A: mean pax by (region, cabin, dtd_bucket) from TRAIN
    base_keys = [c for c in ["region", "cabin_class", "dtd_bucket"] if c in df.columns]

    if base_keys:
        # Vectorized merge approach instead of row-by-row apply
        grp_mean = (df.loc[train_mask]
                    .groupby(base_keys, observed=True)["y_pax_sold_today"]
                    .mean()
                    .reset_index()
                    .rename(columns={"y_pax_sold_today": "_baseA"}))

        test_df = df.loc[test_mask, base_keys].reset_index()
        test_df = test_df.merge(grp_mean, on=base_keys, how="left")
        y_pred_baseA = test_df["_baseA"].fillna(0.0).values
        del test_df, grp_mean
    else:
        y_pred_baseA = np.zeros(test_mask.sum())

    # Baseline B: pax_last_7d / 7
    if "pax_last_7d" in df.columns:
        y_pred_baseB = np.clip(df.loc[test_mask, "pax_last_7d"].astype(float).values / 7.0, 0, None)
    else:
        y_pred_baseB = np.zeros(test_mask.sum())

    y_test = y[test_mask]
    baseA_mae  = float(mean_absolute_error(y_test, y_pred_baseA))
    baseA_rmse = rmse(y_test, y_pred_baseA)
    baseB_mae  = float(mean_absolute_error(y_test, y_pred_baseB))
    baseB_rmse = rmse(y_test, y_pred_baseB)

    print(f"       Baseline A (group mean) MAE={baseA_mae:.4f}  RMSE={baseA_rmse:.4f}", flush=True)
    print(f"       Baseline B (pace/7)     MAE={baseB_mae:.4f}  RMSE={baseB_rmse:.4f}", flush=True)

    # ---- Features: drop IDs + target ----
    print("[3/8] Preparing features (one-hot)...", flush=True)
    drop_cols = {"flight_id", "y_pax_sold_today", "dep_year"}
    X = df.drop(columns=[c for c in df.columns if c in drop_cols])

    # One-hot for categoricals
    cat_cols = [c for c in ["cabin_class", "region"] if c in X.columns]
    # dtd_bucket is already int8 code, keep as numeric
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_names = X.columns.tolist()
    print(f"       Features: {len(feature_names)} columns", flush=True)

    X_train = X.iloc[np.where(train_mask)[0]]
    X_test  = X.iloc[np.where(test_mask)[0]]
    y_train = y[train_mask]
    yb_train = y_bin[train_mask]

    # Free big df
    del df, X

    # ---- Stage 1: Classifier (y > 0) ----
    print("[4/8] Training classifier (y>0)...", flush=True)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )
    clf.fit(X_train, yb_train)
    print("       Classifier trained.", flush=True)

    p_sale_test = clf.predict_proba(X_test)[:, 1]
    yb_test = y_bin[test_mask]
    try:
        auc = float(roc_auc_score(yb_test, p_sale_test))
    except Exception:
        auc = None
    print(f"       AUC = {auc}", flush=True)

    # ---- Stage 2: Regressor on positives only ----
    print("[5/8] Training regressor (y|y>0)...", flush=True)
    pos_mask_train = (yb_train == 1)
    X_train_pos = X_train.iloc[np.where(pos_mask_train)[0]]
    y_train_pos = y_train[pos_mask_train]
    print(f"       Positive train samples: {len(X_train_pos):,}", flush=True)

    reg = XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    if len(X_train_pos) > 0:
        reg.fit(X_train_pos, y_train_pos)
        y_pos_pred_test = np.clip(reg.predict(X_test), 0, None)
    else:
        y_pos_pred_test = np.zeros(test_mask.sum())

    del X_train, X_train_pos, y_train, yb_train, y_train_pos
    print("       Regressor trained.", flush=True)

    # ---- Final hurdle prediction ----
    y_pred_test = p_sale_test * y_pos_pred_test
    y_pred_test = np.clip(y_pred_test, 0, None)

    model_mae  = float(mean_absolute_error(y_test, y_pred_test))
    model_rmse = rmse(y_test, y_pred_test)

    print(f"\n[6/8] === RESULTS ===", flush=True)
    print(f"       Baseline A  MAE={baseA_mae:.4f}  RMSE={baseA_rmse:.4f}")
    print(f"       Baseline B  MAE={baseB_mae:.4f}  RMSE={baseB_rmse:.4f}")
    print(f"       XGB Model   MAE={model_mae:.4f}  RMSE={model_rmse:.4f}  AUC={auc}")

    # Flight-level sum error
    print("\n[7/8] Saving artifacts...", flush=True)

    # Save models
    joblib.dump(clf, OUT_CLF)
    joblib.dump(reg, OUT_REG)

    # Save feature list
    with open(OUT_FEATURES, "w", encoding="utf-8") as f:
        json.dump({"features": feature_names}, f, indent=2)

    # Save predictions (for validation)
    # --- Re-read only the keys we need (avoids keeping full df in memory)
    df_keys = pd.read_parquet(DATA_PATH, columns=["flight_id", "cabin_class", "dtd", "dep_year"])
    df_keys = df_keys[df_keys["dep_year"].isin([2025, 2026])]
    test_keys = df_keys.iloc[np.where(test_mask)[0]].copy()
    del df_keys

    test_keys["y_true"]      = y_test
    test_keys["p_sale_pred"] = p_sale_test
    test_keys["y_pos_pred"]  = y_pos_pred_test
    test_keys["y_pred"]      = y_pred_test
    test_keys.to_parquet(OUT_PRED, index=False)

    # ---- Metrics report ----
    metrics = {
        "rows_train_2025": int(train_mask.sum()),
        "rows_test_2026":  int(test_mask.sum()),
        "zero_rate_overall": float((y == 0).mean()),
        "baseline_A": {"mae": baseA_mae, "rmse": baseA_rmse},
        "baseline_B": {"mae": baseB_mae, "rmse": baseB_rmse},
        "two_stage_model": {
            "mae": model_mae,
            "rmse": model_rmse,
            "auc_sale_classifier": auc,
        },
        "artifacts": {
            "classifier": str(OUT_CLF),
            "regressor":  str(OUT_REG),
            "predictions_test": str(OUT_PRED),
            "features": str(OUT_FEATURES),
        }
    }

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[8/8] All saved:")
    print(f"   {OUT_CLF}")
    print(f"   {OUT_REG}")
    print(f"   {OUT_PRED}")
    print(f"   {OUT_METRICS}")
    print(f"   {OUT_FEATURES}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
