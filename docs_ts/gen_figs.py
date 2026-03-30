"""Generate HD figures for Time Series documentation."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.path.join(PROJECT_DIR, "data", "processed")
out = os.path.join(PROJECT_DIR, "docs_ts")
os.makedirs(out, exist_ok=True)

df = pd.read_parquet(os.path.join(base, "tft_route_daily.parquet"))
pred = pd.read_parquet(os.path.join(base, "tft_predictions_indexed.parquet"))
flat = pd.read_parquet(os.path.join(base, "tft_predictions.parquet"))

base_date = pd.Timestamp("2025-01-01")
df["date"] = base_date + pd.to_timedelta(df["time_idx"], unit="D")

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "text.color": "#e6edf3", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d", "grid.color": "#21262d", "font.size": 11,
})
A = "#58a6ff"; G = "#3fb950"; R = "#f85149"; O = "#d29922"; P = "#bc8cff"; C = "#39d2c0"

# ── FIG 1 ──
print("Fig 1...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Havayolu Verisinin Zaman Serisi Dogasi", fontsize=16, fontweight="bold", y=0.98)

ent_df = df[df["entity_id"] == "IST_LHR_economy"].sort_values("date")
ax = axes[0, 0]
ax.plot(ent_df["date"], ent_df["total_pax"], color=A, linewidth=0.8, alpha=0.9)
ax.fill_between(ent_df["date"], 0, ent_df["total_pax"], alpha=0.1, color=A)
ax.set_title("IST-LHR Economy: Gunluk Yolcu (730 Gun)", fontsize=11, fontweight="600")
ax.set_ylabel("Toplam Yolcu")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for eid, clr in zip(["IST_LHR_economy", "IST_JFK_economy", "IST_DXB_economy", "IST_CDG_economy"], [A, G, O, P]):
    edf = df[df["entity_id"] == eid].sort_values("date")
    ax.plot(edf["date"], edf["total_pax"], color=clr, linewidth=0.7, alpha=0.8,
            label=eid.replace("IST_", "").replace("_economy", ""))
ax.set_title("4 Rotanin Zaman Serisi Karsilastirmasi", fontsize=11, fontweight="600")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax = axes[1, 0]
monthly = df.groupby("dep_month")["total_pax"].mean().sort_index()
ax.bar(range(1, 13), monthly.values, color=[R if v > monthly.mean() else A for v in monthly.values], alpha=0.8)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Oc", "Sb", "Mr", "Ns", "My", "Hz", "Tm", "Ag", "Ey", "Ek", "Ks", "Ar"])
ax.set_title("Aylik Ortalama Yolcu (Mevsimsellik)", fontsize=11, fontweight="600")
ax.axhline(y=monthly.mean(), color=O, linestyle="--", alpha=0.6)
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 1]
dow = df.groupby("dep_dow")["total_pax"].mean().sort_index()
ax.bar(range(7), dow.values, color=[G if i < 5 else P for i in range(7)], alpha=0.8)
ax.set_xticks(range(7)); ax.set_xticklabels(["Pzt", "Sal", "Car", "Per", "Cum", "Cmt", "Paz"])
ax.set_title("Hafta Gunu Ortalama Yolcu", fontsize=11, fontweight="600")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out, "fig01_timeseries_nature.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  OK")

# ── FIG 2 ──
print("Fig 2...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Neden ARIMA/SARIMA Yetmez? - Verinin Karmasikligi", fontsize=16, fontweight="bold", y=0.98)

ax = axes[0, 0]
n_show = 20
sample = df.groupby("entity_id")["total_pax"].mean().sort_values(ascending=False).head(n_show)
ax.barh(range(n_show), sample.values, color=A, alpha=0.7)
ax.set_yticks(range(n_show))
ax.set_yticklabels([s.replace("IST_", "").replace("_", " ") for s in sample.index], fontsize=7)
ax.set_title("200 Entity = 200 Ayri ARIMA Model Gerekir", fontsize=11, fontweight="600")
ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")

ax = axes[0, 1]
fg = {"Fiyat (4)": 4, "Kanal (5)": 5, "Yolcu (4)": 4, "Donem (9)": 9, "Bolge (2)": 2, "Takvim (3)": 3, "Booking (3)": 3, "Hedef (1)": 1}
ax.pie(fg.values(), labels=fg.keys(), autopct="%1.0f%%",
       colors=[A, G, O, P, C, R, "#f0883e", "#8b949e"],
       textprops={"fontsize": 8, "color": "#e6edf3"}, pctdistance=0.8,
       wedgeprops={"linewidth": 1, "edgecolor": "#0d1117"})
ax.set_title("42 Ozellik: ARIMA ile Kullanilamaz", fontsize=11, fontweight="600")

ax = axes[1, 0]
regions = df.groupby("region")["total_pax"].mean().sort_values(ascending=False)
ax.bar(range(len(regions)), regions.values, color=[A, G, O, P, R][:len(regions)], alpha=0.8)
ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions.index, fontsize=9)
ax.set_title("Bolgesel Farklar: ARIMA Goremiyor", fontsize=11, fontweight="600")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 1]
events = {}
for tag in [c for c in df.columns if c.startswith("tag_")]:
    name = tag.replace("tag_", "")
    wt = df[df[tag] == True]["total_pax"].mean()
    wo = df[df[tag] == False]["total_pax"].mean()
    events[name] = ((wt - wo) / wo) * 100
ev = pd.Series(events).sort_values(ascending=False)
ax.barh(range(len(ev)), ev.values, color=[G if v > 0 else R for v in ev.values], alpha=0.7)
ax.set_yticks(range(len(ev))); ax.set_yticklabels(ev.index, fontsize=8)
ax.set_title("Ozel Donem Etkisi (%): ARIMA Bunu Bilemez", fontsize=11, fontweight="600")
ax.axvline(x=0, color="#8b949e", linewidth=0.5); ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out, "fig02_why_not_arima.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  OK")

# ── FIG 3: TFT Results ──
print("Fig 3...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("TFT Model Sonuclari - Tahmin vs Gercek", fontsize=16, fontweight="bold", y=0.98)

ax = axes[0, 0]
s = flat.sample(min(5000, len(flat)), random_state=42)
ax.scatter(s["actual"], s["predicted"], alpha=0.15, s=8, color=A)
mx = max(s["actual"].max(), s["predicted"].max())
ax.plot([0, mx], [0, mx], color=R, linestyle="--", linewidth=1, alpha=0.7, label="Mukemmel Tahmin")
ax.set_xlabel("Gercek Yolcu"); ax.set_ylabel("Tahmin Yolcu")
corr = flat["actual"].corr(flat["predicted"])
ax.set_title(f"Scatter: Gercek vs Tahmin (r={corr:.3f})", fontsize=11, fontweight="600")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
eid = "IST_LHR_economy"
ep = pred[pred["entity_id"] == eid].sort_values("dep_date")
ax.plot(ep["dep_date"], ep["actual"], color="#8b949e", linewidth=1, alpha=0.8, label="Gercek")
ax.plot(ep["dep_date"], ep["predicted"], color=A, linewidth=1, alpha=0.9, label="TFT Tahmin")
ax.set_title("IST-LHR Economy: TFT Tahmin vs Gercek", fontsize=11, fontweight="600")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)

ax = axes[1, 0]
errors = flat["predicted"] - flat["actual"]
ax.hist(errors, bins=100, color=A, alpha=0.7, edgecolor="none", range=(-50, 50))
ax.axvline(x=0, color=R, linestyle="--", linewidth=1)
ax.axvline(x=errors.mean(), color=G, linestyle="--", linewidth=1, label=f"Ort. Hata: {errors.mean():.2f}")
ax.set_title("Hata Dagilimi (Tahmin - Gercek)", fontsize=11, fontweight="600")
ax.set_xlabel("Hata (Yolcu)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
pred["ae"] = (pred["predicted"] - pred["actual"]).abs()
ent_mae = pred.groupby("entity_id")["ae"].mean().sort_values()
best5 = ent_mae.head(5); worst5 = ent_mae.tail(5)
combined = pd.concat([best5, worst5])
colors = [G] * 5 + [R] * 5
labels = [s.replace("IST_", "").replace("_", " ")[:20] for s in combined.index]
ax.barh(range(10), combined.values, color=colors, alpha=0.7)
ax.set_yticks(range(10)); ax.set_yticklabels(labels, fontsize=8)
ax.set_title("En Iyi ve En Kotu 5 Rota (MAE)", fontsize=11, fontweight="600")
ax.axhline(y=4.5, color="#8b949e", linestyle="--", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out, "fig03_tft_results.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  OK")

# ── FIG 4: Multi-entity TFT predictions ──
print("Fig 4...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("TFT Tahminleri: 6 Farkli Rota", fontsize=16, fontweight="bold", y=0.98)
entities = ["IST_LHR_economy", "IST_JFK_economy", "IST_DXB_economy",
            "IST_CDG_economy", "IST_NRT_economy", "IST_GRU_economy"]
for i, eid in enumerate(entities):
    ax = axes[i // 3, i % 3]
    ep = pred[pred["entity_id"] == eid].sort_values("dep_date")
    if len(ep) == 0:
        ax.text(0.5, 0.5, "Veri yok", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(eid, fontsize=9)
        continue
    ax.plot(ep["dep_date"], ep["actual"], color="#8b949e", linewidth=0.8, alpha=0.7, label="Gercek")
    ax.plot(ep["dep_date"], ep["predicted"], color=A, linewidth=0.8, alpha=0.9, label="Tahmin")
    mae = (ep["predicted"] - ep["actual"]).abs().mean()
    label = eid.replace("IST_", "").replace("_", " ")
    ax.set_title(f"{label} (MAE: {mae:.1f})", fontsize=10, fontweight="600")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45, labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out, "fig04_multi_entity.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  OK")

# ── FIG 5: XGBoost vs TFT roles ──
print("Fig 5...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("3 Model Karsilastirmasi: Rol ve Performans", fontsize=15, fontweight="bold", y=1.02)

models = ["TFT\nRoute-Daily", "XGBoost\nPickup", "Two-Stage\nXGBoost"]
mae_vals = [14.03, 3.45, 0.78]
ax = axes[0]
bars = ax.bar(range(3), mae_vals, color=[R, O, A], alpha=0.8, width=0.6)
for b, v in zip(bars, mae_vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v}", ha="center", fontsize=10, fontweight="600")
ax.set_xticks(range(3)); ax.set_xticklabels(models, fontsize=9)
ax.set_title("MAE (Yolcu)", fontsize=12, fontweight="600"); ax.grid(True, alpha=0.3, axis="y")

scope = ["30 gun ileri\nrota-kabin\n(makro)", "Kalan yolcu\nucus-DTD\n(mikro)", "Gunluk satis\nucus-gun\n(mikro)"]
ax = axes[1]
for i, (m, s, clr) in enumerate(zip(models, scope, [R, O, A])):
    ax.text(0.5, 0.85 - i * 0.33, m, ha="center", va="center", fontsize=12, fontweight="700", color=clr, transform=ax.transAxes)
    ax.text(0.5, 0.73 - i * 0.33, s, ha="center", va="center", fontsize=9, color="#c9d1d9", transform=ax.transAxes)
ax.set_title("Kapsam ve Detay", fontsize=12, fontweight="600"); ax.axis("off")

roles = ["HIZ LIMITI\nTavan/taban\nregulatoru", "DIREKSIYON\nPricing engine\nsupply multiplier", "GAZ PEDALI\nGunluk bot\nuretimi"]
ax = axes[2]
for i, (m, r, clr) in enumerate(zip(models, roles, [R, O, A])):
    ax.text(0.5, 0.85 - i * 0.33, m, ha="center", va="center", fontsize=12, fontweight="700", color=clr, transform=ax.transAxes)
    ax.text(0.5, 0.73 - i * 0.33, r, ha="center", va="center", fontsize=9, color="#c9d1d9", transform=ax.transAxes)
ax.set_title("Sistem Icindeki Rol", fontsize=12, fontweight="600"); ax.axis("off")

plt.tight_layout()
fig.savefig(os.path.join(out, "fig05_model_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  OK")

# ── FIG 6: Region-cabin heatmap ──
print("Fig 6...")
fig, ax = plt.subplots(figsize=(14, 8))
pivot = df.groupby(["region", "cabin_class"])["total_pax"].mean().unstack()
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=11,
                color="white" if val > pivot.values.mean() else "black", fontweight="600")
fig.colorbar(im, ax=ax, label="Ort. Gunluk Yolcu")
ax.set_title("Bolge x Kabin: Ortalama Gunluk Yolcu Haritasi", fontsize=14, fontweight="bold")
fig.savefig(os.path.join(out, "fig06_region_cabin_heatmap.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  OK")

print("\nALL DONE!")
for f in sorted(os.listdir(out)):
    if f.endswith(".png"):
        sz = os.path.getsize(os.path.join(out, f)) / 1024
        print(f"  {f}: {sz:.0f} KB")
