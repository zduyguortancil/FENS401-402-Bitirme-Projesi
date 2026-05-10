"""
Seatwise — single "KEY RESULTS" panel for the poster:
  TOP    : per-route revenue bar chart (static vs dynamic, indexed; LF + lift% labels)
  BOTTOM : unified table of those routes  +  a project-wide summary band
Output: <Desktop>/Seatwise_KeyResults/  (light + dark)

All route-level numbers come straight from reports/validation_results.json (300 records);
the summary numbers are computed from the same file (+ report sec. 3.3 for the 21->42% pilot LF
and sec. 3.6 for the DTD sensitivity).
"""
import os, json, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    plt.rcParams["text.parse_math"] = False
except KeyError:
    pass

REPO = Path("C:/Users/ahmet/OneDrive/Desktop/seatwise_v4/seatwise_v4_ahmet")
OUT = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop" / "Seatwise_KeyResults"
OUT.mkdir(parents=True, exist_ok=True)

with open(REPO / "reports" / "validation_results.json", encoding="utf-8") as f:
    D = json.load(f)

# ── project-wide summary ──
N = len(D)
deltas = [r["rev_delta_pct"] for r in D]
mean_d = st.mean(deltas); sd = 27.84                       # report 3.5.1
ci95 = 1.96 * sd / (N ** 0.5)
rev_dyn_tot = sum(r["rev_dynamic"] for r in D)
rev_base_tot = sum(r["rev_baseline"] for r in D)
agg = (rev_dyn_tot - rev_base_tot) / rev_base_tot * 100
n_pos = sum(1 for d in deltas if d > 0)

# ── pick 6 representative scenarios ──
def key(r): return (r["route"], r["cabin"], r.get("period", ""))
picks, seen = [], set()
def add(r, tag):
    if r and key(r) not in seen:
        rr = dict(r); rr["_tag"] = tag
        picks.append(rr); seen.add(key(r))

s_sorted = sorted(D, key=lambda r: r["rev_delta_pct"])
add(max(D, key=lambda r: r["rev_delta_pct"]), "biggest gain")
add(max(D, key=lambda r: r["lf"]), "fullest flight")
add(max(D, key=lambda r: r["rev_dynamic"]), "highest revenue")
add(s_sorted[len(s_sorted)//2], "median scenario")
# one "strong but not extreme" — closest delta to +50 not already picked
strong = min((r for r in D if key(r) not in seen), key=lambda r: abs(r["rev_delta_pct"] - 50))
add(strong, "strong route")
add(min(D, key=lambda r: r["rev_delta_pct"]), "worst case")

picks.sort(key=lambda r: -r["rev_delta_pct"])     # left = biggest lift
print("Selected scenarios:")
for r in picks:
    print(f"  {r['route']:9s} {r['cabin']:9s} {r['region']:12s} {r.get('period',''):8s} "
          f"LF={r['lf']:5.1f}%  Δ={r['rev_delta_pct']:+7.2f}%  "
          f"static=${r['rev_baseline']:,.0f}  dyn=${r['rev_dynamic']:,.0f}  [{r['_tag']}]")
print(f"\nSummary: mean {mean_d:+.2f}% · 95%CI ±{ci95:.2f}% · σ {sd}% · agg {agg:+.2f}% "
      f"(${rev_dyn_tot:,.0f}/${rev_base_tot:,.0f}) · {n_pos}/{N} beat static")

# ── theme ──
THEMES = {
 "light": dict(bg="#ffffff", panel="#f6f8fc", line="#d7dce6", grid="#e6eaf1",
               head="#10131a", body="#1a1a1a", mut="#5b6270",
               navy="#0b3d91", gray="#a9b0bd", pos="#0a6b3d", neg="#a4202a",
               hdrbg="#0b3d91", hdrtx="#ffffff", bandbg="#eef1f8"),
 "dark":  dict(bg="#0d1117", panel="#161b22", line="#30363d", grid="#21262d",
               head="#e6edf3", body="#e6edf3", mut="#8b949e",
               navy="#58a6ff", gray="#5b6675", pos="#34d399", neg="#f0716c",
               hdrbg="#1f2733", hdrtx="#e6edf3", bandbg="#161b22"),
}
F_SANS = "DejaVu Sans"


def build(theme_name):
    t = THEMES[theme_name]
    fig = plt.figure(figsize=(15, 10.4))
    fig.patch.set_facecolor(t["bg"])
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 1.0, 0.30],
                          left=0.055, right=0.965, top=0.93, bottom=0.045, hspace=0.30)

    fig.suptitle("KEY RESULTS — Dynamic vs Static Pricing, by Route",
                 fontsize=18, fontweight="bold", color=t["navy"], x=0.055, ha="left", y=0.975)

    # ─────────────── TOP: bar chart (indexed: static = 100) ───────────────
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(t["bg"])
    x = np.arange(len(picks)); w = 0.30
    static_idx = [100.0] * len(picks)
    dyn_idx = [100.0 + r["rev_delta_pct"] for r in picks]
    ax.bar(x - w/2, static_idx, w, color=t["gray"], edgecolor=t["bg"], linewidth=1.2,
           label="Static EMSR pricing")
    bars = ax.bar(x + w/2, dyn_idx, w, color=t["navy"], edgecolor=t["bg"], linewidth=1.2,
                  label="Dynamic pricing")
    ax.axhline(100, color=t["gray"], ls="--", lw=1.0, zorder=0)
    for xi, r, b in zip(x, picks, bars):
        d = r["rev_delta_pct"]
        ax.text(xi + w/2, max(dyn_idx[xi % len(dyn_idx)], 100) + 4,
                f"{d:+.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=(t["pos"] if d >= 0 else t["neg"]))
    # x labels: route + cabin + LF
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['route']}\n{r['cabin'].capitalize()} · {r.get('period','').capitalize()}\nLF {r['lf']:.0f}%"
         for r in picks], fontsize=9.6, color=t["body"])
    ax.set_ylabel("Revenue  (indexed: static baseline = 100)", fontsize=12, color=t["body"])
    ax.set_ylim(0, max(dyn_idx) * 1.16)
    ax.tick_params(colors=t["mut"])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(t["line"])
    ax.yaxis.grid(True, color=t["grid"], lw=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right", labelcolor=t["body"])
    ax.text(0.0, 1.02, "selected scenarios from the 300-scenario Monte-Carlo validation",
            transform=ax.transAxes, fontsize=9.4, style="italic", color=t["mut"])

    # ─────────────── BOTTOM: unified table ───────────────
    axt = fig.add_subplot(gs[1]); axt.axis("off")
    cols = ["Route", "Cabin", "Region", "Type", "Period",
            "Static ($M)", "Dynamic ($M)", "Lift %", "Load Factor"]
    rows = []
    for r in picks:
        rows.append([
            r["route"], r["cabin"].capitalize(), r["region"],
            r.get("route_type", "—"), r.get("period", "—").capitalize(),
            f"{r['rev_baseline']/1e6:,.2f}", f"{r['rev_dynamic']/1e6:,.2f}",
            f"{r['rev_delta_pct']:+.1f}%", f"{r['lf']:.1f}%",
        ])
    tbl = axt.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                    colWidths=[0.105, 0.085, 0.115, 0.085, 0.095, 0.115, 0.115, 0.09, 0.10])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.0); tbl.scale(1, 2.0)
    for j in range(len(cols)):
        c = tbl[0, j]; c.set_facecolor(t["hdrbg"])
        c.set_text_props(color=t["hdrtx"], fontweight="bold"); c.set_edgecolor(t["bg"])
    for i in range(1, len(rows)+1):
        d = picks[i-1]["rev_delta_pct"]
        for j in range(len(cols)):
            c = tbl[i, j]
            c.set_facecolor(t["panel"] if i % 2 else t["bg"]); c.set_edgecolor(t["line"])
            c.set_text_props(color=t["body"])
            if j == 0: c.set_text_props(fontweight="bold", color=t["navy"])
            if j == 7: c.set_text_props(fontweight="bold", color=(t["pos"] if d >= 0 else t["neg"]))

    # ─────────────── SUMMARY band ───────────────
    axb = fig.add_subplot(gs[2]); axb.axis("off")
    axb.add_patch(mpatches.FancyBboxPatch((0.0, 0.05), 1.0, 0.9,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor=t["bandbg"], edgecolor=t["navy"], linewidth=1.4,
        transform=axb.transAxes, clip_on=False))
    summ = (f"ALL {N} SCENARIOS:   mean revenue lift {mean_d:+.2f}%  (95% CI ±{ci95:.1f}% · σ {sd}%)"
            f"   |   revenue-weighted aggregate {agg:+.2f}%  (${rev_dyn_tot/1e6:,.0f} M vs ${rev_base_tot/1e6:,.0f} M)"
            f"   |   {n_pos}/{N} beat static")
    axb.text(0.5, 0.66, summ, transform=axb.transAxes, ha="center", va="center",
             fontsize=11.0, fontweight="bold", color=t["navy"])
    axb.text(0.5, 0.27, "pilot (6 flights, 181 days): economy load factor 21% → 42%   ·   "
                        "sensitivity: DTD is the dominant price driver (≈ ±32%)   ·   "
                        "demand models: Two-Stage XGBoost AUC 0.835 · Pickup WAPE 9.82% · TFT q50 WAPE ≈ 5.9%",
             transform=axb.transAxes, ha="center", va="center",
             fontsize=8.8, color=t["mut"], style="italic")

    fig.text(0.055, 0.018,
        "Source: 300-scenario Monte-Carlo validation (5 regions × 3 seasons). Selected scenarios span the "
        "biggest gain (+181%), the fullest flight (LF ≈ 101%), the highest-revenue route ($6.5 M) and the "
        "worst case (−5.7%) — only 9 of 300 fell below static.",
        fontsize=8.4, color=t["mut"], style="italic")

    fig.savefig(OUT / f"KEY_RESULTS_panel_{theme_name}.png", dpi=300,
                facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


for th in ("light", "dark"):
    build(th)

(OUT / "README.txt").write_text(
    "SEATWISE — KEY RESULTS PANEL (single-piece, poster-ready)\n"
    "=========================================================\n\n"
    "  KEY_RESULTS_panel_light.png   — light background (academic poster default)\n"
    "  KEY_RESULTS_panel_dark.png    — dark background\n\n"
    "Layout:  top  = per-route revenue bar chart (static vs dynamic, indexed to static = 100;\n"
    "                each route labelled with its load factor and revenue-lift %)\n"
    "         bottom = unified table of those same routes (region, type, period, $static, $dynamic,\n"
    "                lift %, load factor) + a project-wide summary band.\n\n"
    "The chart uses an INDEXED y-axis (static = 100) instead of raw dollars so that routes of very\n"
    "different size stay visually comparable; the real dollar figures are in the table below it.\n\n"
    "Every route number is taken directly from reports/validation_results.json (300 records);\n"
    "the summary band: mean lift +28.26% (95% CI ±3.2%, σ 27.84%), revenue-weighted +21.98%\n"
    "($445.6 M vs $365.3 M), 291/300 beat static — plus the pilot LF 21%->42% (report 3.3) and the\n"
    "DTD sensitivity (report 3.6).\n",
    encoding="utf-8")

print("\nOK ->", OUT)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
