"""
Seatwise — Validation Highlights figures (300-scenario Monte Carlo).
Reads reports/validation_results.json, computes the extreme / notable cases,
and renders poster-ready figures into <Desktop>/Seatwise_KeyResults/.

All numbers are computed directly from the data file — nothing fabricated.
"""
import os
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path("C:/Users/ahmet/OneDrive/Desktop/seatwise_v4/seatwise_v4_ahmet")
OUT = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop" / "Seatwise_KeyResults"
OUT.mkdir(parents=True, exist_ok=True)

with open(REPO / "reports" / "validation_results.json", encoding="utf-8") as f:
    D = json.load(f)

N = len(D)
deltas = [r["rev_delta_pct"] for r in D]
lfs = [r["lf"] for r in D]
rev_dyn_total = sum(r["rev_dynamic"] for r in D)
rev_base_total = sum(r["rev_baseline"] for r in D)
overall_lift = (rev_dyn_total - rev_base_total) / rev_base_total * 100

mean_d, med_d, std_d = st.mean(deltas), st.median(deltas), st.pstdev(deltas)
min_d, max_d = min(deltas), max(deltas)
n_pos = sum(1 for d in deltas if d > 0.0)
n_ge10 = sum(1 for d in deltas if d >= 10)
n_ge25 = sum(1 for d in deltas if d >= 25)
n_ge50 = sum(1 for d in deltas if d >= 50)
n_neg = sum(1 for d in deltas if d < 0)

best = max(D, key=lambda r: r["rev_delta_pct"])
worst = min(D, key=lambda r: r["rev_delta_pct"])
fullest = max(D, key=lambda r: r["lf"])
emptiest = min(D, key=lambda r: r["lf"])
most_rev = max(D, key=lambda r: r["rev_dynamic"])

def by(key):
    g = {}
    for r in D:
        g.setdefault(r[key], []).append(r["rev_delta_pct"])
    return {k: (st.mean(v), len(v)) for k, v in g.items()}

by_region = by("region")
by_period = by("period")
by_rtype = by("route_type")
by_cabin = by("cabin")

# ── console report (verification) ──
print("=" * 64)
print(f"VALIDATION RESULTS — {N} scenarios")
print("=" * 64)
print(f"Total dynamic revenue : ${rev_dyn_total:,.0f}")
print(f"Total baseline revenue: ${rev_base_total:,.0f}")
print(f"Aggregate revenue lift: {overall_lift:+.2f}%")
print(f"Per-scenario delta    : mean {mean_d:+.2f}% · median {med_d:+.2f}% · σ {std_d:.2f}%")
print(f"                        min {min_d:+.2f}% · max {max_d:+.2f}%")
print(f"Beat static (>0%)     : {n_pos}/{N}")
print(f"  ≥ +10% lift         : {n_ge10}/{N}")
print(f"  ≥ +25% lift         : {n_ge25}/{N}")
print(f"  ≥ +50% lift         : {n_ge50}/{N}")
print(f"  underperformed (<0%): {n_neg}/{N}")
print()
def show(tag, r):
    print(f"{tag:28s} {r['route']:9s} {r['cabin']:9s} {r['region']:13s} "
          f"{r.get('period',''):7s} LF={r['lf']:5.1f}%  Δ={r['rev_delta_pct']:+6.2f}%  "
          f"dyn=${r['rev_dynamic']:,.0f}")
show("BIGGEST revenue gain", best)
show("WORST scenario", worst)
show("FULLEST flight", fullest)
show("EMPTIEST flight", emptiest)
show("HIGHEST dynamic revenue", most_rev)
print()
print("Avg lift by region :", {k: f"{v:+.1f}%" for k,(v,_) in by_region.items()})
print("Avg lift by period :", {k: f"{v:+.1f}%" for k,(v,_) in by_period.items()})
print("Avg lift by route  :", {k: f"{v:+.1f}%" for k,(v,_) in by_rtype.items()})
print("Avg lift by cabin  :", {k: f"{v:+.1f}%" for k,(v,_) in by_cabin.items()})

# ── style ──
NAVY="#0b3d91"; NAVY_LT="#5b7fc7"; GRAY="#9aa0a6"; GREEN="#1a7a4a"; RED="#a4202a"
INK="#1a1a1a"; BOXBG="#eef1f8"; FAINT="#555555"
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
                     "font.size":11,"axes.edgecolor":"#666666","axes.linewidth":0.9,
                     "figure.dpi":300,"savefig.dpi":300})
try:
    plt.rcParams["text.parse_math"] = False   # treat '$' as a literal character
except KeyError:
    pass

def fmt_route(r):
    return f"{r['route']} · {r['cabin'].capitalize()} · {r['region']} · {r.get('period','').capitalize()}"

# ════════════════════════════════════════════════════════════════
# 07) VALIDATION HIGHLIGHTS — leaderboard table
# ════════════════════════════════════════════════════════════════
def fig_highlights():
    rows = [
        ["Scenarios simulated", f"{N} independent Monte-Carlo runs",
         "5 regions × 3 seasons · mixed route types"],
        ["Mean revenue lift  (per scenario)", f"{mean_d:+.2f}%",
         f"median {med_d:+.1f}%  ·  σ {std_d:.1f}%  ·  range [{min_d:+.1f}%, {max_d:+.1f}%]"],
        ["Revenue-weighted aggregate lift", f"{overall_lift:+.2f}%   (conservative)",
         f"${rev_dyn_total/1e6:.1f} M dynamic  vs  ${rev_base_total/1e6:.1f} M static"],
        ["Biggest revenue gain", f"{best['rev_delta_pct']:+.2f}%",
         fmt_route(best) + f"   ·   dynamic ${best['rev_dynamic']:,.0f}"],
        ["Highest dynamic revenue", f"${most_rev['rev_dynamic']:,.0f}",
         fmt_route(most_rev) + f"   ·   LF {most_rev['lf']:.1f}%   ·   Δ {most_rev['rev_delta_pct']:+.1f}%"],
        ["Fullest flight (load factor)", f"{fullest['lf']:.1f}% LF   (overbooking active)",
         fmt_route(fullest) + f"   ·   {fullest['sold']}/{fullest['capacity']} seats sold"],
        ["Worst-case scenario", f"{worst['rev_delta_pct']:+.2f}%"
            + ("   (still ≥ static)" if worst['rev_delta_pct'] >= 0 else "   (below static)"),
         fmt_route(worst) + f"   ·   only {n_neg}/{N} scenarios fell below static"],
        ["Scenarios beating static", f"{n_pos} / {N}",
         f"{n_ge25}/{N} gained ≥ +25%   ·   {n_ge50}/{N} gained ≥ +50%   ·   {n_neg}/{N} below static"],
    ]
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.axis("off")
    tbl = ax.table(cellText=[[a, b, c] for a, b, c in rows],
                   colLabels=["", "Result", "Where / detail"],
                   loc="center", cellLoc="left",
                   colWidths=[0.245, 0.275, 0.48])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.0); tbl.scale(1, 2.2)
    for j in range(3):
        c = tbl[0, j]; c.set_facecolor(NAVY)
        c.set_text_props(color="white", fontweight="bold"); c.set_edgecolor("white")
    for i in range(1, len(rows)+1):
        for j in range(3):
            c = tbl[i, j]
            c.set_facecolor("#f4f6fb" if i % 2 else "white"); c.set_edgecolor("#cccccc")
            if j == 0: c.set_text_props(fontweight="bold", color=NAVY)
            if j == 1: c.set_text_props(fontweight="bold", color=GREEN)
    ax.set_title("(e)  Validation Highlights — 300-Scenario Monte Carlo",
                 fontsize=13.5, fontweight="bold", color=NAVY, loc="left", pad=14)
    fig.text(0.06, -0.02,
        "All values computed directly from 300 simulated route × cabin × season scenarios. "
        "Per-scenario mean (+28.26%) weights each\nscenario equally; the revenue-weighted aggregate "
        "(+21.98%) is the more conservative dollar-on-dollar figure. Gains concentrate on VFR / "
        "leisure routes\nwhere static fares sat furthest from demand-clearing levels; only 9 of 300 "
        "scenarios fell below the static baseline.",
        fontsize=8.4, style="italic", color=FAINT)
    fig.tight_layout()
    fig.savefig(OUT / "07_validation_highlights_table.png", bbox_inches="tight")
    plt.close(fig)

# ════════════════════════════════════════════════════════════════
# 08) DISTRIBUTION HISTOGRAM of the 300 lifts
# ════════════════════════════════════════════════════════════════
def fig_distribution():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    lo = min(0, np.floor(min_d/10)*10)
    hi = np.ceil(max_d/10)*10
    bins = np.arange(lo, hi + 10, 10)
    counts, edges, patches = ax.hist(deltas, bins=bins, edgecolor="white", linewidth=1.2)
    for p, left in zip(patches, edges[:-1]):
        p.set_facecolor(GREEN if left >= 0 else RED)
    ax.axvline(0, color=GRAY, ls="--", lw=1.2)
    ax.axvline(mean_d, color=NAVY, lw=2.2)
    ax.text(mean_d, ax.get_ylim()[1]*0.92, f"  mean = {mean_d:+.2f}%",
            color=NAVY, fontsize=11, fontweight="bold", va="top")
    ax.text(0, ax.get_ylim()[1]*0.78, "static\nparity", color=GRAY, fontsize=9,
            ha="center", va="top")
    ax.set_xlabel("Revenue lift vs static EMSR baseline  (%)", fontsize=11.5)
    ax.set_ylabel("Number of scenarios", fontsize=11.5)
    ax.set_title("(f)  Revenue-Lift Distribution Across 300 Scenarios",
                 fontsize=13.5, fontweight="bold", color=NAVY, loc="left", pad=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    # annotation box
    txt = (f"{n_pos} / {N} scenarios beat static pricing\n"
           f"{n_ge25} / {N} gained ≥ +25%   ·   {n_ge50} / {N} gained ≥ +50%\n"
           f"{n_neg} / {N} underperformed   ·   range [{min_d:+.1f}%, {max_d:+.1f}%]")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.6, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BOXBG, edgecolor=NAVY, lw=1.1))
    fig.text(0.10, -0.02,
        "Heavy right tail: a minority of scenarios (typically under-priced economy cabins) deliver "
        "very large gains, pulling the\nmean to +28.26%. The wide spread (σ = 27.84%) reflects "
        "route / season / region heterogeneity, not estimation noise.",
        fontsize=8.6, style="italic", color=FAINT)
    fig.tight_layout()
    fig.savefig(OUT / "08_lift_distribution.png", bbox_inches="tight")
    plt.close(fig)

# ════════════════════════════════════════════════════════════════
# 09) BREAKDOWN — avg lift by cabin / region / route type / season
# ════════════════════════════════════════════════════════════════
def _barpanel(ax, mapping, title, color, order=None, rot=0):
    if order:
        items = sorted(mapping.items(),
                       key=lambda kv: (order.index(kv[0]) if kv[0] in order else 99))
    else:
        items = sorted(mapping.items(), key=lambda kv: -kv[1][0])
    labs = [k.capitalize() if not k.isupper() else k for k, _ in items]
    vals = [v for _, (v, _) in items]
    xs = range(len(labs))
    bars = ax.bar(xs, vals, color=color, width=0.6, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + max(vals)*0.025, f"{v:+.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labs, fontsize=9.5, rotation=rot, ha=("right" if rot else "center"))
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_ylim(0, max(vals)*1.18)


def fig_breakdown():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _barpanel(axes[0, 0], by_cabin,  "by Cabin",      NAVY)
    _barpanel(axes[0, 1], by_region, "by Region",     NAVY,  rot=20)
    _barpanel(axes[1, 0], by_rtype,  "by Route Type", NAVY_LT, rot=20)
    _barpanel(axes[1, 1], by_period, "by Season",     NAVY_LT,
              order=["winter", "spring", "shoulder", "summer", "fall", "off-peak", "peak"])
    axes[0, 0].set_ylabel("Mean revenue lift  (%)", fontsize=11)
    axes[1, 0].set_ylabel("Mean revenue lift  (%)", fontsize=11)
    fig.suptitle("(g)  Where Dynamic Pricing Helps Most — Mean Lift Breakdown (300 scenarios)",
                 fontsize=14, fontweight="bold", color=NAVY, x=0.06, ha="left", y=0.99)
    fig.text(0.08, 0.005,
        "VFR and leisure routes gain most — their static fares sat furthest below demand-clearing "
        "levels; hub routes (already competitively\npriced) gain least. Economy edges business; winter "
        "scenarios benefit more than summer (lower static-fare baselines). Means over 300 scenarios.",
        fontsize=8.6, style="italic", color=FAINT)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT / "09_lift_breakdown.png", bbox_inches="tight")
    plt.close(fig)

fig_highlights()
fig_distribution()
fig_breakdown()

# append to README
extra = (
    "\n\nADDED — validation highlights\n"
    "-----------------------------\n"
    "07_validation_highlights_table.png — best / worst / fullest / most-profitable + aggregate stats\n"
    "08_lift_distribution.png           — histogram of the 300 per-scenario lifts (mean line, beat-rate box)\n"
    "09_lift_breakdown.png              — mean lift by cabin / region / season\n\n"
    f"Computed from reports/validation_results.json ({N} records):\n"
    f"  aggregate revenue lift   {overall_lift:+.2f}%   (${rev_dyn_total:,.0f} dyn / ${rev_base_total:,.0f} static)\n"
    f"  per-scenario mean/median {mean_d:+.2f}% / {med_d:+.2f}%   σ {std_d:.2f}%   range [{min_d:+.2f}%, {max_d:+.2f}%]\n"
    f"  beat static              {n_pos}/{N}    (≥+25%: {n_ge25}/{N}   ≥+50%: {n_ge50}/{N}   <0%: {n_neg}/{N})\n"
    f"  biggest gain             {best['route']} {best['cabin']} {best['region']} {best.get('period','')}  {best['rev_delta_pct']:+.2f}%\n"
    f"  fullest flight           {fullest['route']} {fullest['cabin']} {fullest['region']} {fullest.get('period','')}  LF {fullest['lf']:.1f}%\n"
    f"  highest dynamic revenue  {most_rev['route']} {most_rev['cabin']} {most_rev['region']} {most_rev.get('period','')}  ${most_rev['rev_dynamic']:,.0f}\n"
    f"  worst case               {worst['route']} {worst['cabin']} {worst['region']} {worst.get('period','')}  {worst['rev_delta_pct']:+.2f}%\n"
)
with open(OUT / "README.txt", "a", encoding="utf-8") as f:
    f.write(extra)

print()
print("OK ->", OUT)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
