"""
Seatwise — compact "KEY RESULT" callout for the poster.
Output: <Desktop>/Seatwise_KeyResults/  (dark + light · box + strip variants)

All numbers verified from the project's report files:
  +28.26% mean lift, 95% CI +-3.2%   reports/validation_results.json (300 records) + report 3.5.1
  21% -> 42% economy load factor      report 3.3
  291 / 300 scenarios beat static     reports/validation_results.json (computed)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    plt.rcParams["text.parse_math"] = False
except KeyError:
    pass

OUT = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop" / "Seatwise_KeyResults"
OUT.mkdir(parents=True, exist_ok=True)

# fonts that always exist in matplotlib
F_SANS = "DejaVu Sans"
F_MONO = "DejaVu Sans Mono"

# ── content (single source of truth) ──────────────────────────
ROWS = [
    # big value,        line 1 (label),                      line 2 (sub / caveat)
    ("+28.26%",   "revenue vs static EMSR pricing",   "300-scenario Monte Carlo  ·  95% CI ±3.2%"),
    ("21% → 42%", "economy load factor",              "by filling empty seats — NOT by discounting"),
    ("291 / 300", "scenarios beat the static baseline","best +181%  ·  only 9 / 300 below static"),
]
FOOTER = "Seatwise · Dynamic Pricing & DSS for Airline Revenue Management · Group 16"

THEMES = {
    "dark": dict(
        bg="#0d1117", panel="#161b22", border="#30363d",
        head="#e6edf3", label="#c9d1d9", sub="#8b949e",
        accent="#58a6ff", num="#22d3ee", good="#34d399", rule="#30363d",
    ),
    "light": dict(
        bg="#ffffff", panel="#f6f8fc", border="#d7dce6",
        head="#10131a", label="#2b2f38", sub="#5b6270",
        accent="#0b3d91", num="#0b3d91", good="#0a6b3d", rule="#cdd3df",
    ),
}


def _round_box(ax, x, y, w, h, fc, ec, lw=1.4, rad=0.04):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, clip_on=False))


# ════════════════════════════════════════════════════════════════
# VERTICAL BOX  (tall poster slot)
# ════════════════════════════════════════════════════════════════
def make_box(theme_name):
    t = THEMES[theme_name]
    fig = plt.figure(figsize=(7.6, 6.2))
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor(t["bg"])

    # outer panel
    _round_box(ax, 0.035, 0.035, 0.93, 0.93, t["panel"], t["border"], lw=1.6, rad=0.035)

    # header
    ax.text(0.085, 0.905, "K E Y   R E S U L T", transform=ax.transAxes,
            fontfamily=F_SANS, fontsize=14, fontweight="bold",
            color=t["accent"], va="center")
    ax.plot([0.085, 0.915], [0.865, 0.865], color=t["rule"], lw=1.2,
            transform=ax.transAxes)

    # three rows
    ys = [0.74, 0.50, 0.265]
    for (val, l1, l2), y in zip(ROWS, ys):
        col = t["good"] if (val.startswith("+") or "/" in val) else t["num"]
        ax.text(0.085, y, val, transform=ax.transAxes, fontfamily=F_MONO,
                fontsize=33, fontweight="bold", color=col, va="center")
        ax.text(0.085, y - 0.085, l1, transform=ax.transAxes, fontfamily=F_SANS,
                fontsize=13.5, color=t["label"], va="center")
        ax.text(0.085, y - 0.135, l2, transform=ax.transAxes, fontfamily=F_SANS,
                fontsize=10.2, color=t["sub"], style="italic", va="center")
        if y != ys[-1]:
            ax.plot([0.085, 0.915], [y - 0.175, y - 0.175], color=t["rule"],
                    lw=0.8, alpha=0.6, transform=ax.transAxes)

    # footer
    ax.plot([0.085, 0.915], [0.10, 0.10], color=t["rule"], lw=1.0,
            transform=ax.transAxes)
    ax.text(0.085, 0.065, FOOTER, transform=ax.transAxes, fontfamily=F_SANS,
            fontsize=8.6, color=t["sub"], va="center")

    fig.savefig(OUT / f"RESULT_box_{theme_name}.png", dpi=300,
                facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# HORIZONTAL STRIP  (wide / short poster slot)
# ════════════════════════════════════════════════════════════════
def make_strip(theme_name):
    t = THEMES[theme_name]
    fig = plt.figure(figsize=(16, 3.4))
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor(t["bg"])

    # title bar (left)
    ax.text(0.018, 0.80, "KEY", transform=ax.transAxes, fontfamily=F_SANS,
            fontsize=15, fontweight="bold", color=t["accent"], va="center")
    ax.text(0.018, 0.50, "RESULT", transform=ax.transAxes, fontfamily=F_SANS,
            fontsize=15, fontweight="bold", color=t["accent"], va="center")
    ax.plot([0.105, 0.105], [0.12, 0.88], color=t["rule"], lw=1.4,
            transform=ax.transAxes)

    # three cells
    cell_x = [0.135, 0.42, 0.71]
    cell_w = 0.265
    for (val, l1, l2), cx in zip(ROWS, cell_x):
        _round_box(ax, cx, 0.10, cell_w, 0.80, t["panel"], t["border"], lw=1.4, rad=0.06)
        col = t["good"] if (val.startswith("+") or "/" in val) else t["num"]
        ax.text(cx + cell_w/2, 0.66, val, transform=ax.transAxes, fontfamily=F_MONO,
                fontsize=30, fontweight="bold", color=col, ha="center", va="center")
        ax.text(cx + cell_w/2, 0.40, l1, transform=ax.transAxes, fontfamily=F_SANS,
                fontsize=12.5, color=t["label"], ha="center", va="center")
        ax.text(cx + cell_w/2, 0.22, l2, transform=ax.transAxes, fontfamily=F_SANS,
                fontsize=9.0, color=t["sub"], style="italic", ha="center", va="center")

    fig.savefig(OUT / f"RESULT_strip_{theme_name}.png", dpi=300,
                facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# ULTRA-MINIMAL 2-LINE  (smallest possible slot)
# ════════════════════════════════════════════════════════════════
def make_minimal(theme_name):
    t = THEMES[theme_name]
    fig = plt.figure(figsize=(13, 2.0))
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor(t["bg"])
    _round_box(ax, 0.01, 0.06, 0.98, 0.88, t["panel"], t["border"], lw=1.4, rad=0.06)
    # line 1
    ax.text(0.035, 0.66, "+28.26%", transform=ax.transAxes, fontfamily=F_MONO,
            fontsize=22, fontweight="bold", color=t["good"], va="center")
    ax.text(0.175, 0.66, "revenue over static EMSR pricing", transform=ax.transAxes,
            fontfamily=F_SANS, fontsize=15, color=t["head"], va="center")
    ax.text(0.605, 0.66, "(300 scenarios · 291/300 positive · 95% CI ±3.2%)",
            transform=ax.transAxes, fontfamily=F_SANS, fontsize=11, color=t["sub"],
            style="italic", va="center")
    # line 2
    ax.text(0.035, 0.30, "21% → 42%", transform=ax.transAxes, fontfamily=F_MONO,
            fontsize=18, fontweight="bold", color=t["num"], va="center")
    ax.text(0.205, 0.30, "economy load factor — gained by filling empty seats, not by cutting the average fare",
            transform=ax.transAxes, fontfamily=F_SANS, fontsize=12.5, color=t["label"], va="center")
    fig.savefig(OUT / f"RESULT_minimal_{theme_name}.png", dpi=300,
                facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


for th in ("dark", "light"):
    make_box(th)
    make_strip(th)
    make_minimal(th)

(OUT / "README.txt").write_text(
    "SEATWISE — KEY RESULT CALLOUT (poster, English)\n"
    "================================================\n\n"
    "Six files — pick by your poster's available slot shape and theme:\n\n"
    "  RESULT_box_dark.png      tall slot,  dark poster   (RECOMMENDED default)\n"
    "  RESULT_box_light.png     tall slot,  light poster\n"
    "  RESULT_strip_dark.png    wide/short slot, dark poster\n"
    "  RESULT_strip_light.png   wide/short slot, light poster\n"
    "  RESULT_minimal_dark.png  smallest possible slot, dark poster\n"
    "  RESULT_minimal_light.png smallest possible slot, light poster\n\n"
    "All three layouts carry the same three facts:\n"
    "  1) +28.26% revenue vs static EMSR pricing  (300-scenario Monte Carlo, 95% CI +-3.2%)\n"
    "  2) economy load factor 21% -> 42%  (by filling empty seats, not by discounting)\n"
    "  3) 291 / 300 scenarios beat the static baseline  (best +181%, only 9/300 below static)\n\n"
    "Why these three (and not more): together they answer every jury follow-up in minimum space\n"
    "  - 'compared to what?'        -> 'vs static EMSR pricing'\n"
    "  - 'one lucky run?'           -> '300-scenario Monte Carlo / 95% CI / 291 of 300'\n"
    "  - 'did you just discount?'   -> 'NOT by discounting / load factor 21% -> 42%'\n\n"
    "Sources\n-------\n"
    "+28.26% mean lift, sigma 27.84%  reports/validation_results.json (300 records) + report sec. 3.5.1\n"
    "95% CI +-3.2%                     1.96 * 27.84 / sqrt(300)\n"
    "21% -> 42% economy load factor    report sec. 3.3\n"
    "291/300 beat static, best +181%   computed from reports/validation_results.json\n",
    encoding="utf-8")

print("OK ->", OUT)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
