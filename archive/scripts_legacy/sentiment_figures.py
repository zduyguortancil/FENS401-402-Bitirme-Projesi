"""
Sentiment Analysis — Academic Figures
3 figures: Pipeline, Keyword Classifier, Scoring Mechanism
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import math
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTDIR = os.path.expanduser('~/OneDrive/Desktop/Sentiment_Figures')
os.makedirs(OUTDIR, exist_ok=True)

# Colors
C1 = '#2c3e50'  # dark
C2 = '#3498db'  # blue
C3 = '#e74c3c'  # red
C4 = '#27ae60'  # green
C5 = '#f39c12'  # orange
C6 = '#8e44ad'  # purple
C7 = '#1abc9c'  # teal
GRAY = '#95a5a6'
LIGHT = '#ecf0f1'


def box(ax, x, y, w, h, color, alpha=0.12, lw=1.5):
    r = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3',
                        facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw)
    ax.add_patch(r)


def arr(ax, x1, y1, x2, y2, color=GRAY, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


# ═══════════════════════════════════════════
# FIGURE A: Full Pipeline
# ═══════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    ax.text(50, 58, 'Sentiment Intelligence Pipeline', ha='center',
            fontsize=16, fontweight='bold', color=C1)
    ax.text(50, 56, '51 destination cities, real-time news monitoring',
            ha='center', fontsize=10, color=GRAY)

    # Stage 1: Data Sources
    box(ax, 3, 42, 20, 12, C2)
    ax.text(13, 52.5, 'GDELT API', ha='center', fontsize=11, fontweight='bold', color=C2)
    ax.text(13, 50, 'Global news database', ha='center', fontsize=8, color=GRAY)
    ax.text(13, 48, 'Built-in tone score', ha='center', fontsize=8, color=GRAY)
    ax.text(13, 46, '(-100 to +100)', ha='center', fontsize=8, color=GRAY)
    ax.text(13, 44, 'No API key required', ha='center', fontsize=7, color=C4)

    box(ax, 27, 42, 20, 12, C7)
    ax.text(37, 52.5, 'Google News RSS', ha='center', fontsize=11, fontweight='bold', color=C7)
    ax.text(37, 50, 'City + aviation query', ha='center', fontsize=8, color=GRAY)
    ax.text(37, 48, 'No tone score', ha='center', fontsize=8, color=GRAY)
    ax.text(37, 46, '(keyword fallback)', ha='center', fontsize=8, color=GRAY)
    ax.text(37, 44, 'Free, unlimited', ha='center', fontsize=7, color=C4)

    # Stage 2: Classification
    arr(ax, 23, 42, 38, 37, C2)
    arr(ax, 37, 42, 38, 37, C7)

    box(ax, 28, 27, 30, 10, C5)
    ax.text(43, 35.5, 'Keyword Classifier', ha='center', fontsize=12, fontweight='bold', color=C5)
    ax.text(43, 33, '~250 keywords, 8 event categories', ha='center', fontsize=9, color=GRAY)
    ax.text(43, 31, 'False positive filtering + aviation context check', ha='center', fontsize=8, color=GRAY)
    ax.text(43, 29, 'Microsecond inference, zero memory', ha='center', fontsize=7, color=C4)

    # Stage 3: Scoring
    arr(ax, 43, 27, 43, 22, C5)

    box(ax, 23, 12, 40, 10, C6)
    ax.text(43, 20.5, 'Article Scoring', ha='center', fontsize=12, fontweight='bold', color=C6)
    ax.text(43, 18, 'GDELT: score = 0.6 * tone_norm + 0.4 * event_impact', ha='center', fontsize=9, color=C1)
    ax.text(43, 16, 'RSS:   score = event_impact (no tone available)', ha='center', fontsize=9, color=C1)
    ax.text(43, 14, 'Label: positive (>0.05) | negative (<-0.05) | neutral', ha='center', fontsize=8, color=GRAY)

    # Stage 4: City Aggregation
    arr(ax, 43, 12, 43, 7, C6)

    box(ax, 15, -2, 56, 9, C3)
    ax.text(43, 5.5, 'City-Level Composite Score', ha='center', fontsize=12, fontweight='bold', color=C3)
    ax.text(43, 3, r'$S_{city} = \frac{\sum_i s_i \cdot e^{-\lambda h_i}}{\sum_i e^{-\lambda h_i}}$' +
            r'$\quad\quad \lambda = 0.1,\ \ h_i = $ article age (hours)',
            ha='center', fontsize=10, color=C1)
    ax.text(43, 0.5, '14-day recency filter  |  alert levels: LOW / MEDIUM / HIGH',
            ha='center', fontsize=8, color=GRAY)

    # Stage 5: Integration
    box(ax, 68, 32, 28, 20, C4)
    ax.text(82, 50.5, 'Pricing Integration', ha='center', fontsize=11, fontweight='bold', color=C4)
    ax.text(82, 47.5, 'Price multiplier:', ha='center', fontsize=9, color=GRAY)
    ax.text(82, 45, r'$m_{sent} = 1.0 + S_{city} \times 0.15$', ha='center', fontsize=10, color=C1)
    ax.text(82, 42, 'Range: 0.85 to 1.15', ha='center', fontsize=9, color=GRAY)
    ax.text(82, 39, 'Demand multiplier:', ha='center', fontsize=9, color=GRAY)
    ax.text(82, 37, r'$m_{demand} = 1.0 + S_{city} \times 0.30$', ha='center', fontsize=10, color=C1)
    ax.text(82, 34.5, 'Range: 0.70 to 1.30', ha='center', fontsize=9, color=GRAY)

    arr(ax, 63, 5, 68, 38, C3)

    # 51 cities annotation
    box(ax, 68, 12, 28, 14, C2, alpha=0.06)
    ax.text(82, 24, '51 Cities Monitored', ha='center', fontsize=10, fontweight='bold', color=C2)
    ax.text(82, 21.5, 'Europe: LHR, CDG, FRA, FCO, ...', ha='center', fontsize=7.5, color=GRAY)
    ax.text(82, 19.5, 'Middle East: DXB, DOH, RUH, ...', ha='center', fontsize=7.5, color=GRAY)
    ax.text(82, 17.5, 'Asia: NRT, SIN, BKK, DEL, ...', ha='center', fontsize=7.5, color=GRAY)
    ax.text(82, 15.5, 'Americas: JFK, MIA, GRU, ...', ha='center', fontsize=7.5, color=GRAY)
    ax.text(82, 13.5, 'Africa: CAI, JNB, NBO, ...', ha='center', fontsize=7.5, color=GRAY)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'FigA_Sentiment_Pipeline.png'))
    plt.savefig(os.path.join(OUTDIR, 'FigA_Sentiment_Pipeline.pdf'))
    plt.close()
    print('Figure A OK — Pipeline')


# ═══════════════════════════════════════════
# FIGURE B: Keyword Classifier Categories
# ═══════════════════════════════════════════
def fig_classifier():
    categories = [
        ('security_threat', 'Security Threat', -0.8, C3, 35,
         ['missile, bomb, terrorism', 'hostage, assassination', 'airspace closed, travel ban']),
        ('weather_disaster', 'Weather / Disaster', -0.7, '#e67e22', 30,
         ['hurricane, earthquake', 'flood, wildfire, tsunami', 'blizzard, heatwave']),
        ('health_crisis', 'Health Crisis', -0.7, C6, 25,
         ['pandemic, outbreak', 'quarantine, lockdown', 'travel restriction']),
        ('strike_protest', 'Strike / Protest', -0.6, C5, 22,
         ['strike, walkout, protest', 'pilot strike, ATC strike', 'riot, civil unrest']),
        ('political_instability', 'Political Instability', -0.6, '#c0392b', 18,
         ['coup, sanctions, embargo', 'martial law, curfew', 'border closure']),
        ('flight_disruption', 'Flight Disruption', -0.5, GRAY, 28,
         ['cancelled, delayed', 'grounded, diverted', 'airport closed']),
        ('tourism_growth', 'Tourism Growth', +0.5, C4, 40,
         ['tourism boom, record arrivals', 'hotel occupancy, cruise', 'festival, peak season']),
        ('positive_travel', 'Positive Travel', +0.4, C7, 35,
         ['new route, expansion', 'passenger record, award', 'visa-free, cheap flights']),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.5, len(categories) - 0.5)
    ax.set_xlabel('Event Impact Score', fontsize=12)
    ax.set_title('Keyword-Based Event Classification\n8 Categories, ~250 Keywords, Impact Scores',
                 fontsize=14, fontweight='bold', pad=15)

    # Zero line
    ax.axvline(x=0, color=GRAY, linestyle='-', linewidth=0.8, alpha=0.5)

    for i, (key, name, impact, color, n_kw, examples) in enumerate(categories):
        y = len(categories) - 1 - i

        # Impact bar
        ax.barh(y, impact, height=0.55, color=color, alpha=0.7, edgecolor=color, linewidth=1)

        # Category name
        side = 'right' if impact < 0 else 'left'
        x_txt = -0.02 if impact < 0 else 0.02
        ax.text(x_txt, y, f'  {name}  ', ha=side, va='center', fontsize=10,
                fontweight='bold', color=C1)

        # Impact value
        ax.text(impact + (0.03 if impact > 0 else -0.03), y,
                f'{impact:+.1f}', ha='left' if impact > 0 else 'right',
                va='center', fontsize=9, color=color, fontweight='bold')

        # Example keywords
        ex_str = ' | '.join(examples)
        x_ex = 0.55 if impact < 0 else -0.95
        ax.text(x_ex, y, ex_str, ha='left', va='center', fontsize=7, color=GRAY,
                style='italic')

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Negative / Positive labels
    ax.text(-0.5, len(categories) - 0.1, 'NEGATIVE IMPACT', ha='center',
            fontsize=9, color=C3, fontweight='bold', alpha=0.5)
    ax.text(0.5, len(categories) - 0.1, 'POSITIVE IMPACT', ha='center',
            fontsize=9, color=C4, fontweight='bold', alpha=0.5)

    # False positive box
    ax.text(0, -0.35, 'False positive filter: "box office bomb", "strike gold", "fire sale", "crash course", ...',
            ha='center', fontsize=8, color=GRAY, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT, edgecolor=GRAY, alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'FigB_Keyword_Classifier.png'))
    plt.savefig(os.path.join(OUTDIR, 'FigB_Keyword_Classifier.pdf'))
    plt.close()
    print('Figure B OK — Keyword Classifier')


# ═══════════════════════════════════════════
# FIGURE C: Recency Decay + Scoring Example
# ═══════════════════════════════════════════
def fig_scoring():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Recency decay curve
    hours = np.linspace(0, 336, 500)  # 14 days
    decay = np.exp(-0.1 * hours)

    ax1.plot(hours, decay, color=C2, linewidth=2)
    ax1.fill_between(hours, decay, alpha=0.1, color=C2)

    # Annotations
    key_points = [(1, 'fresh'), (24, '1 day'), (72, '3 days'), (168, '1 week'), (336, '14 days')]
    for h, label in key_points:
        w = math.exp(-0.1 * h)
        ax1.plot(h, w, 'o', color=C3, markersize=5, zorder=5)
        ax1.annotate(f'{label}\nw={w:.3f}', xy=(h, w), xytext=(h + 15, w + 0.08),
                     fontsize=8, color=C1,
                     arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8))

    ax1.axhline(y=0.01, color=C3, linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.text(300, 0.03, 'negligible', fontsize=8, color=C3, ha='center')

    ax1.set_xlabel('Article Age (hours)')
    ax1.set_ylabel('Recency Weight $e^{-0.1 \cdot h}$')
    ax1.set_title('(a) Exponential Recency Decay ($\\lambda$ = 0.1)')
    ax1.set_xlim(0, 350)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # (b) Scoring example — Barcelona scenario
    articles = [
        {'title': 'Barcelona airport strike announced', 'age_h': 2, 'score': -0.6, 'type': 'strike'},
        {'title': 'New terminal opening at BCN', 'age_h': 8, 'score': +0.4, 'type': 'positive'},
        {'title': 'Record tourism in Catalonia', 'age_h': 24, 'score': +0.5, 'type': 'tourism'},
        {'title': 'Barcelona festival draws crowds', 'age_h': 48, 'score': +0.3, 'type': 'tourism'},
        {'title': 'Spain weather warning', 'age_h': 120, 'score': -0.3, 'type': 'weather'},
    ]

    y_pos = np.arange(len(articles))
    weights = [math.exp(-0.1 * a['age_h']) for a in articles]
    weighted_scores = [a['score'] * w for a, w in zip(articles, weights)]
    composite = sum(weighted_scores) / sum(weights)

    colors_bar = [C3 if a['score'] < 0 else C4 for a in articles]

    bars = ax2.barh(y_pos, [a['score'] for a in articles], height=0.4,
                     color=colors_bar, alpha=0.3, edgecolor=colors_bar, label='Raw score')
    bars2 = ax2.barh(y_pos + 0.4, weighted_scores, height=0.4,
                      color=colors_bar, alpha=0.7, edgecolor=colors_bar, label='Weighted score')

    ax2.set_yticks(y_pos + 0.2)
    labels = [f'{a["title"][:30]}...\n({a["age_h"]}h old, w={w:.2f})'
              for a, w in zip(articles, weights)]
    ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.set_xlabel('Sentiment Score')
    ax2.set_title('(b) City Score Calculation — Barcelona Example')
    ax2.axvline(x=0, color=GRAY, linewidth=0.8, alpha=0.5)
    ax2.legend(fontsize=8, loc='lower right', frameon=True, fancybox=False, edgecolor=GRAY)

    # Composite score annotation
    ax2.axvline(x=composite, color=C6, linewidth=2, linestyle='--')
    ax2.text(composite + 0.02, len(articles) - 0.5,
             f'$S_{{city}}$ = {composite:.3f}\n$m_{{sent}}$ = {1 + composite * 0.15:.3f}',
             fontsize=10, fontweight='bold', color=C6,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C6, alpha=0.9))

    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Sentiment Scoring Mechanism', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'FigC_Scoring_Mechanism.png'))
    plt.savefig(os.path.join(OUTDIR, 'FigC_Scoring_Mechanism.pdf'))
    plt.close()
    print('Figure C OK — Scoring Mechanism')


if __name__ == '__main__':
    fig_pipeline()
    fig_classifier()
    fig_scoring()
    print(f'\nAll sentiment figures saved to: {OUTDIR}')
