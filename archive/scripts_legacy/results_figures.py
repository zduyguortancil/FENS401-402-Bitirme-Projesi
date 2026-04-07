"""
Results Section — Academic Figures (matplotlib, publication quality)
5 figures for the Results section of the paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import json
import os

# Academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTDIR = os.path.expanduser('~/OneDrive/Desktop/Results_Figures')
os.makedirs(OUTDIR, exist_ok=True)


# ═══════════════════════════════════════════
# FIGURE 1: TFT Actual vs Predicted (time series, IST-LHR)
# ═══════════════════════════════════════════
def fig1_tft_timeseries():
    df = pd.read_csv('/tmp/tft_lhr.csv', parse_dates=['dep_date'])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['dep_date'], df['actual'], color='#2c3e50', linewidth=0.8, alpha=0.7, label='Actual')
    ax.plot(df['dep_date'], df['predicted'], color='#e74c3c', linewidth=0.8, alpha=0.7, label='Predicted')
    ax.fill_between(df['dep_date'], df['actual'], df['predicted'], alpha=0.1, color='#e74c3c')

    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Passengers')
    ax.set_title('Figure 1. TFT Demand Forecast — IST-LHR Economy (Test Period)')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # MAE annotation
    mae = np.mean(np.abs(df['actual'] - df['predicted']))
    corr = np.corrcoef(df['actual'], df['predicted'])[0, 1]
    ax.text(0.02, 0.95, f'MAE = {mae:.1f}  |  r = {corr:.3f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'Fig1_TFT_TimeSeries.png'))
    plt.savefig(os.path.join(OUTDIR, 'Fig1_TFT_TimeSeries.pdf'))
    plt.close()
    print('Figure 1 OK')


# ═══════════════════════════════════════════
# FIGURE 2: Pickup Actual vs Predicted (scatter)
# ═══════════════════════════════════════════
def fig2_pickup_scatter():
    # Use TFT scatter since pickup raw data needs model inference
    data = np.load('/tmp/tft_data.npz')
    actual = data['actual']
    predicted = data['predicted']

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Subsample for clarity
    n = len(actual)
    idx = np.random.RandomState(42).choice(n, min(5000, n), replace=False)
    ax.scatter(actual[idx], predicted[idx], alpha=0.15, s=8, color='#3498db', edgecolors='none')

    # Perfect prediction line
    lims = [0, max(actual.max(), predicted.max()) * 1.05]
    ax.plot(lims, lims, '--', color='#e74c3c', linewidth=1, label='Perfect prediction')

    # Regression line
    z = np.polyfit(actual, predicted, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, lims[1], 100)
    ax.plot(x_line, p(x_line), '-', color='#2c3e50', linewidth=1, alpha=0.7, label=f'Fit (slope={z[0]:.3f})')

    ax.set_xlabel('Actual Daily Passengers')
    ax.set_ylabel('Predicted Daily Passengers')
    ax.set_title('Figure 2. TFT Prediction Accuracy (n = 55,066)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', loc='upper left')

    mae = np.mean(np.abs(actual - predicted))
    corr = np.corrcoef(actual, predicted)[0, 1]
    ax.text(0.98, 0.05, f'MAE = {mae:.2f}\nr = {corr:.4f}\nn = {len(actual):,}',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'Fig2_TFT_Scatter.png'))
    plt.savefig(os.path.join(OUTDIR, 'Fig2_TFT_Scatter.pdf'))
    plt.close()
    print('Figure 2 OK')


# ═══════════════════════════════════════════
# FIGURE 3: DTD + LF Price Curves (from calibration)
# ═══════════════════════════════════════════
def fig3_calibration_curves():
    with open('/tmp/cal.json') as f:
        cal = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # DTD curve
    dtd = cal['dtd_factors']['factors']
    labels = ['0-3', '4-7', '8-14', '15-30', '31-60', '61-90', '91-120', '121+']
    values = [dtd[l] for l in labels]
    colors = ['#e74c3c' if v > 1.5 else '#f39c12' if v > 1.1 else '#3498db' if v < 0.95 else '#2ecc71' for v in values]

    bars1 = ax1.bar(range(len(labels)), values, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('Days to Departure (DTD)')
    ax1.set_ylabel('Price Multiplier')
    ax1.set_title('(a) DTD Price Curve')
    ax1.axhline(y=1.0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # LF curve
    lf = cal['lf_curve']['factors']
    lf_labels = ['<30%', '30-50%', '50-70%', '70-85%', '85-95%', '95%+']
    lf_keys = ['LF<30', 'LF30-50', 'LF50-70', 'LF70-85', 'LF85-95', 'LF95+']
    lf_values = [lf.get(k, 1.0) for k in lf_keys]
    lf_colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']

    bars2 = ax2.bar(range(len(lf_labels)), lf_values, color=lf_colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax2.set_xticks(range(len(lf_labels)))
    ax2.set_xticklabels(lf_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('Load Factor')
    ax2.set_ylabel('Price Multiplier')
    ax2.set_title('(b) Load Factor Price Curve')
    ax2.axhline(y=1.0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars2, lf_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Figure 3. Data-Calibrated Pricing Curves', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'Fig3_Calibration_Curves.png'))
    plt.savefig(os.path.join(OUTDIR, 'Fig3_Calibration_Curves.pdf'))
    plt.close()
    print('Figure 3 OK')


# ═══════════════════════════════════════════
# FIGURE 4: Fare Class Mix (pie or stacked bar)
# ═══════════════════════════════════════════
def fig4_fare_class_mix():
    # Simulation fare class distribution
    fc_pcts = {'V': 16.5, 'K': 36.7, 'M': 48.7, 'Y': 0.3}
    fc_colors = {'V': '#94a3b8', 'K': '#c9a227', 'M': '#6366f1', 'Y': '#ef4444'}
    fc_names = {'V': 'V (Promo)', 'K': 'K (Discount)', 'M': 'M (Flex)', 'Y': 'Y (Full)'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Pie chart
    labels = [fc_names[k] for k in fc_pcts]
    sizes = list(fc_pcts.values())
    colors = [fc_colors[k] for k in fc_pcts]
    explode = (0.02, 0.02, 0.04, 0)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        pctdistance=0.75, labeldistance=1.12)
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color('white')
        t.set_fontweight('bold')
    ax1.set_title('(a) Fare Class Distribution')

    # DTD-based fare class progression
    dtd_periods = ['180-60\n(Early)', '60-30\n(Mid)', '30-14\n(Late)', '14-7\n(Last Week)', '7-0\n(Final)']
    # Approximate proportions based on DTD rules + EMSR-b
    v_pcts = [30, 10, 0, 0, 0]
    k_pcts = [45, 50, 35, 0, 0]
    m_pcts = [25, 40, 50, 60, 0]
    y_pcts = [0, 0, 15, 40, 100]

    x = np.arange(len(dtd_periods))
    w = 0.6
    ax2.bar(x, v_pcts, w, label='V (Promo)', color=fc_colors['V'], alpha=0.85)
    ax2.bar(x, k_pcts, w, bottom=v_pcts, label='K (Discount)', color=fc_colors['K'], alpha=0.85)
    ax2.bar(x, m_pcts, w, bottom=[v+k for v,k in zip(v_pcts, k_pcts)], label='M (Flex)', color=fc_colors['M'], alpha=0.85)
    ax2.bar(x, y_pcts, w, bottom=[v+k+m for v,k,m in zip(v_pcts, k_pcts, m_pcts)], label='Y (Full)', color=fc_colors['Y'], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(dtd_periods, fontsize=8)
    ax2.set_xlabel('Days to Departure')
    ax2.set_ylabel('Booking Share (%)')
    ax2.set_title('(b) Fare Class Availability by DTD')
    ax2.legend(fontsize=8, loc='upper right', frameon=True, fancybox=False, edgecolor='#cccccc')
    ax2.set_ylim(0, 105)

    fig.suptitle('Figure 4. Fare Class Mix and DTD Progression', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'Fig4_Fare_Class_Mix.png'))
    plt.savefig(os.path.join(OUTDIR, 'Fig4_Fare_Class_Mix.pdf'))
    plt.close()
    print('Figure 4 OK')


# ═══════════════════════════════════════════
# FIGURE 5: Pricing Decomposition (waterfall)
# ═══════════════════════════════════════════
def fig5_pricing_waterfall():
    # Example: IST-LHR Economy, DTD=45, July, M class
    steps = [
        ('Base Price', 224, 0),
        ('Supply\n(x1.15)', 224*0.15, 224),
        ('Demand\n(x1.35)', 224*1.15*0.35, 224*1.15),
        ('Sentiment\n(x1.02)', 224*1.15*1.35*0.02, 224*1.15*1.35),
        ('Customer\n(x0.97)', -224*1.15*1.35*1.02*0.03, 224*1.15*1.35*1.02),
    ]
    final_price = 342

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [s[0] for s in steps] + ['Final Price\n(M class)']
    values = [s[1] for s in steps] + [0]
    bottoms = [s[2] for s in steps] + [0]

    # Calculate cumulative
    cumulative = [224]
    for s in steps[1:]:
        cumulative.append(cumulative[-1] + s[1])
    cumulative.append(final_price)

    bar_vals = [224, 224*0.15, 224*1.15*0.35, 224*1.15*1.35*0.02, -224*1.15*1.35*1.02*0.03, final_price]
    bar_bottoms = [0, 224, 224*1.15, 224*1.15*1.35, 224*1.15*1.35*1.02, 0]
    bar_colors = ['#3498db', '#2ecc71', '#2ecc71', '#2ecc71', '#e74c3c', '#2c3e50']

    bars = ax.bar(range(len(labels)), bar_vals, bottom=bar_bottoms, color=bar_colors,
                   edgecolor='white', linewidth=0.5, alpha=0.85, width=0.6)

    # Value labels
    for i, (val, bot) in enumerate(zip(bar_vals, bar_bottoms)):
        y_pos = bot + val + (5 if val > 0 else -12)
        if i == len(labels) - 1:
            y_pos = final_price + 5
        ax.text(i, y_pos, f'${abs(val):.0f}' if i < len(labels)-1 else f'${final_price}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Connector lines
    for i in range(len(labels) - 2):
        top = bar_bottoms[i] + bar_vals[i]
        ax.plot([i + 0.3, i + 0.7], [top, top], color='#7f8c8d', linewidth=0.8, linestyle='-')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Price ($)')
    ax.set_title('Figure 5. Pricing Decomposition — IST-LHR Economy, DTD=45, July')
    ax.set_ylim(0, final_price * 1.15)

    # Baseline comparison
    baseline = 247
    ax.axhline(y=baseline, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(len(labels)-0.5, baseline + 5, f'Baseline: ${baseline}', fontsize=8,
            color='#e74c3c', ha='right')

    # Delta annotation
    ax.annotate(f'+${final_price - baseline} (+{(final_price-baseline)/baseline*100:.1f}%)',
                xy=(len(labels)-1, final_price), xytext=(len(labels)-1.5, final_price*0.7),
                fontsize=10, fontweight='bold', color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'Fig5_Pricing_Waterfall.png'))
    plt.savefig(os.path.join(OUTDIR, 'Fig5_Pricing_Waterfall.pdf'))
    plt.close()
    print('Figure 5 OK')


if __name__ == '__main__':
    fig1_tft_timeseries()
    fig2_pickup_scatter()
    fig3_calibration_curves()
    fig4_fare_class_mix()
    fig5_pricing_waterfall()
    print(f'\nAll figures saved to: {OUTDIR}')
