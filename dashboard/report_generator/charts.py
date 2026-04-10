"""
Chart Generator — Creates matplotlib figures for PDF embedding.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tempfile


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

_COLORS = {
    'V': '#94a3b8', 'K': '#c9a227', 'M': '#6366f1', 'Y': '#ef4444',
    'green': '#27ae60', 'red': '#e74c3c', 'blue': '#3498db',
    'orange': '#f39c12', 'purple': '#8e44ad',
}


def _save(fig, name):
    path = os.path.join(tempfile.gettempdir(), f"seatwise_chart_{name}.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def fare_class_pie(rd):
    """Fare class distribution pie chart."""
    fc = rd.fare_class_totals
    labels = ['V (Promo)', 'K (Discount)', 'M (Flex)', 'Y (Full)']
    sizes = [fc.get('V', 0), fc.get('K', 0), fc.get('M', 0), fc.get('Y', 0)]
    colors = [_COLORS['V'], _COLORS['K'], _COLORS['M'], _COLORS['Y']]

    if sum(sizes) == 0:
        return None

    fig, ax = plt.subplots(figsize=(4, 3))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight('bold')
    ax.set_title('Fare Class Distribution')
    return _save(fig, 'fc_pie')


def revenue_comparison(rd):
    """Revenue: dynamic vs baseline bar chart."""
    fig, ax = plt.subplots(figsize=(5, 3))
    cats = ['Baseline', 'Dynamic']
    vals = [rd.revenue_baseline, rd.revenue_dynamic]
    colors = [_COLORS['blue'], _COLORS['green']]

    bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')

    delta = rd.revenue_delta
    ax.set_title(f'Revenue Comparison (Delta: ${delta:,.0f}, {rd.revenue_delta_pct:+.1f}%)')
    ax.set_ylabel('Revenue ($)')
    return _save(fig, 'rev_compare')


def lf_by_flight(rd):
    """Load factor bar chart per flight."""
    flights = sorted(rd.flights, key=lambda f: f.get('load_factor', 0), reverse=True)[:10]
    if not flights:
        return None

    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [f"{f['route']}\n{f['cabin'][:3]}" for f in flights]
    values = [f.get('load_factor', 0) * 100 for f in flights]
    colors = [_COLORS['green'] if v >= 80 else _COLORS['orange'] if v >= 60 else _COLORS['red'] for v in values]

    ax.barh(range(len(labels)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Load Factor (%)')
    ax.set_title('Load Factor by Flight')
    ax.set_xlim(0, 110)
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=7, fontweight='bold')

    return _save(fig, 'lf_flights')


def region_performance(rd):
    """Region performance grouped bar."""
    regs = rd.region_performance
    if not regs:
        return None

    fig, ax = plt.subplots(figsize=(5, 3))
    labels = [r['region'] for r in regs]
    lfs = [r['avg_lf'] * 100 for r in regs]
    colors = [_COLORS['green'] if v >= 80 else _COLORS['orange'] if v >= 60 else _COLORS['red'] for v in lfs]

    bars = ax.bar(labels, lfs, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, lfs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold')

    ax.set_ylabel('Avg Load Factor (%)')
    ax.set_title('Regional Performance')
    ax.set_ylim(0, 110)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    return _save(fig, 'region_perf')


def generate_all(rd):
    """Generate all charts, return dict of name -> file path."""
    charts = {}
    for name, func in [
        ('fc_pie', fare_class_pie),
        ('rev_compare', revenue_comparison),
        ('lf_flights', lf_by_flight),
        ('region_perf', region_performance),
    ]:
        try:
            path = func(rd)
            if path:
                charts[name] = path
        except Exception as e:
            print(f"[Report] Chart {name} error: {e}")
    return charts
