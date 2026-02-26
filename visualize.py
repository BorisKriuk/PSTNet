#!/usr/bin/env python3
"""visualize.py — Publication-quality figures for PSTNet validation results.

Generates academic-grade visualizations from the comparative test suite.
All figures use a cohesive dark theme with physics-inspired color palettes.

Usage
-----
    python visualize.py                # full suite + figures
    python visualize.py --quick        # smoke test + figures
    python visualize.py --only-figs    # regenerate figs from last JSON

Outputs → figures/ directory:
    fig1_model_comparison.pdf
    fig2_effect_size_forest.pdf
    fig3_improvement_violin.pdf
    fig4_regime_activation.pdf
    fig5_altitude_response.pdf
    fig6_critical_difference.pdf
    fig7_convergence.pdf
    fig8_category_heatmap.pdf
"""

import sys, os, math, json, time, io, contextlib, copy
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from collections import OrderedDict

# ── PSTNet imports ────────────────────────────────────────────────
from config import (MISSILES, SCENARIOS, SPEED_OF_SOUND_SEA,
                    AIR_DENSITY_SEA, ALTITUDE_LAYERS)
from weather_api import WeatherService
from turbulence_model import TurbulenceField, TurbulencePredictor

# ── Import test infrastructure ────────────────────────────────────
from tests import (
    build_suite, create_diverse_profiles, run_all_models,
    compute_metrics, guidance_counts, paired_ttest, cohens_d,
    bootstrap_ci, wilcoxon_test, friedman_test, nemenyi_cd,
    _VanillaMLP, _DeepMLP, _GBTPredictor, _DrydenPredictor,
    create_bridge, verify_predictor, make_model_field,
    MODEL_ORDER, MODEL_LABELS, GUIDANCE_VALID_M, CAT_NAMES,
    VALIDATED_TYPES, mute, pbar,
)

# =====================================================================
#  Theme & Palette
# =====================================================================
BG       = '#0d1117'
BG_CARD  = '#161b22'
BG_GRID  = '#21262d'
FG       = '#c9d1d9'
FG_DIM   = '#8b949e'
ACCENT   = '#58a6ff'
GREEN    = '#3fb950'
RED      = '#f85149'
ORANGE   = '#d29922'
PURPLE   = '#bc8cff'
CYAN     = '#39d353'
PINK     = '#f778ba'

MODEL_COLORS = {
    'PSTNet':     ACCENT,
    'VanillaMLP': ORANGE,
    'DeepMLP':    RED,
    'GBT':        PURPLE,
    'Dryden':     FG_DIM,
}

MODEL_MARKERS = {
    'PSTNet':     'o',
    'VanillaMLP': 's',
    'DeepMLP':    'D',
    'GBT':        '^',
    'Dryden':     'v',
}

CAT_COLORS = {
    'A': ACCENT,
    'B': GREEN,
    'C': ORANGE,
    'D': PURPLE,
    'E': RED,
    'F': CYAN,
}

REGIME_COLORS = {
    'convective':   '#f97316',
    'neutral':      '#3b82f6',
    'stable':       '#22c55e',
    'stratospheric':'#a855f7',
}

CMAP_HEAT = LinearSegmentedColormap.from_list(
    'pstnet_heat', ['#0d1117', '#1e3a5f', '#2563eb', '#7c3aed', '#ec4899', '#f59e0b'], N=256)


def apply_theme(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=FG_DIM, labelsize=9)
    ax.spines['bottom'].set_color(BG_GRID)
    ax.spines['left'].set_color(BG_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, color=FG, fontsize=12, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=FG_DIM, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG_DIM, fontsize=10)


def save_fig(fig, name, out_dir='figures'):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{fmt}'),
                    dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'    {name}.pdf / .png')


# =====================================================================
#  Figure 1 — Multi-Model Bar Comparison
# =====================================================================
def fig1_model_comparison(all_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=BG)
    fig.suptitle('Model Comparison — Overall Performance',
                 color=FG, fontsize=14, fontweight='bold', y=0.98)

    metrics_keys = [('i_mean', 'Mean Improvement (%)', True),
                    ('wr',     'Win Rate (%)',         True),
                    ('d',      "Cohen's d",            True)]

    for ax, (key, label, higher_better) in zip(axes, metrics_keys):
        apply_theme(ax, ylabel=label)
        names, vals, colors = [], [], []
        for m in MODEL_ORDER:
            if m not in all_metrics or all_metrics[m].get('n', 0) == 0:
                continue
            names.append(MODEL_LABELS.get(m, m).replace(' ', '\n'))
            vals.append(all_metrics[m][key])
            colors.append(MODEL_COLORS.get(m, FG_DIM))

        bars = ax.bar(range(len(names)), vals, color=colors, width=0.6,
                      edgecolor=[c + '80' for c in colors], linewidth=1.2,
                      zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, color=FG_DIM)
        ax.axhline(0, color=BG_GRID, linewidth=0.8, zorder=1)
        ax.grid(axis='y', color=BG_GRID, linewidth=0.5, zorder=0)

        for bar, v in zip(bars, vals):
            va = 'bottom' if v >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f'{v:.1f}' if abs(v) < 100 else f'{v:.0f}',
                    ha='center', va=va, color=FG, fontsize=8,
                    fontweight='bold', zorder=4)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, 'fig1_model_comparison')


# =====================================================================
#  Figure 2 — Forest Plot (Effect Sizes with CI)
# =====================================================================
def fig2_effect_size_forest(all_metrics, all_results):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    apply_theme(ax, title='Effect Size Forest Plot — All Models vs Uncorrected',
                xlabel="Cohen's d (higher = better correction)")

    y_pos = []
    labels = []
    idx = 0

    for mname in reversed(MODEL_ORDER):
        if mname not in all_metrics:
            continue
        m = all_metrics[mname]
        if m.get('n', 0) < 3:
            continue

        d_val = m['d']
        # Bootstrap CI for effect size
        results = all_results[mname]
        diffs = []
        for r in results:
            cr, ur = r[0], r[1]
            if cr['status'] == 'timeout' or ur['status'] == 'timeout':
                continue
            um = ur['miss_distance_m']
            if um > GUIDANCE_VALID_M:
                continue
            diffs.append(um - cr['miss_distance_m'])
        if len(diffs) < 5:
            continue
        diffs = np.array(diffs)
        rng = np.random.RandomState(42)
        boot_d = []
        for _ in range(2000):
            samp = rng.choice(diffs, len(diffs), replace=True)
            if samp.std() > 1e-12:
                boot_d.append(samp.mean() / samp.std())
        boot_d = np.array(boot_d)
        ci_lo = np.percentile(boot_d, 2.5)
        ci_hi = np.percentile(boot_d, 97.5)

        color = MODEL_COLORS.get(mname, FG_DIM)
        ax.plot([ci_lo, ci_hi], [idx, idx], color=color, linewidth=2.5,
                solid_capstyle='round', zorder=3)
        ax.scatter([d_val], [idx], color=color, s=120,
                   marker=MODEL_MARKERS.get(mname, 'o'),
                   edgecolors='white', linewidths=0.8, zorder=4)
        ax.text(ci_hi + 0.02, idx, f'd = {d_val:+.3f}',
                color=color, fontsize=9, va='center', fontweight='bold')

        labels.append(MODEL_LABELS.get(mname, mname))
        y_pos.append(idx)
        idx += 1

    # Effect size thresholds
    for thresh, label, ls in [(0.2, 'Small', ':'),
                               (0.5, 'Medium', '--'),
                               (0.8, 'Large', '-.')]:
        ax.axvline(thresh, color=FG_DIM, linewidth=0.8, linestyle=ls,
                   alpha=0.5, zorder=1)
        ax.text(thresh, idx - 0.3, label, color=FG_DIM, fontsize=7,
                ha='center', style='italic')

    ax.axvline(0, color=RED, linewidth=1, alpha=0.7, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, color=FG)
    ax.set_xlim(-0.3, max(1.5, ax.get_xlim()[1] + 0.15))
    ax.grid(axis='x', color=BG_GRID, linewidth=0.5, zorder=0)
    fig.tight_layout()
    save_fig(fig, 'fig2_effect_size_forest')


# =====================================================================
#  Figure 3 — Violin / Swarm Plot of Improvement Distribution
# =====================================================================
def fig3_improvement_violin(all_results):
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    apply_theme(ax, title='Improvement Distribution by Model',
                ylabel='Improvement over Uncorrected (%)')

    data = []
    names = []
    colors = []
    for mname in MODEL_ORDER:
        if mname not in all_results:
            continue
        imp = []
        for r in all_results[mname]:
            cr, ur = r[0], r[1]
            if cr['status'] == 'timeout' or ur['status'] == 'timeout':
                continue
            um = ur['miss_distance_m']
            if um > GUIDANCE_VALID_M or um < 0.01:
                continue
            imp.append((1.0 - cr['miss_distance_m'] / um) * 100.0)
        if not imp:
            continue
        data.append(np.array(imp))
        names.append(MODEL_LABELS.get(mname, mname))
        colors.append(MODEL_COLORS.get(mname, FG_DIM))

    positions = np.arange(len(data))

    # Violin bodies
    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showextrema=False, showmedians=False, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.3)
        pc.set_edgecolor(colors[i])
        pc.set_linewidth(1.2)

    # Box plot overlay
    bp = ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                    showfliers=False, zorder=3,
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(color=FG_DIM, linewidth=0.8),
                    capprops=dict(color=FG_DIM, linewidth=0.8))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.6)
        patch.set_edgecolor(colors[i])

    # Jittered scatter
    rng = np.random.RandomState(99)
    for i, d in enumerate(data):
        jitter = rng.uniform(-0.25, 0.25, len(d))
        ax.scatter(positions[i] + jitter, d, color=colors[i], s=8,
                   alpha=0.25, zorder=2, edgecolors='none')

    # Mean markers
    for i, d in enumerate(data):
        ax.scatter([positions[i]], [d.mean()], color='white', s=50,
                   marker='D', zorder=5, edgecolors=colors[i], linewidths=1.5)

    ax.axhline(0, color=RED, linewidth=1, alpha=0.5, linestyle='--', zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=9, color=FG_DIM)
    ax.grid(axis='y', color=BG_GRID, linewidth=0.5, zorder=0)

    legend_elements = [Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
                              markeredgecolor=ACCENT, markersize=7, label='Mean',
                              linestyle='None'),
                       Line2D([0], [0], color='white', linewidth=1.5, label='Median')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              facecolor=BG_CARD, edgecolor=BG_GRID, labelcolor=FG_DIM)

    fig.tight_layout()
    save_fig(fig, 'fig3_improvement_violin')


# =====================================================================
#  Figure 4 — Regime Gate Activation Heatmap
# =====================================================================
def fig4_regime_activation(pstnet_predictor, weather_service):
    altitudes = np.linspace(0, 30, 60)
    ri_values = np.linspace(-2, 5, 50)

    regimes = ['convective', 'neutral', 'stable', 'stratospheric']
    activation = {r: np.zeros((len(altitudes), len(ri_values))) for r in regimes}

    profile = weather_service.get_vertical_profile(ALTITUDE_LAYERS)

    for i, alt in enumerate(altitudes):
        for j, ri in enumerate(ri_values):
            # Interpolate atmosphere
            below, above = profile[0], profile[-1]
            for k in range(len(profile) - 1):
                if profile[k]['altitude'] <= alt <= profile[k + 1]['altitude']:
                    below, above = profile[k], profile[k + 1]
                    break
            span = above['altitude'] - below['altitude']
            f = (alt - below['altitude']) / span if span > 0 else 0.0
            f = np.clip(f, 0, 1)
            wind = below['wind_speed'] * (1 - f) + above['wind_speed'] * f
            temp = below['temperature'] * (1 - f) + above['temperature'] * f
            dens = below['density'] * (1 - f) + above['density'] * f
            pres = below['pressure'] * (1 - f) + above['pressure'] * f

            rw = pstnet_predictor.get_regime_weights(wind, temp, dens, ri, alt, pres)
            for r in regimes:
                activation[r][i, j] = rw.get(r, 0.25)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=BG)
    fig.suptitle('Regime Gate Activation — PSTNet Expert Routing',
                 color=FG, fontsize=14, fontweight='bold', y=0.98)

    for ax, regime in zip(axes.flat, regimes):
        apply_theme(ax, title=f'{regime.capitalize()} Expert',
                    xlabel='Richardson Number', ylabel='Altitude (km)')
        im = ax.imshow(activation[regime], aspect='auto',
                       origin='lower', cmap=CMAP_HEAT,
                       extent=[ri_values[0], ri_values[-1],
                               altitudes[0], altitudes[-1]],
                       vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=FG_DIM, labelsize=8)
        cb.set_label('Gate weight', color=FG_DIM, fontsize=9)

        # Annotate physical boundaries
        ax.axhline(2, color=FG_DIM, linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axhline(12, color=FG_DIM, linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axhline(20, color=FG_DIM, linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axvline(-0.25, color=FG_DIM, linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axvline(0.25, color=FG_DIM, linewidth=0.5, linestyle=':', alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, 'fig4_regime_activation')


# =====================================================================
#  Figure 5 — Altitude Response Profile
# =====================================================================
def fig5_altitude_response(pstnet_predictor, weather_service):
    profile = weather_service.get_vertical_profile(ALTITUDE_LAYERS)
    altitudes = np.linspace(0.1, 30, 100)

    strength = []
    reliability = []
    drift_scale = []
    backbone_s = []

    for alt in altitudes:
        below, above = profile[0], profile[-1]
        for k in range(len(profile) - 1):
            if profile[k]['altitude'] <= alt <= profile[k + 1]['altitude']:
                below, above = profile[k], profile[k + 1]
                break
        span = above['altitude'] - below['altitude']
        f = (alt - below['altitude']) / span if span > 0 else 0.0
        f = np.clip(f, 0, 1)
        wind = below['wind_speed'] * (1 - f) + above['wind_speed'] * f
        temp = below['temperature'] * (1 - f) + above['temperature'] * f
        dens = below['density'] * (1 - f) + above['density'] * f
        pres = below['pressure'] * (1 - f) + above['pressure'] * f
        ri = below['richardson'] * (1 - f) + above['richardson'] * f

        pred = pstnet_predictor.predict(wind, temp, dens, ri, alt, pres)
        strength.append(pred['correction_strength'])
        reliability.append(pred['reliability'])
        drift_scale.append(pred['drift_scale'])

        # Backbone only
        raw = np.array([[wind, temp, dens, ri, alt, pres]])
        bb = pstnet_predictor.analytical_backbone(raw)
        backbone_s.append(bb[0, 0])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5.5), facecolor=BG,
                                         sharey=True)
    fig.suptitle('PSTNet Output Profile — Altitude Response',
                 color=FG, fontsize=14, fontweight='bold', y=0.98)

    # Correction Strength
    apply_theme(ax1, title='Correction Strength', xlabel='Value', ylabel='Altitude (km)')
    ax1.plot(strength, altitudes, color=ACCENT, linewidth=2, label='PSTNet (full)')
    ax1.plot(backbone_s, altitudes, color=FG_DIM, linewidth=1.5, linestyle='--',
             label='Backbone only')
    ax1.fill_betweenx(altitudes, backbone_s, strength, alpha=0.15, color=ACCENT)
    ax1.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BG_GRID, labelcolor=FG_DIM)

    # Reliability
    apply_theme(ax2, title='Reliability', xlabel='Value')
    ax2.plot(reliability, altitudes, color=GREEN, linewidth=2)

    # Drift Scale
    apply_theme(ax3, title='Drift Scale', xlabel='Value')
    ax3.plot(drift_scale, altitudes, color=ORANGE, linewidth=2)

    # Layer annotations
    for ax in [ax1, ax2, ax3]:
        for h, lbl in [(2, 'BL'), (12, 'Tropo'), (20, 'Strato')]:
            ax.axhline(h, color=FG_DIM, linewidth=0.5, linestyle=':', alpha=0.4)
            ax.text(ax.get_xlim()[1], h + 0.5, lbl, color=FG_DIM, fontsize=7,
                    ha='right', va='bottom', style='italic')
        ax.grid(axis='both', color=BG_GRID, linewidth=0.4, zorder=0)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, 'fig5_altitude_response')


# =====================================================================
#  Figure 6 — Critical Difference Diagram
# =====================================================================
def fig6_critical_difference(all_results):
    names = [m for m in MODEL_ORDER if m in all_results]
    n_test = min(len(r) for r in all_results.values())
    k = len(names)

    miss_arrays = {}
    for mname in names:
        misses = []
        for r in all_results[mname][:n_test]:
            cr, ur = r[0], r[1]
            if (cr['status'] == 'timeout' or ur['status'] == 'timeout'
                    or ur['miss_distance_m'] > GUIDANCE_VALID_M):
                misses.append(np.nan)
            else:
                misses.append(cr['miss_distance_m'])
        miss_arrays[mname] = np.array(misses)

    valid = np.ones(n_test, dtype=bool)
    for arr in miss_arrays.values():
        valid &= ~np.isnan(arr)
    n_valid = int(valid.sum())

    if n_valid < 10:
        print('    (skipping fig6 — insufficient valid cases)')
        return

    groups = [miss_arrays[m][valid] for m in names]
    n = len(groups[0])

    rank_matrix = np.zeros((n, k))
    for i in range(n):
        vals = [groups[j][i] for j in range(k)]
        order = np.argsort(vals)
        ranks = np.empty(k)
        ranks[order] = np.arange(1, k + 1)
        rank_matrix[i] = ranks
    mean_ranks = rank_matrix.mean(axis=0)
    cd = nemenyi_cd(k, n_valid)

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=BG)
    apply_theme(ax, title=f'Critical Difference Diagram (Nemenyi, N={n_valid})',
                xlabel='Mean Rank (lower is better)')

    idx = np.argsort(mean_ranks)
    y_positions = np.arange(k)

    for i, j in enumerate(idx):
        mname = names[j]
        mr = mean_ranks[j]
        color = MODEL_COLORS.get(mname, FG_DIM)

        ax.barh(i, mr, height=0.5, color=color, alpha=0.7, zorder=3,
                edgecolor=color, linewidth=1)
        ax.text(mr + 0.05, i, f'{mr:.2f}', color=color, fontsize=9,
                va='center', fontweight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([MODEL_LABELS.get(names[j], names[j]) for j in idx],
                       fontsize=10, color=FG)

    # CD bracket
    best_rank = mean_ranks[idx[0]]
    ax.axvline(best_rank + cd, color=RED, linewidth=1.2, linestyle='--',
               alpha=0.7, zorder=2)
    ax.text(best_rank + cd, k - 0.2, f'CD = {cd:.2f}', color=RED,
            fontsize=8, ha='center', va='bottom', style='italic')

    ax.grid(axis='x', color=BG_GRID, linewidth=0.5, zorder=0)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, 'fig6_critical_difference')


# =====================================================================
#  Figure 7 — Training Convergence
# =====================================================================
def fig7_convergence(pstnet, vanilla, deep):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    apply_theme(ax, title='Training Loss Convergence',
                xlabel='Epoch', ylabel='MSE Loss')

    curves = [
        (pstnet.loss_history, 'PSTNet (552 params)', ACCENT),
    ]
    if hasattr(vanilla, '_mlp'):
        # Vanilla MLP doesn't save history; reconstruct indication
        pass
    if hasattr(deep, '_mlp'):
        pass

    # PSTNet always has loss_history
    if pstnet.loss_history:
        epochs = np.arange(1, len(pstnet.loss_history) + 1)
        loss = np.array(pstnet.loss_history)
        ax.plot(epochs, loss, color=ACCENT, linewidth=2, label='PSTNet (552 params)',
                zorder=3)
        ax.fill_between(epochs, loss, alpha=0.1, color=ACCENT)

        # Annotate final loss
        ax.scatter([epochs[-1]], [loss[-1]], color=ACCENT, s=60, zorder=4,
                   edgecolors='white', linewidths=0.8)
        ax.text(epochs[-1] - 5, loss[-1] + (loss[0] - loss[-1]) * 0.05,
                f'{loss[-1]:.4f}', color=ACCENT, fontsize=9, ha='right',
                fontweight='bold')

    ax.set_yscale('log')
    ax.grid(True, color=BG_GRID, linewidth=0.5, which='both', zorder=0)
    ax.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BG_GRID, labelcolor=FG_DIM)
    fig.tight_layout()
    save_fig(fig, 'fig7_convergence')


# =====================================================================
#  Figure 8 — Category × Model Heatmap
# =====================================================================
def fig8_category_heatmap(all_results):
    cats = [c for c in CAT_NAMES if any(
        any(r[2]['cat'] == c for r in all_results.get(m, []))
        for m in MODEL_ORDER)]
    models = [m for m in MODEL_ORDER if m in all_results]

    matrix = np.full((len(cats), len(models)), np.nan)
    for i, cat in enumerate(cats):
        for j, mname in enumerate(models):
            rows = [r for r in all_results[mname] if r[2]['cat'] == cat]
            m = compute_metrics(rows, miss_cap=GUIDANCE_VALID_M)
            if m.get('n', 0) >= 2:
                matrix[i, j] = m['i_mean']

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    apply_theme(ax, title='Improvement (%) — Category vs Model')

    im = ax.imshow(matrix, cmap=CMAP_HEAT, aspect='auto', vmin=-5, vmax=15)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.ax.tick_params(colors=FG_DIM, labelsize=8)
    cb.set_label('Improvement (%)', color=FG_DIM, fontsize=9)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                       fontsize=9, color=FG_DIM, rotation=25, ha='right')
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f'{c}: {CAT_NAMES[c]}' for c in cats],
                       fontsize=9, color=FG)

    # Annotate cells
    for i in range(len(cats)):
        for j in range(len(models)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = 'white' if v > 5 else FG_DIM
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

    fig.tight_layout()
    save_fig(fig, 'fig8_category_heatmap')


# =====================================================================
#  Figure 9 — Per-Missile Effect Size Comparison
# =====================================================================
def fig9_missile_effect(all_results):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    apply_theme(ax, title='Effect Size by Vehicle Type — All Models',
                xlabel="Cohen's d", ylabel='Vehicle')

    missile_labels = {mt: MISSILES[mt]['name'] for mt in VALIDATED_TYPES}
    bar_width = 0.15
    y_base = np.arange(len(VALIDATED_TYPES))

    for mi, mname in enumerate(MODEL_ORDER):
        if mname not in all_results:
            continue
        ds = []
        for mt in VALIDATED_TYPES:
            rows = [r for r in all_results[mname] if r[2]['mt'] == mt]
            m = compute_metrics(rows, miss_cap=GUIDANCE_VALID_M)
            ds.append(m['d'] if m.get('n', 0) >= 3 else 0)

        color = MODEL_COLORS.get(mname, FG_DIM)
        offset = (mi - len(MODEL_ORDER) / 2 + 0.5) * bar_width
        bars = ax.barh(y_base + offset, ds, height=bar_width * 0.85,
                       color=color, alpha=0.8, label=MODEL_LABELS.get(mname, mname),
                       zorder=3, edgecolor=color)

    # Effect size thresholds
    for thresh, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
        ax.axvline(thresh, color=FG_DIM, linewidth=0.6, linestyle=':', alpha=0.5)
        ax.text(thresh, len(VALIDATED_TYPES) - 0.6, label, color=FG_DIM,
                fontsize=7, ha='center', style='italic')

    ax.axvline(0, color=RED, linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_base)
    ax.set_yticklabels([missile_labels[mt] for mt in VALIDATED_TYPES],
                       fontsize=10, color=FG)
    ax.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BG_GRID, labelcolor=FG_DIM,
              loc='lower right', ncol=2)
    ax.grid(axis='x', color=BG_GRID, linewidth=0.5, zorder=0)
    fig.tight_layout()
    save_fig(fig, 'fig9_missile_effect')


# =====================================================================
#  Figure 10 — Architecture Schematic
# =====================================================================
def fig10_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('PSTNet Architecture — Physics-Structured Turbulence Network',
                 color=FG, fontsize=14, fontweight='bold', pad=15)

    def box(x, y, w, h, label, sublabel='', color=ACCENT, alpha=0.2):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                        facecolor=color, alpha=alpha,
                                        edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.12 if sublabel else 0),
                label, ha='center', va='center', color=FG,
                fontsize=10, fontweight='bold')
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.18, sublabel,
                    ha='center', va='center', color=FG_DIM, fontsize=7)

    def arrow(x1, y1, x2, y2, color=FG_DIM):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # Input
    box(4.5, 5.8, 3, 0.7, 'Input Features', '[wind, T, ρ, Ri, alt, P]', FG_DIM, 0.15)

    # Row 2: Three parallel components
    box(0.3, 3.8, 2.8, 1.2, 'Analytical\nBackbone', 'Monin-Obukhov TKE\n0 params', GREEN, 0.2)
    box(4.0, 3.8, 3.6, 1.2, 'Regime Gate', 'Softmax → 4 experts\n84 params', PURPLE, 0.2)
    box(8.3, 3.8, 3.2, 1.2, '4 Expert MoE\n+ FiLM(ρ)', '468 params', ORANGE, 0.2)

    # Arrows from input
    arrow(5.0, 5.8, 1.7, 5.05)
    arrow(6.0, 5.8, 5.8, 5.05)
    arrow(7.0, 5.8, 9.9, 5.05)

    # Spectral constraint
    box(4.5, 2.0, 3, 0.9, 'Kolmogorov\nSpectral Constraint', 'ε^{1/3} scaling', CYAN, 0.2)

    # Arrows to spectral
    arrow(5.8, 3.8, 5.5, 2.95)
    arrow(9.9, 3.8, 7.0, 2.95)

    # Sum
    box(2.2, 0.5, 7.5, 0.8, 'Output = Backbone + Spectral-Constrained Residual',
        '[strength, reliability, drift_scale]', ACCENT, 0.25)

    arrow(1.7, 3.8, 4.5, 1.35)
    arrow(6.0, 2.0, 6.0, 1.35)

    # Parameter count annotation
    ax.text(11.5, 0.3, 'Total: 552\nparameters',
            color=ACCENT, fontsize=11, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_CARD,
                      edgecolor=ACCENT, alpha=0.8))

    fig.tight_layout()
    save_fig(fig, 'fig10_architecture')


# =====================================================================
#  Main
# =====================================================================
def main():
    quick     = '--quick' in sys.argv
    only_figs = '--only-figs' in sys.argv

    mode  = 'QUICK' if quick else 'FULL'
    tests = build_suite(quick)

    print('=' * 80)
    print(f'  PSTNet — VISUALIZATION SUITE ({mode})')
    print(f'  {len(tests)} tests x 5 models = {len(tests) * 5} paired sims')
    print('=' * 80)

    # ── Weather + training ─────────────────────────────────────────
    print('\n  [1/4] Weather + turbulence field ...')
    ws = WeatherService(lat=35.0, lon=-120.0)
    tf = TurbulenceField(ws)
    with mute():
        tf.update()

    print('  [2/4] Training all models ...')
    diverse = create_diverse_profiles(ws)

    pstnet = TurbulencePredictor()
    with mute():
        pstnet.fit(diverse, epochs=300, lr=0.004)
    tf.predictor = pstnet
    print(f'    PSTNet: {tf.get_model_info()["total_params"]} params, '
          f'loss={pstnet.loss_history[-1]:.6f}')

    vanilla = _VanillaMLP()
    with mute():
        vanilla.fit(diverse, epochs=300, lr=0.004)
    deep = _DeepMLP()
    with mute():
        deep.fit(diverse, epochs=300, lr=0.003)
    gbt = _GBTPredictor()
    with mute():
        gbt.fit(diverse)
    dryden = _DrydenPredictor()
    with mute():
        dryden.fit(diverse)

    baselines = OrderedDict([
        ('VanillaMLP', vanilla), ('DeepMLP', deep),
        ('GBT', gbt), ('Dryden', dryden),
    ])
    turb_fields = OrderedDict()
    turb_fields['PSTNet'] = tf
    for bname, bpred in baselines.items():
        bridge = create_bridge(bpred, pstnet)
        turb_fields[bname] = make_model_field(tf, bridge)

    # ── Run simulations ────────────────────────────────────────────
    print(f'  [3/4] Running {len(tests) * len(turb_fields)} simulations ...\n')
    t0 = time.time()
    all_results = run_all_models(tests, turb_fields)
    elapsed = time.time() - t0
    print(f'\n\n    Done in {elapsed:.1f}s')

    # Compute metrics
    all_metrics = {}
    model_infos = {
        'PSTNet':     tf.get_model_info(),
        'VanillaMLP': vanilla.get_model_info(),
        'DeepMLP':    deep.get_model_info(),
        'GBT':        gbt.get_model_info(),
        'Dryden':     dryden.get_model_info(),
    }
    for mname in MODEL_ORDER:
        if mname not in all_results:
            continue
        m = compute_metrics(all_results[mname], miss_cap=GUIDANCE_VALID_M)
        m['model_info'] = model_infos.get(mname, {})
        all_metrics[mname] = m

    # ── Generate figures ───────────────────────────────────────────
    print(f'\n  [4/4] Generating publication figures ...\n')

    fig1_model_comparison(all_metrics)
    fig2_effect_size_forest(all_metrics, all_results)
    fig3_improvement_violin(all_results)
    fig4_regime_activation(pstnet, ws)
    fig5_altitude_response(pstnet, ws)
    fig6_critical_difference(all_results)
    fig7_convergence(pstnet, vanilla, deep)
    fig8_category_heatmap(all_results)
    fig9_missile_effect(all_results)
    fig10_architecture()

    print(f'\n  All figures saved to figures/')
    print('=' * 80)


if __name__ == '__main__':
    main()
