#!/usr/bin/env python3
"""tests.py — Comprehensive paired-comparison test suite for PSTNet

Runs 330+ corrected-vs-uncorrected trajectory pairs under identical
perturbation seeds across diverse flight regimes.  The 4 dashboard
scenarios are category A; categories B–F probe altitude sweeps, range
variation, lateral offsets, edge-cases, and pure Monte Carlo spread.

Usage
-----
  python tests.py                   # full suite  (~330 pairs)
  python tests.py --quick           # smoke test  (~80 pairs)
  python tests.py --json results.json   # export detailed JSON
"""

import sys, os, time, math, json, io, contextlib
import numpy as np
from collections import OrderedDict
from datetime import datetime

from config import MISSILES, SCENARIOS, SPEED_OF_SOUND_SEA, AIR_DENSITY_SEA
from weather_api import WeatherService
from turbulence_model import TurbulenceField
from trajectory import MissileTrajectory

# =====================================================================
#  Helpers
# =====================================================================
SEP  = '═' * 80
THIN = '─' * 80

@contextlib.contextmanager
def mute():
    """Suppress print output from inner simulation code."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def pbar(cur, tot, width=40, tag=''):
    """Inline progress bar."""
    f = cur / max(tot, 1)
    filled = int(width * f)
    b = '█' * filled + '░' * (width - filled)
    sys.stdout.write(f'\r  {tag:>12} [{b}] {f*100:5.1f}%  ({cur}/{tot})')
    sys.stdout.flush()


# =====================================================================
#  Core runner — one corrected + one uncorrected, same seed
# =====================================================================
def run_pair(missile_type, turb_field, start, target, alt, seed, max_time=3000):
    """Return (corrected_result, uncorrected_result) impact dicts."""
    corr   = MissileTrajectory(missile_type, turb_field, True,  seed)
    uncorr = MissileTrajectory(missile_type, turb_field, False, seed)
    corr.launch(start, target, alt)
    uncorr.launch(start, target, alt)

    speed = MISSILES[missile_type]['speed'] * SPEED_OF_SOUND_SEA
    dt = min(0.5, 0.12 / speed)

    cr = ur = None
    t = 0.0
    while t < max_time:
        if cr is None:
            r = corr.step(dt)
            if r and r['status'] == 'impact':
                cr = r
        if ur is None:
            r = uncorr.step(dt)
            if r and r['status'] == 'impact':
                ur = r
        if cr is not None and ur is not None:
            break
        t += dt

    timeout = dict(status='timeout', miss_distance_m=1e6, time=max_time,
                   avg_turbulence=0, max_cross_track_m=1e6, corrected=None)
    return (cr or timeout), (ur or timeout)


# =====================================================================
#  Test-suite builder
# =====================================================================
def build_suite(quick=False):
    nA = 3  if quick else 15   # seeds per standard scenario
    nB = 2  if quick else 5    # seeds per altitude config
    nC = 2  if quick else 5    # seeds per range config
    nD = 2  if quick else 5    # seeds per offset config
    nE = 2  if quick else 5    # seeds per stress config
    nF = 4  if quick else 20   # pure Monte Carlo seeds per missile

    T = []

    # ── A: Monte Carlo on the 4 visualised scenarios ──────────────
    for sid, cfg in SCENARIOS.items():
        for i in range(nA):
            T.append(dict(cat='A', grp=sid, lbl=cfg['name'],
                          mt=cfg['missile_type'],
                          s=[30.0, 100.0], t=[150.0, 100.0],
                          alt=cfg['launch_alt'], seed=i * 137 + 7))

    # ── B: Altitude sweep  (3 altitudes per missile) ──────────────
    alt_map = {
        'SUBSONIC':        [0.5, 2.0, 5.0],
        'SUPERSONIC':      [5.0, 10.0, 15.0],
        'HIGH_SUPERSONIC': [12.0, 18.0, 22.0],
        'HYPERSONIC':      [18.0, 25.0, 30.0],
    }
    for mt, alts in alt_map.items():
        for a in alts:
            for i in range(nB):
                T.append(dict(cat='B', grp=f'{mt}@{a}km',
                              lbl=f'{mt} {a}km',
                              mt=mt, s=[30.0, 100.0], t=[150.0, 100.0],
                              alt=a, seed=i * 251 + 13))

    # ── C: Range sweep  (60 / 120 / 180 km) ──────────────────────
    for mt in MISSILES:
        ca = MISSILES[mt].get('cruise_altitude', 10)
        for rng in [60, 120, 180]:
            for i in range(nC):
                T.append(dict(cat='C', grp=f'{mt}_R{rng}',
                              lbl=f'{mt} {rng}km',
                              mt=mt, s=[30.0, 100.0],
                              t=[30.0 + rng, 100.0],
                              alt=ca, seed=i * 373 + 19))

    # ── D: Lateral offset targets  (-30 / 0 / +30 km) ────────────
    for mt in MISSILES:
        ca = MISSILES[mt].get('cruise_altitude', 10)
        for off in [-30, 0, 30]:
            for i in range(nD):
                T.append(dict(cat='D', grp=f'{mt}_L{off:+d}',
                              lbl=f'{mt} off={off:+d}km',
                              mt=mt, s=[30.0, 100.0],
                              t=[150.0, 100.0 + off],
                              alt=ca, seed=i * 499 + 23))

    # ── E: Stress / edge cases ────────────────────────────────────
    stress = [
        ('SUBSONIC',        0.3,  [30, 100], [150, 100], 'ultra-low 0.3km'),
        ('SUBSONIC',        1.0,  [30, 100], [70, 100],  'short 40km range'),
        ('SUPERSONIC',      3.0,  [30, 100], [150, 100], 'low-for-type 3km'),
        ('SUPERSONIC',      15.0, [30, 100], [150, 100], 'high-for-type 15km'),
        ('HIGH_SUPERSONIC', 12.0, [30, 100], [150, 135], 'oblique 12km'),
        ('HIGH_SUPERSONIC', 22.0, [30, 100], [180, 100], 'extended 22km'),
        ('HYPERSONIC',      30.0, [30, 100], [150, 100], 'max altitude 30km'),
        ('HYPERSONIC',      18.0, [30, 100], [230, 100], 'long range 200km'),
    ]
    for mt, a, s, tgt, desc in stress:
        for i in range(nE):
            T.append(dict(cat='E', grp=f'{mt}_{desc[:15]}',
                          lbl=f'{mt} {desc}',
                          mt=mt, s=s, t=tgt, alt=a, seed=i * 613 + 29))

    # ── F: Pure stochastic spread (many seeds, two missiles) ──────
    for i in range(nF):
        T.append(dict(cat='F', grp='MC_SUPERSONIC',
                      lbl='SUPERSONIC Monte Carlo',
                      mt='SUPERSONIC', s=[30, 100], t=[150, 100],
                      alt=10, seed=i * 7 + 1))
    for i in range(nF):
        T.append(dict(cat='F', grp='MC_SUBSONIC',
                      lbl='SUBSONIC Monte Carlo',
                      mt='SUBSONIC', s=[30, 100], t=[150, 100],
                      alt=2, seed=i * 11 + 3))
    for i in range(nF):
        T.append(dict(cat='F', grp='MC_HYPERSONIC',
                      lbl='HYPERSONIC Monte Carlo',
                      mt='HYPERSONIC', s=[30, 100], t=[150, 100],
                      alt=25, seed=i * 17 + 5))

    return T


# =====================================================================
#  Metrics computation
# =====================================================================
def compute_metrics(results):
    """Aggregate statistics from list of (cr, ur, test_info) tuples."""
    valid = [(r[0]['miss_distance_m'], r[1]['miss_distance_m'],
              r[0], r[1], r[2])
             for r in results
             if r[0]['status'] != 'timeout' and r[1]['status'] != 'timeout']
    if not valid:
        return dict(n=0)

    cm = np.array([v[0] for v in valid])
    um = np.array([v[1] for v in valid])
    imp = (1.0 - cm / np.maximum(um, 0.01)) * 100.0
    wins = int(np.sum(cm < um))
    ties = int(np.sum(np.abs(cm - um) < 0.01))

    ct_c = np.array([v[2].get('max_cross_track_m', 0) for v in valid])
    ct_u = np.array([v[3].get('max_cross_track_m', 0) for v in valid])
    turb = np.array([v[2].get('avg_turbulence', 0) for v in valid])

    return dict(
        n       = len(valid),
        c_mean  = float(cm.mean()),
        c_med   = float(np.median(cm)),
        c_std   = float(cm.std()),
        c_max   = float(cm.max()),
        c_p90   = float(np.percentile(cm, 90)),
        u_mean  = float(um.mean()),
        u_med   = float(np.median(um)),
        u_std   = float(um.std()),
        u_max   = float(um.max()),
        u_p90   = float(np.percentile(um, 90)),
        i_mean  = float(imp.mean()),
        i_med   = float(np.median(imp)),
        i_std   = float(imp.std()),
        i_min   = float(imp.min()),
        i_max   = float(imp.max()),
        i_p25   = float(np.percentile(imp, 25)),
        i_p75   = float(np.percentile(imp, 75)),
        win     = wins,
        lose    = len(valid) - wins - ties,
        tie     = ties,
        wr      = float(wins / len(valid) * 100),
        ct_c    = float(ct_c.mean()),
        ct_u    = float(ct_u.mean()),
        turb_avg= float(turb.mean()),
    )


# =====================================================================
#  Reporting
# =====================================================================
CAT_NAMES = OrderedDict([
    ('A', 'Monte Carlo — Standard 4 Scenarios'),
    ('B', 'Altitude Sweep'),
    ('C', 'Range Sweep'),
    ('D', 'Lateral Offset Targets'),
    ('E', 'Stress / Edge Cases'),
    ('F', 'Pure Stochastic Spread'),
])


def heading(title, ch='═'):
    print(f'\n{ch * 80}\n  {title}\n{ch * 80}')


def print_category(cat, all_results):
    """Print table for one test category."""
    rows = [r for r in all_results if r[2]['cat'] == cat]
    if not rows:
        return
    heading(f'CATEGORY {cat} — {CAT_NAMES[cat]}')

    hdr = (f'  {"Config":<34} {"N":>3}  '
           f'{"Corr CEP":>9}  {"Uncorr CEP":>10}  '
           f'{"Improv":>7}  {"Win%":>6}')
    print(hdr)
    print(f'  {THIN[:76]}')

    groups = list(dict.fromkeys(r[2]['grp'] for r in rows))
    for g in groups:
        gr = [r for r in rows if r[2]['grp'] == g]
        m = compute_metrics(gr)
        if m['n'] == 0:
            continue
        lbl = gr[0][2]['lbl']
        if len(lbl) > 34:
            lbl = lbl[:31] + '…'
        c_s = f'{m["c_med"]:.1f}m'
        u_s = f'{m["u_med"]:.1f}m'
        i_s = f'{m["i_mean"]:.1f}%'
        w_s = f'{m["wr"]:.1f}%'
        print(f'  {lbl:<34} {m["n"]:>3}  {c_s:>9}  {u_s:>10}  {i_s:>7}  {w_s:>6}')

    m = compute_metrics(rows)
    if m['n']:
        c_s = f'{m["c_med"]:.1f}m'
        u_s = f'{m["u_med"]:.1f}m'
        i_s = f'{m["i_mean"]:.1f}%'
        w_s = f'{m["wr"]:.1f}%'
        print(f'  {THIN[:76]}')
        print(f'  {"SUBTOTAL":<34} {m["n"]:>3}  '
              f'{c_s:>9}  {u_s:>10}  {i_s:>7}  {w_s:>6}')


def print_by_missile(all_results):
    heading('BREAKDOWN BY MISSILE TYPE')
    hdr = (f'  {"Type":<24} {"N":>3}  {"Corr":>8}  {"Uncorr":>9}  '
           f'{"Improv":>7}  {"Win%":>6}  {"ΔXtrack":>8}')
    print(hdr)
    print(f'  {THIN[:72]}')
    for mt in MISSILES:
        rows = [r for r in all_results if r[2]['mt'] == mt]
        m = compute_metrics(rows)
        if m['n'] == 0:
            continue
        ct_imp = (1.0 - m['ct_c'] / max(m['ct_u'], 0.01)) * 100
        print(f'  {MISSILES[mt]["name"]:<24} {m["n"]:>3}  '
              f'{m["c_mean"]:.1f}m'.ljust(10) +
              f'{m["u_mean"]:.1f}m'.ljust(11) +
              f'{m["i_mean"]:.1f}%'.rjust(7) +
              f'  {m["wr"]:.1f}%'.rjust(8) +
              f'  {ct_imp:.1f}%'.rjust(9))


def print_by_altitude(all_results):
    heading('BREAKDOWN BY ALTITUDE BAND')
    bands = [
        ('Surface  < 2 km',   0,  2),
        ('Low      2 – 5 km', 2,  5),
        ('Mid      5 – 12 km',5, 12),
        ('High    12 – 20 km',12,20),
        ('Strat    > 20 km',  20,99),
    ]
    print(f'  {"Band":<24} {"N":>3}  '
          f'{"Corr CEP":>9}  {"Uncorr CEP":>10}  '
          f'{"Improv":>7}  {"Win%":>6}')
    print(f'  {THIN[:68]}')
    for name, lo, hi in bands:
        rows = [r for r in all_results if lo <= r[2]['alt'] < hi]
        m = compute_metrics(rows)
        if m['n'] == 0:
            continue
        print(f'  {name:<24} {m["n"]:>3}  '
              f'{m["c_med"]:>7.1f}m  {m["u_med"]:>8.1f}m  '
              f'{m["i_mean"]:>6.1f}%  {m["wr"]:>5.1f}%')


def print_by_turbulence(all_results):
    heading('BREAKDOWN BY TURBULENCE SEVERITY')
    bands = [
        ('Low       < 15%',  0.00, 0.15),
        ('Moderate 15 – 30%',0.15, 0.30),
        ('High     30 – 50%',0.30, 0.50),
        ('Severe    > 50%',  0.50, 1.00),
    ]
    print(f'  {"Severity":<24} {"N":>3}  '
          f'{"Corr":>8}  {"Uncorr":>9}  '
          f'{"Improv":>7}  {"Win%":>6}')
    print(f'  {THIN[:64]}')
    for name, lo, hi in bands:
        rows = [r for r in all_results
                if lo <= r[0].get('avg_turbulence', 0) < hi
                and r[0]['status'] != 'timeout']
        m = compute_metrics(rows)
        if m['n'] == 0:
            continue
        print(f'  {name:<24} {m["n"]:>3}  '
              f'{m["c_mean"]:>6.1f}m  {m["u_mean"]:>7.1f}m  '
              f'{m["i_mean"]:>6.1f}%  {m["wr"]:>5.1f}%')


def print_by_range(all_results):
    heading('BREAKDOWN BY ENGAGEMENT RANGE')
    bands = [
        ('Short   < 80 km',  0,  80),
        ('Medium 80–140 km', 80, 140),
        ('Long    > 140 km',140, 999),
    ]
    print(f'  {"Range":<24} {"N":>3}  '
          f'{"Corr CEP":>9}  {"Uncorr CEP":>10}  '
          f'{"Improv":>7}  {"Win%":>6}')
    print(f'  {THIN[:68]}')
    for name, lo, hi in bands:
        rows = []
        for r in all_results:
            dx = r[2]['t'][0] - r[2]['s'][0]
            dy = r[2]['t'][1] - r[2]['s'][1]
            rng = math.sqrt(dx * dx + dy * dy)
            if lo <= rng < hi:
                rows.append(r)
        m = compute_metrics(rows)
        if m['n'] == 0:
            continue
        print(f'  {name:<24} {m["n"]:>3}  '
              f'{m["c_med"]:>7.1f}m  {m["u_med"]:>8.1f}m  '
              f'{m["i_mean"]:>6.1f}%  {m["wr"]:>5.1f}%')


def print_histogram(all_results):
    """ASCII histogram of improvement percentages."""
    pairs = [(r[0]['miss_distance_m'], r[1]['miss_distance_m'])
             for r in all_results
             if r[0]['status'] != 'timeout' and r[1]['status'] != 'timeout'
             and r[1]['miss_distance_m'] > 0.01]
    if not pairs:
        return
    imp = [(1.0 - c / u) * 100.0 for c, u in pairs]

    heading('IMPROVEMENT DISTRIBUTION (histogram)')
    edges = list(range(-30, 101, 10))
    cnts = [0] * (len(edges) - 1)
    for v in imp:
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                cnts[i] += 1
                placed = True
                break
        if not placed and v >= edges[-1]:
            cnts[-1] += 1

    mx = max(cnts) if cnts else 1
    for i, c in enumerate(cnts):
        lo, hi = edges[i], edges[i + 1]
        blen = int(c / mx * 45)
        bl = '█' * blen
        marker = ' ◄── zero' if lo <= 0 < hi else ''
        print(f'  {lo:>+4} to {hi:>+3}% │{bl} {c}{marker}')
    print(f'\n  Mean: {np.mean(imp):.1f}%   '
          f'Median: {np.median(imp):.1f}%   '
          f'IQR: [{np.percentile(imp,25):.1f}%, {np.percentile(imp,75):.1f}%]')


def print_extremes(all_results, n=5):
    """Show best and worst individual runs."""
    ok = [(r[0]['miss_distance_m'], r[1]['miss_distance_m'], r[2])
          for r in all_results
          if r[0]['status'] != 'timeout' and r[1]['status'] != 'timeout'
          and r[1]['miss_distance_m'] > 0.01]
    if not ok:
        return
    ranked = sorted(ok, key=lambda x: (1.0 - x[0] / x[1]))
    heading('EXTREME CASES')
    print(f'  {n} Worst (least / negative improvement):')
    for c, u, info in ranked[:n]:
        iv = (1.0 - c / u) * 100.0
        print(f'    {info["lbl"]:<38} corr={c:>7.1f}m  '
              f'uncorr={u:>7.1f}m  Δ={iv:>+6.1f}%')
    print(f'\n  {n} Best (most improvement):')
    for c, u, info in ranked[-n:]:
        iv = (1.0 - c / u) * 100.0
        print(f'    {info["lbl"]:<38} corr={c:>7.1f}m  '
              f'uncorr={u:>7.1f}m  Δ={iv:>+6.1f}%')


def print_consistency(all_results):
    """Show that correction never makes things dramatically worse."""
    ok = [(r[0]['miss_distance_m'], r[1]['miss_distance_m'])
          for r in all_results
          if r[0]['status'] != 'timeout' and r[1]['status'] != 'timeout']
    if not ok:
        return
    cm = np.array([o[0] for o in ok])
    um = np.array([o[1] for o in ok])
    imp = (1.0 - cm / np.maximum(um, 0.01)) * 100.0

    heading('CONSISTENCY ANALYSIS')
    thresholds = [0, -5, -10, -20]
    for th in thresholds:
        worse = int(np.sum(imp < th))
        pct = worse / len(imp) * 100
        label = f'correction > {abs(th)}% worse' if th < 0 else 'correction worse at all'
        print(f'  Cases where {label}: {worse:>4} / {len(imp)}  ({pct:.1f}%)')

    print(f'\n  Improvement percentiles:')
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f'    P{p:<2}: {np.percentile(imp, p):>+7.1f}%')


def print_overall(all_results, elapsed):
    """Grand summary with statistical test."""
    m = compute_metrics(all_results)
    if m['n'] == 0:
        print('\n  No valid results.')
        return m

    diffs = np.array([r[1]['miss_distance_m'] - r[0]['miss_distance_m']
                      for r in all_results
                      if r[0]['status'] != 'timeout'
                      and r[1]['status'] != 'timeout'])
    n = len(diffs)
    se = diffs.std(ddof=1) / math.sqrt(n) + 1e-12
    t_stat = diffs.mean() / se
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2))))

    # 95% CI for mean improvement
    ci_lo = m['i_mean'] - 1.96 * m['i_std'] / math.sqrt(m['n'])
    ci_hi = m['i_mean'] + 1.96 * m['i_std'] / math.sqrt(m['n'])

    heading('OVERALL RESULTS')
    print(f'''
  Total paired runs:          {m['n']:>6}
  Timeouts:                   {len(all_results) - m['n']:>6}
  Wall-clock time:            {elapsed:>6.1f} s   ({elapsed / max(m['n'], 1):.2f} s/pair)

  ┌──────────────────────────────────────────────────────────┐
  │  CORRECTED  (PSTNet ML active)                           │
  │    Mean miss distance:        {m['c_mean']:>8.1f} m               │
  │    Median (CEP):              {m['c_med']:>8.1f} m               │
  │    90th percentile:           {m['c_p90']:>8.1f} m               │
  │    Std deviation:             {m['c_std']:>8.1f} m               │
  │    Worst single case:         {m['c_max']:>8.1f} m               │
  ├──────────────────────────────────────────────────────────┤
  │  UNCORRECTED  (physics only)                             │
  │    Mean miss distance:        {m['u_mean']:>8.1f} m               │
  │    Median (CEP):              {m['u_med']:>8.1f} m               │
  │    90th percentile:           {m['u_p90']:>8.1f} m               │
  │    Std deviation:             {m['u_std']:>8.1f} m               │
  │    Worst single case:         {m['u_max']:>8.1f} m               │
  ├──────────────────────────────────────────────────────────┤
  │  IMPROVEMENT                                             │
  │    Mean:                      {m['i_mean']:>7.1f} %               │
  │    Median:                    {m['i_med']:>7.1f} %               │
  │    IQR:            [{m['i_p25']:>+6.1f}%, {m['i_p75']:>+5.1f}%]               │
  │    Best single case:          {m['i_max']:>7.1f} %               │
  │    Worst single case:         {m['i_min']:>7.1f} %               │
  │    Win / Lose / Tie:  {m['win']:>4} / {m['lose']:>4} / {m['tie']:>4}                │
  │    Win rate:                  {m['wr']:>6.1f} %                │
  │    95% CI for mean:  [{ci_lo:>+6.1f}%, {ci_hi:>+5.1f}%]               │
  ├──────────────────────────────────────────────────────────┤
  │  CROSS-TRACK ERROR                                       │
  │    Mean max cross-track (corr):   {m['ct_c']:>8.1f} m           │
  │    Mean max cross-track (uncorr): {m['ct_u']:>8.1f} m           │''')
    ct_imp = (1.0 - m['ct_c'] / max(m['ct_u'], 0.01)) * 100
    print(f'  │    Cross-track reduction:          {ct_imp:>6.1f} %           │')
    print(f'  ├──────────────────────────────────────────────────────────┤')
    print(f'  │  STATISTICAL SIGNIFICANCE (paired t-test)               │')
    print(f'  │    t-statistic:               {t_stat:>10.2f}               │')
    print(f'  │    p-value:                   {p_val:>14.2e}           │')
    if p_val < 0.001:
        print(f'  │    → Highly significant  (p < 0.001)  ✓                │')
    elif p_val < 0.01:
        print(f'  │    → Very significant    (p < 0.01)   ✓                │')
    elif p_val < 0.05:
        print(f'  │    → Significant         (p < 0.05)   ✓                │')
    else:
        print(f'  │    → NOT significant at α = 0.05      ✗                │')
    print(f'  └──────────────────────────────────────────────────────────┘')
    return m


def print_paper_table(all_results):
    """Markdown table suitable for direct inclusion in a paper."""
    heading('PAPER-READY SUMMARY TABLE (Markdown)')
    print()
    print('  | Category | N | Corr CEP (m) | Uncorr CEP (m) | '
          'Mean Δ (%) | Win Rate (%) |')
    print('  |---|---|---|---|---|---|')
    for cid, cname in CAT_NAMES.items():
        rows = [r for r in all_results if r[2]['cat'] == cid]
        m = compute_metrics(rows)
        if m['n'] == 0:
            continue
        short = cname.split('—')[-1].strip() if '—' in cname else cname
        print(f'  | {short} | {m["n"]} | {m["c_med"]:.1f} '
              f'| {m["u_med"]:.1f} | {m["i_mean"]:.1f} | {m["wr"]:.1f} |')
    m = compute_metrics(all_results)
    if m['n']:
        print(f'  | **Overall** | **{m["n"]}** | **{m["c_med"]:.1f}** '
              f'| **{m["u_med"]:.1f}** | **{m["i_mean"]:.1f}** '
              f'| **{m["wr"]:.1f}** |')

    print()
    print('  Per-missile summary:')
    print()
    print('  | Missile | N | Corr Mean (m) | Uncorr Mean (m) | '
          'Mean Δ (%) | Xtrack Δ (%) |')
    print('  |---|---|---|---|---|---|')
    for mt in MISSILES:
        rows = [r for r in all_results if r[2]['mt'] == mt]
        m = compute_metrics(rows)
        if m['n'] == 0:
            continue
        ctx = (1.0 - m['ct_c'] / max(m['ct_u'], 0.01)) * 100
        print(f'  | {MISSILES[mt]["name"]} | {m["n"]} | {m["c_mean"]:.1f} '
              f'| {m["u_mean"]:.1f} | {m["i_mean"]:.1f} | {ctx:.1f} |')


# =====================================================================
#  JSON export
# =====================================================================
def export_json(path, all_results, overall_metrics, elapsed, mode):
    records = []
    for cr, ur, info in all_results:
        c_ok = cr['status'] != 'timeout'
        u_ok = ur['status'] != 'timeout'
        imp = None
        if c_ok and u_ok and ur['miss_distance_m'] > 0.01:
            imp = (1.0 - cr['miss_distance_m'] / ur['miss_distance_m']) * 100.0
        records.append(dict(
            category=info['cat'],
            group=info['grp'],
            label=info['lbl'],
            missile=info['mt'],
            altitude=info['alt'],
            seed=info['seed'],
            range_km=math.sqrt((info['t'][0] - info['s'][0]) ** 2 +
                               (info['t'][1] - info['s'][1]) ** 2),
            corr_miss=cr['miss_distance_m'] if c_ok else None,
            uncorr_miss=ur['miss_distance_m'] if u_ok else None,
            improvement_pct=imp,
            corr_time=cr.get('time'),
            uncorr_time=ur.get('time'),
            avg_turbulence=cr.get('avg_turbulence'),
            corr_max_xtrack=cr.get('max_cross_track_m'),
            uncorr_max_xtrack=ur.get('max_cross_track_m'),
        ))
    blob = dict(
        generated=datetime.now().isoformat(),
        mode=mode,
        total_pairs=len(all_results),
        elapsed_s=elapsed,
        overall=overall_metrics,
        results=records,
    )
    with open(path, 'w') as f:
        json.dump(blob, f, indent=2, default=str)
    print(f'\n  Detailed results → {path}')


# =====================================================================
#  Main
# =====================================================================
def main():
    quick     = '--quick' in sys.argv
    json_path = None
    if '--json' in sys.argv:
        idx = sys.argv.index('--json')
        if idx + 1 < len(sys.argv):
            json_path = sys.argv[idx + 1]

    mode  = 'QUICK' if quick else 'FULL'
    tests = build_suite(quick)

    n_cats    = len(set(t['cat'] for t in tests))
    n_configs = len(set(t['grp'] for t in tests))
    n_missiles = len(set(t['mt'] for t in tests))

    print(SEP)
    print(f'  PSTNet — COMPREHENSIVE TEST SUITE ({mode})')
    print(f'  {len(tests)} paired simulations  •  '
          f'{n_cats} categories  •  '
          f'{n_configs} unique configs  •  '
          f'{n_missiles} missile types')
    print(SEP)

    # ---- Initialise shared services ----------------------------------
    print('\n  Initialising NASA weather service …')
    ws = WeatherService(lat=35.0, lon=-120.0)
    tf = TurbulenceField(ws)
    tf.update()
    mi = tf.get_model_info()
    print(f'  Model: {mi["name"]}')
    print(f'  Params: {mi.get("total_params", "?")}   '
          f'Trained: {mi["trained"]}   '
          f'Loss: {mi["final_loss"]:.6f}' if mi['final_loss'] else '')

    # ---- Run all pairs -----------------------------------------------
    print(f'\n  Running {len(tests)} paired trajectory simulations …\n')
    all_results = []
    t0 = time.time()

    for i, tc in enumerate(tests):
        pbar(i, len(tests), tag=tc['cat'])
        with mute():
            cr, ur = run_pair(tc['mt'], tf,
                              tc['s'], tc['t'], tc['alt'], tc['seed'])
        all_results.append((cr, ur, tc))

    pbar(len(tests), len(tests), tag='DONE')
    elapsed = time.time() - t0
    timeouts = sum(1 for r in all_results
                   if r[0]['status'] == 'timeout' or r[1]['status'] == 'timeout')
    print(f'\n\n  Completed in {elapsed:.1f}s  '
          f'({len(tests)} pairs, {timeouts} timeouts)\n')

    # ---- Reports -----------------------------------------------------
    for cid in CAT_NAMES:
        print_category(cid, all_results)

    print_by_missile(all_results)
    print_by_altitude(all_results)
    print_by_range(all_results)
    print_by_turbulence(all_results)
    print_histogram(all_results)
    print_extremes(all_results, n=5)
    print_consistency(all_results)
    m = print_overall(all_results, elapsed)
    print_paper_table(all_results)

    # ---- JSON --------------------------------------------------------
    if json_path and m and m.get('n', 0) > 0:
        export_json(json_path, all_results, m, elapsed, mode)

    print(f'\n{SEP}')
    print(f'  Test suite complete.')
    print(SEP)


if __name__ == '__main__':
    main()