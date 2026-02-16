#!/usr/bin/env python3
"""tests.py — Rigorous paired-comparison test suite for PSTNet

Improvements over baseline:
  1. Multi-location diverse training  (12 atmospheric profiles, not 1)
  2. Guidance-validity filtering — separates turbulence-correction
     efficacy from fundamental guidance failures
  3. Wilcoxon signed-rank test + paired t-test + bootstrap CI
  4. Cohen's d effect size per group
  5. Per-missile statistical significance testing
  6. Regime-gate correlation analysis
  7. Two-tier reporting: unfiltered (everything) + filtered (guidance-valid)

Usage
-----
  python tests.py                       # full suite  (~340 pairs)
  python tests.py --quick               # smoke test  (~80 pairs)
  python tests.py --json results.json   # export detailed JSON
"""

import sys, os, time, math, json, io, contextlib
import numpy as np
from collections import OrderedDict
from datetime import datetime

from config import (MISSILES, SCENARIOS, SPEED_OF_SOUND_SEA,
                    AIR_DENSITY_SEA, ALTITUDE_LAYERS)
from weather_api import WeatherService
from turbulence_model import TurbulenceField, TurbulencePredictor
from trajectory import MissileTrajectory

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False

# =====================================================================
#  Constants & Thresholds
# =====================================================================
SEP  = '═' * 80
THIN = '─' * 80

# Above this uncorrected miss distance the guidance system has
# fundamentally failed (target unreachable, wrong geometry, etc.)
# and turbulence correction is irrelevant.
GUIDANCE_VALID_M = 1000.0

BOOTSTRAP_N    = 5000
BOOTSTRAP_SEED = 42


# =====================================================================
#  Helpers
# =====================================================================
@contextlib.contextmanager
def mute():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def pbar(cur, tot, width=40, tag=''):
    f = cur / max(tot, 1)
    filled = int(width * f)
    b = '█' * filled + '░' * (width - filled)
    sys.stdout.write(f'\r  {tag:>12} [{b}] {f*100:5.1f}%  ({cur}/{tot})')
    sys.stdout.flush()


def heading(title, ch='═'):
    print(f'\n{ch * 80}\n  {title}\n{ch * 80}')


def sig_marker(p):
    if p is None:  return '  —'
    if p < 0.001:  return '***'
    if p < 0.01:   return ' **'
    if p < 0.05:   return '  *'
    return ' ns'


# =====================================================================
#  Statistical Utilities
# =====================================================================
def paired_ttest(x, y):
    """Paired t-test (two-tailed).  Returns (t, p)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    if n < 2:
        return 0.0, 1.0
    se = d.std(ddof=1) / math.sqrt(n) + 1e-15
    t = float(d.mean() / se)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test.  Returns (W, p) or (None, None)."""
    if not HAS_SCIPY_STATS:
        return None, None
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) < 10:
        return None, None
    try:
        w, p = scipy_wilcoxon(d)
        return float(w), float(p)
    except Exception:
        return None, None


def cohens_d(x, y):
    """Cohen's d for paired samples."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if len(d) < 2 or d.std() < 1e-15:
        return 0.0
    return float(d.mean() / d.std(ddof=1))


def bootstrap_ci(data, n_boot=BOOTSTRAP_N, ci=0.95, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap CI for the mean."""
    rng = np.random.RandomState(seed)
    data = np.asarray(data, float)
    n = len(data)
    if n < 2:
        v = float(data[0]) if n == 1 else 0.0
        return v, v
    boots = np.array([rng.choice(data, n, replace=True).mean()
                      for _ in range(n_boot)])
    a = (1 - ci) / 2
    return float(np.percentile(boots, a * 100)), \
           float(np.percentile(boots, (1 - a) * 100))


# =====================================================================
#  Enhanced Training — Multi-Location Diverse Profiles
# =====================================================================
def create_diverse_profiles(weather_service):
    """Generate 12 atmospheric profiles representing varied conditions.

    One real profile from NASA POWER plus 11 synthetic variants with
    different wind regimes, temperature offsets, and resulting
    stability / density changes.  This addresses the single-profile
    training weakness (11 altitude points from one location).
    """
    base_profile = weather_service.get_vertical_profile(ALTITUDE_LAYERS)
    profiles = [base_profile]

    rng = np.random.RandomState(42)
    specs = [
        dict(ws=0.5,  dt=-10),
        dict(ws=1.5,  dt=5),
        dict(ws=2.5,  dt=-5),
        dict(ws=0.8,  dt=15),
        dict(ws=3.0,  dt=-15),
        dict(ws=1.2,  dt=0),
        dict(ws=2.0,  dt=10),
        dict(ws=0.3,  dt=-8),
        dict(ws=1.8,  dt=8),
        dict(ws=2.8,  dt=-12),
        dict(ws=1.0,  dt=20),
    ]
    for sp in specs:
        variant = []
        for L in base_profile:
            lay = dict(L)
            lay['wind_speed']  = max(0.5,
                L['wind_speed'] * sp['ws'] + rng.uniform(-1, 1))
            lay['temperature'] = max(180.0, L['temperature'] + sp['dt'])
            lay['density']     = max(1e-5,
                (lay['pressure'] * 1000) / (287.05 * lay['temperature']))
            alt = lay['altitude']
            dT  = -6.5 if alt < 11 else (0.0 if alt < 20 else 1.0)
            ws  = max(lay['wind_speed'] / max(alt, 0.1), 0.1)
            lay['richardson'] = float(np.clip(
                (9.81 / lay['temperature']) * (dT + 9.8) / (ws**2 + 0.01),
                -5, 10))
            variant.append(lay)
        profiles.append(variant)
    return profiles


# =====================================================================
#  Core Runner
# =====================================================================
def run_pair(missile_type, turb_field, start, target, alt, seed,
             max_time=3000):
    """Run one corrected + one uncorrected trajectory with the same seed.

    Returns (cr_result, ur_result, extra_dict).
    """
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

    # Regime weights at cruise altitude (for analysis)
    regime_weights = {}
    if turb_field.predictor.trained:
        try:
            _, layer = turb_field.get_at(alt)
            regime_weights = turb_field.predictor.get_regime_weights(
                layer['wind_speed'], layer['temperature'], layer['density'],
                layer['richardson'], alt, layer['pressure'])
        except Exception:
            pass

    extra = dict(regime_weights=regime_weights)
    return (cr or timeout), (ur or timeout), extra


# =====================================================================
#  Test-Suite Builder  (same structure as original)
# =====================================================================
def build_suite(quick=False):
    nA = 3  if quick else 15
    nB = 2  if quick else 5
    nC = 2  if quick else 5
    nD = 2  if quick else 5
    nE = 2  if quick else 5
    nF = 4  if quick else 20

    T = []

    # ── A: Standard 4 scenarios ───────────────────────────────────
    for sid, cfg in SCENARIOS.items():
        for i in range(nA):
            T.append(dict(cat='A', grp=sid, lbl=cfg['name'],
                          mt=cfg['missile_type'],
                          s=[30.0, 100.0], t=[150.0, 100.0],
                          alt=cfg['launch_alt'], seed=i * 137 + 7))

    # ── B: Altitude sweep ─────────────────────────────────────────
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

    # ── C: Range sweep ────────────────────────────────────────────
    for mt in MISSILES:
        ca = MISSILES[mt].get('cruise_altitude', 10)
        for rng in [60, 120, 180]:
            for i in range(nC):
                T.append(dict(cat='C', grp=f'{mt}_R{rng}',
                              lbl=f'{mt} {rng}km',
                              mt=mt, s=[30.0, 100.0],
                              t=[30.0 + rng, 100.0],
                              alt=ca, seed=i * 373 + 19))

    # ── D: Lateral offset ─────────────────────────────────────────
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

    # ── F: Pure stochastic spread ─────────────────────────────────
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
#  Metrics Computation  (with optional miss-distance filter)
# =====================================================================
def compute_metrics(results, miss_cap=None):
    """Aggregate statistics.

    Parameters
    ----------
    results : list of (cr, ur, info, extra) tuples
    miss_cap : float or None
        If set, exclude runs where **uncorrected** miss > miss_cap.
    """
    valid = []
    for r in results:
        cr, ur = r[0], r[1]
        if cr['status'] == 'timeout' or ur['status'] == 'timeout':
            continue
        um = ur['miss_distance_m']
        if miss_cap is not None and um > miss_cap:
            continue
        valid.append((cr['miss_distance_m'], um, cr, ur, r[2]))

    if not valid:
        return dict(n=0)

    cm  = np.array([v[0] for v in valid])
    um  = np.array([v[1] for v in valid])
    imp = (1.0 - cm / np.maximum(um, 0.01)) * 100.0
    wins = int(np.sum(cm < um))
    ties = int(np.sum(np.abs(cm - um) < 0.01))

    ct_c = np.array([v[2].get('max_cross_track_m', 0) for v in valid])
    ct_u = np.array([v[3].get('max_cross_track_m', 0) for v in valid])
    turb = np.array([v[2].get('avg_turbulence', 0) for v in valid])

    t_s, p_t = paired_ttest(um, cm)
    w_s, p_w = wilcoxon_test(um, cm)
    d        = cohens_d(um, cm)
    b_lo, b_hi = bootstrap_ci(imp) if len(imp) >= 5 else (
        float(imp.mean()), float(imp.mean()))
    ci_se = imp.std(ddof=1) / math.sqrt(len(imp)) if len(imp) > 1 else 0
    ci_lo = float(imp.mean() - 1.96 * ci_se)
    ci_hi = float(imp.mean() + 1.96 * ci_se)

    return dict(
        n=len(valid),
        c_mean=float(cm.mean()), c_med=float(np.median(cm)),
        c_std=float(cm.std()),   c_max=float(cm.max()),
        c_p90=float(np.percentile(cm, 90)),
        u_mean=float(um.mean()), u_med=float(np.median(um)),
        u_std=float(um.std()),   u_max=float(um.max()),
        u_p90=float(np.percentile(um, 90)),
        i_mean=float(imp.mean()), i_med=float(np.median(imp)),
        i_std=float(imp.std()),   i_min=float(imp.min()),
        i_max=float(imp.max()),
        i_p25=float(np.percentile(imp, 25)),
        i_p75=float(np.percentile(imp, 75)),
        win=wins, lose=len(valid) - wins - ties, tie=ties,
        wr=float(wins / len(valid) * 100),
        ct_c=float(ct_c.mean()), ct_u=float(ct_u.mean()),
        turb_avg=float(turb.mean()),
        t_stat=t_s, p_t=p_t,
        w_stat=w_s, p_w=p_w,
        d=d,
        ci_lo=ci_lo, ci_hi=ci_hi,
        boot_lo=b_lo, boot_hi=b_hi,
    )


def guidance_counts(results):
    """Return (valid, failed, timeout) counts."""
    v = f = t = 0
    for r in results:
        if r[0]['status'] == 'timeout' or r[1]['status'] == 'timeout':
            t += 1
        elif r[1]['miss_distance_m'] > GUIDANCE_VALID_M:
            f += 1
        else:
            v += 1
    return v, f, t


# =====================================================================
#  Reporting — Building Blocks
# =====================================================================
CAT_NAMES = OrderedDict([
    ('A', 'Monte Carlo — Standard 4 Scenarios'),
    ('B', 'Altitude Sweep'),
    ('C', 'Range Sweep'),
    ('D', 'Lateral Offset Targets'),
    ('E', 'Stress / Edge Cases'),
    ('F', 'Pure Stochastic Spread'),
])


def _catrows(cat, results):
    return [r for r in results if r[2]['cat'] == cat]


# ── guidance validity ────────────────────────────────────────────
def report_guidance(results):
    heading('GUIDANCE VALIDITY ANALYSIS')
    print(f'\n  Threshold: uncorrected miss ≤ {GUIDANCE_VALID_M:.0f} m'
          f'  →  guidance valid')
    print(f'  Runs above this are excluded from the filtered analysis.\n')
    print(f'  {"Missile":<22} {"Tot":>4} {"OK":>4} {"Fail":>5}'
          f' {"T/O":>4}  {"%OK":>5}')
    print(f'  {THIN[:50]}')
    for mt in MISSILES:
        rows = [r for r in results if r[2]['mt'] == mt]
        v, f, t = guidance_counts(rows)
        tot = v + f + t
        print(f'  {MISSILES[mt]["name"]:<22} {tot:>4} {v:>4} {f:>5}'
              f' {t:>4}  {v / max(tot,1)*100:>5.1f}%')
    v, f, t = guidance_counts(results)
    tot = v + f + t
    print(f'  {THIN[:50]}')
    print(f'  {"TOTAL":<22} {tot:>4} {v:>4} {f:>5}'
          f' {t:>4}  {v / max(tot,1)*100:>5.1f}%')


# ── per-category table ───────────────────────────────────────────
def report_category(cat, results, filt):
    rows = _catrows(cat, results)
    if not rows:
        return
    cap = GUIDANCE_VALID_M if filt else None
    tag = '  [FILTERED]' if filt else ''
    heading(f'CATEGORY {cat} — {CAT_NAMES[cat]}{tag}')

    hdr = (f'  {"Config":<34} {"N":>3}  {"Corr":>8}  {"Uncorr":>9}'
           f'  {"Δ%":>6}  {"Win%":>5}  {"p(t)":>7} {"":>3}')
    print(hdr)
    print(f'  {THIN[:78]}')

    groups = list(dict.fromkeys(r[2]['grp'] for r in rows))
    for g in groups:
        gr = [r for r in rows if r[2]['grp'] == g]
        m = compute_metrics(gr, miss_cap=cap)
        if m['n'] == 0:
            continue
        lbl = gr[0][2]['lbl'][:34]
        pv = f'{m["p_t"]:.4f}' if m['n'] >= 3 else '     —'
        sg = sig_marker(m['p_t'] if m['n'] >= 3 else None)
        print(f'  {lbl:<34} {m["n"]:>3}  {m["c_med"]:>6.1f}m'
              f'  {m["u_med"]:>7.1f}m  {m["i_mean"]:>5.1f}%'
              f'  {m["wr"]:>4.0f}%  {pv:>7} {sg}')

    m = compute_metrics(rows, miss_cap=cap)
    if m['n']:
        pv = f'{m["p_t"]:.4f}' if m['n'] >= 3 else '     —'
        sg = sig_marker(m['p_t'] if m['n'] >= 3 else None)
        print(f'  {THIN[:78]}')
        print(f'  {"SUBTOTAL":<34} {m["n"]:>3}  {m["c_med"]:>6.1f}m'
              f'  {m["u_med"]:>7.1f}m  {m["i_mean"]:>5.1f}%'
              f'  {m["wr"]:>4.0f}%  {pv:>7} {sg}')


# ── per-missile significance ─────────────────────────────────────
def report_per_missile(results, filt):
    cap = GUIDANCE_VALID_M if filt else None
    tag = '  [FILTERED]' if filt else ''
    heading(f'PER-MISSILE STATISTICAL SIGNIFICANCE{tag}')
    print(f'  {"Type":<22} {"N":>4}  {"Corr":>7}  {"Uncorr":>8}'
          f'  {"Δ%":>6}  {"d":>6}  {"t":>7}  {"p(t)":>8}'
          f'  {"p(W)":>8} {"":>3}')
    print(f'  {THIN[:80]}')

    for mt in MISSILES:
        rows = [r for r in results if r[2]['mt'] == mt]
        m = compute_metrics(rows, miss_cap=cap)
        if m['n'] < 3:
            v, f, _ = guidance_counts(rows)
            print(f'  {MISSILES[mt]["name"]:<22}  N={m["n"]}  '
                  f'({f} guidance failures)')
            continue
        pw = f'{m["p_w"]:.5f}' if m['p_w'] is not None else '      —'
        sg = sig_marker(m['p_t'])
        print(f'  {MISSILES[mt]["name"]:<22} {m["n"]:>4}'
              f'  {m["c_mean"]:>6.1f}m {m["u_mean"]:>7.1f}m'
              f'  {m["i_mean"]:>5.1f}% {m["d"]:>6.3f}'
              f' {m["t_stat"]:>7.2f}  {m["p_t"]:>8.5f}'
              f'  {pw:>8} {sg}')

    print(f'\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns not sig.')
    print(f'  Cohen d: |d|<0.2 negligible, 0.2-0.5 small, '
          f'0.5-0.8 medium, >0.8 large')


# ── by altitude ──────────────────────────────────────────────────
def report_by_altitude(results, filt):
    cap = GUIDANCE_VALID_M if filt else None
    tag = '  [FILTERED]' if filt else ''
    heading(f'BREAKDOWN BY ALTITUDE BAND{tag}')
    bands = [
        ('Surface  <  2 km',  0,  2),
        ('Low      2 – 5 km', 2,  5),
        ('Mid      5 – 12',   5, 12),
        ('High    12 – 20',  12, 20),
        ('Strat    > 20 km', 20, 99),
    ]
    print(f'  {"Band":<22} {"N":>3}  {"Corr":>8}  {"Uncorr":>9}'
          f'  {"Δ%":>6}  {"Win%":>5}  {"d":>6} {"":>3}')
    print(f'  {THIN[:66]}')
    for name, lo, hi in bands:
        rows = [r for r in results if lo <= r[2]['alt'] < hi]
        m = compute_metrics(rows, miss_cap=cap)
        if m['n'] == 0:
            continue
        sg = sig_marker(m['p_t'] if m['n'] >= 3 else None)
        print(f'  {name:<22} {m["n"]:>3}  {m["c_med"]:>6.1f}m'
              f'  {m["u_med"]:>7.1f}m  {m["i_mean"]:>5.1f}%'
              f'  {m["wr"]:>4.0f}%  {m["d"]:>6.3f} {sg}')


# ── by range ─────────────────────────────────────────────────────
def report_by_range(results, filt):
    cap = GUIDANCE_VALID_M if filt else None
    tag = '  [FILTERED]' if filt else ''
    heading(f'BREAKDOWN BY ENGAGEMENT RANGE{tag}')
    bands = [
        ('Short   < 80 km',    0,  80),
        ('Medium  80 – 140 km',80, 140),
        ('Long    > 140 km',  140, 999),
    ]
    print(f'  {"Range":<24} {"N":>3}  {"Corr":>8}  {"Uncorr":>9}'
          f'  {"Δ%":>6}  {"Win%":>5}  {"d":>6} {"":>3}')
    print(f'  {THIN[:68]}')
    for name, lo, hi in bands:
        rows = []
        for r in results:
            dx = r[2]['t'][0] - r[2]['s'][0]
            dy = r[2]['t'][1] - r[2]['s'][1]
            rng = math.hypot(dx, dy)
            if lo <= rng < hi:
                rows.append(r)
        m = compute_metrics(rows, miss_cap=cap)
        if m['n'] == 0:
            continue
        sg = sig_marker(m['p_t'] if m['n'] >= 3 else None)
        print(f'  {name:<24} {m["n"]:>3}  {m["c_med"]:>6.1f}m'
              f'  {m["u_med"]:>7.1f}m  {m["i_mean"]:>5.1f}%'
              f'  {m["wr"]:>4.0f}%  {m["d"]:>6.3f} {sg}')


# ── by turbulence severity ───────────────────────────────────────
def report_by_turbulence(results, filt):
    cap = GUIDANCE_VALID_M if filt else None
    tag = '  [FILTERED]' if filt else ''
    heading(f'BREAKDOWN BY TURBULENCE SEVERITY{tag}')
    bands = [
        ('Low       < 15%',    0.00, 0.15),
        ('Moderate 15 – 30%',  0.15, 0.30),
        ('High     30 – 50%',  0.30, 0.50),
        ('Severe    > 50%',    0.50, 1.00),
    ]
    print(f'  {"Severity":<22} {"N":>3}  {"Corr":>8}  {"Uncorr":>9}'
          f'  {"Δ%":>6}  {"Win%":>5}  {"d":>6} {"":>3}')
    print(f'  {THIN[:66]}')
    for name, lo, hi in bands:
        rows = [r for r in results
                if lo <= r[0].get('avg_turbulence', 0) < hi
                and r[0]['status'] != 'timeout']
        m = compute_metrics(rows, miss_cap=cap)
        if m['n'] == 0:
            continue
        sg = sig_marker(m['p_t'] if m['n'] >= 3 else None)
        print(f'  {name:<22} {m["n"]:>3}  {m["c_mean"]:>6.1f}m'
              f'  {m["u_mean"]:>7.1f}m  {m["i_mean"]:>5.1f}%'
              f'  {m["wr"]:>4.0f}%  {m["d"]:>6.3f} {sg}')


# ── histogram ────────────────────────────────────────────────────
def report_histogram(results, filt):
    cap = GUIDANCE_VALID_M if filt else None
    pairs = []
    for r in results:
        if r[0]['status'] == 'timeout' or r[1]['status'] == 'timeout':
            continue
        if r[1]['miss_distance_m'] < 0.01:
            continue
        if cap and r[1]['miss_distance_m'] > cap:
            continue
        pairs.append((r[0]['miss_distance_m'], r[1]['miss_distance_m']))
    if not pairs:
        return
    imp = [(1.0 - c / u) * 100.0 for c, u in pairs]
    tag = '  [FILTERED]' if filt else ''
    heading(f'IMPROVEMENT DISTRIBUTION{tag}')
    edges = list(range(-30, 101, 10))
    cnts = [0] * (len(edges) - 1)
    for v in imp:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                cnts[i] += 1
                break
        else:
            if v >= edges[-1]:
                cnts[-1] += 1
    mx = max(cnts) if cnts else 1
    for i, c in enumerate(cnts):
        lo, hi = edges[i], edges[i + 1]
        bl = '█' * int(c / mx * 45)
        mk = ' ◄── zero' if lo <= 0 < hi else ''
        print(f'  {lo:>+4} to {hi:>+3}% │{bl} {c}{mk}')
    print(f'\n  Mean: {np.mean(imp):.1f}%   '
          f'Median: {np.median(imp):.1f}%   '
          f'IQR: [{np.percentile(imp,25):.1f}%, '
          f'{np.percentile(imp,75):.1f}%]')


# ── extremes ─────────────────────────────────────────────────────
def report_extremes(results, n=5, filt=False):
    cap = GUIDANCE_VALID_M if filt else None
    ok = []
    for r in results:
        if r[0]['status'] == 'timeout' or r[1]['status'] == 'timeout':
            continue
        um = r[1]['miss_distance_m']
        if um < 0.01:
            continue
        if cap and um > cap:
            continue
        ok.append((r[0]['miss_distance_m'], um, r[2]))
    if not ok:
        return
    ranked = sorted(ok, key=lambda x: (1.0 - x[0] / x[1]))
    tag = '  [FILTERED]' if filt else ''
    heading(f'EXTREME CASES{tag}')
    print(f'  {n} Worst (least / negative improvement):')
    for c, u, info in ranked[:n]:
        iv = (1.0 - c / u) * 100.0
        print(f'    {info["lbl"]:<36} corr={c:>7.1f}m'
              f'  uncorr={u:>7.1f}m  Δ={iv:>+6.1f}%')
    print(f'\n  {n} Best (most improvement):')
    for c, u, info in ranked[-n:]:
        iv = (1.0 - c / u) * 100.0
        print(f'    {info["lbl"]:<36} corr={c:>7.1f}m'
              f'  uncorr={u:>7.1f}m  Δ={iv:>+6.1f}%')


# ── consistency ──────────────────────────────────────────────────
def report_consistency(results, filt):
    cap = GUIDANCE_VALID_M if filt else None
    ok = []
    for r in results:
        if r[0]['status'] == 'timeout' or r[1]['status'] == 'timeout':
            continue
        um = r[1]['miss_distance_m']
        if cap and um > cap:
            continue
        ok.append((r[0]['miss_distance_m'], um))
    if not ok:
        return
    cm = np.array([o[0] for o in ok])
    um = np.array([o[1] for o in ok])
    imp = (1.0 - cm / np.maximum(um, 0.01)) * 100.0
    tag = '  [FILTERED]' if filt else ''
    heading(f'CONSISTENCY ANALYSIS{tag}')
    for th in [0, -5, -10, -20]:
        worse = int(np.sum(imp < th))
        pct = worse / len(imp) * 100
        lbl = (f'correction > {abs(th)}% worse' if th < 0
               else 'correction worse at all')
        print(f'  Cases where {lbl}: '
              f'{worse:>4} / {len(imp)}  ({pct:.1f}%)')
    print(f'\n  Improvement percentiles:')
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f'    P{p:<2}: {np.percentile(imp, p):>+7.1f}%')


# ── regime gate correlation ──────────────────────────────────────
def report_regime_analysis(results):
    heading('REGIME-GATE CORRELATION ANALYSIS')
    data = []
    for r in results:
        cr, ur, info = r[0], r[1], r[2]
        extra = r[3] if len(r) > 3 else {}
        if cr['status'] == 'timeout' or ur['status'] == 'timeout':
            continue
        um = ur['miss_distance_m']
        if um < 0.01 or um > GUIDANCE_VALID_M:
            continue
        rw = extra.get('regime_weights', {})
        if not rw:
            continue
        imp = (1.0 - cr['miss_distance_m'] / um) * 100.0
        dom = max(rw, key=rw.get)
        data.append(dict(imp=imp, dom=dom, **rw))

    if len(data) < 5:
        print('  Insufficient guidance-valid data for regime analysis.')
        return

    regimes = ['convective', 'neutral', 'stable', 'stratospheric']
    print(f'\n  By dominant regime (guidance-valid runs only):')
    print(f'  {"Dominant":<16} {"N":>4}  {"Mean Δ%":>8}  {"Med Δ%":>8}'
          f'  {"Win%":>6}  {"Avg gate":>8}')
    print(f'  {THIN[:58]}')
    for reg in regimes:
        grp = [d for d in data if d['dom'] == reg]
        if not grp:
            continue
        imps = np.array([d['imp'] for d in grp])
        gv   = np.array([d.get(reg, 0) for d in grp])
        wr   = np.sum(imps > 0) / len(imps) * 100
        print(f'  {reg:<16} {len(grp):>4}  {imps.mean():>+7.1f}%'
              f'  {np.median(imps):>+7.1f}%  {wr:>5.1f}%'
              f'  {gv.mean():>7.3f}')

    # Pearson r between each gate weight and improvement
    print(f'\n  Pearson r  (gate weight vs improvement):')
    imps = np.array([d['imp'] for d in data])
    for reg in regimes:
        vals = np.array([d.get(reg, 0) for d in data])
        if vals.std() < 1e-10 or imps.std() < 1e-10:
            continue
        r = np.corrcoef(vals, imps)[0, 1]
        bar = '█' * int(abs(r) * 40)
        sign = '+' if r >= 0 else '-'
        print(f'    {reg:<16}  r = {r:>+.3f}  {sign}{bar}')


# ── overall results box ──────────────────────────────────────────
def report_overall(results, elapsed, filt):
    cap = GUIDANCE_VALID_M if filt else None
    m = compute_metrics(results, miss_cap=cap)
    if m['n'] == 0:
        print('\n  No valid results.')
        return m

    total_nto = sum(1 for r in results
                    if r[0]['status'] != 'timeout'
                    and r[1]['status'] != 'timeout')
    n_excl = total_nto - m['n'] if filt else 0

    tag = f'  [uncorr miss ≤ {GUIDANCE_VALID_M:.0f} m]' if filt else ''
    heading(f'OVERALL RESULTS{tag}')

    pw_s = f'{m["p_w"]:.2e}' if m['p_w'] is not None else 'N/A'
    ct_red = (1.0 - m['ct_c'] / max(m['ct_u'], 0.01)) * 100

    lines = [
        f'',
        f'  Total pairs:       {len(results):>6}',
        f'  Timeouts:          {len(results) - total_nto:>6}',
    ]
    if filt:
        lines.append(
        f'  Guidance failures: {n_excl:>6}    (excluded)')
    lines += [
        f'  Analysed:          {m["n"]:>6}',
        f'  Wall-clock:        {elapsed:>6.1f} s  '
        f'({elapsed / max(len(results), 1):.2f} s/pair)',
        f'',
        f'  ┌──────────────────────────────────────────────────────────┐',
        f'  │  CORRECTED  (PSTNet ML active)                           │',
        f'  │    Mean miss:       {m["c_mean"]:>9.1f} m                      │',
        f'  │    Median (CEP):    {m["c_med"]:>9.1f} m                      │',
        f'  │    P90:             {m["c_p90"]:>9.1f} m                      │',
        f'  │    Std:             {m["c_std"]:>9.1f} m                      │',
        f'  │    Worst:           {m["c_max"]:>9.1f} m                      │',
        f'  ├──────────────────────────────────────────────────────────┤',
        f'  │  UNCORRECTED  (physics only)                             │',
        f'  │    Mean miss:       {m["u_mean"]:>9.1f} m                      │',
        f'  │    Median (CEP):    {m["u_med"]:>9.1f} m                      │',
        f'  │    P90:             {m["u_p90"]:>9.1f} m                      │',
        f'  │    Std:             {m["u_std"]:>9.1f} m                      │',
        f'  │    Worst:           {m["u_max"]:>9.1f} m                      │',
        f'  ├──────────────────────────────────────────────────────────┤',
        f'  │  IMPROVEMENT                                             │',
        f'  │    Mean:            {m["i_mean"]:>+8.1f} %                      │',
        f'  │    Median:          {m["i_med"]:>+8.1f} %                      │',
        f'  │    IQR:    [{m["i_p25"]:>+6.1f}%, {m["i_p75"]:>+6.1f}%]'
        f'                      │',
        f'  │    Best:            {m["i_max"]:>+8.1f} %                      │',
        f'  │    Worst:           {m["i_min"]:>+8.1f} %                      │',
        f'  │    Win/Lose/Tie:  {m["win"]:>4} / {m["lose"]:>4} / {m["tie"]:>4}'
        f'                   │',
        f'  │    Win rate:        {m["wr"]:>7.1f} %                      │',
        f'  ├──────────────────────────────────────────────────────────┤',
        f'  │  EFFECT SIZE                                             │',
        f'  │    Cohen d:         {m["d"]:>+8.3f}                        │',
        f'  │    Parametric 95% CI: [{m["ci_lo"]:>+6.1f}%, {m["ci_hi"]:>+6.1f}%]'
        f'                │',
        f'  │    Bootstrap  95% CI: [{m["boot_lo"]:>+6.1f}%, {m["boot_hi"]:>+6.1f}%]'
        f'                │',
        f'  ├──────────────────────────────────────────────────────────┤',
        f'  │  CROSS-TRACK                                             │',
        f'  │    Corr mean max:   {m["ct_c"]:>9.1f} m                      │',
        f'  │    Uncorr mean max: {m["ct_u"]:>9.1f} m                      │',
        f'  │    Reduction:       {ct_red:>+8.1f} %                      │',
        f'  ├──────────────────────────────────────────────────────────┤',
        f'  │  STATISTICAL TESTS                                       │',
        f'  │    Paired t-test                                         │',
        f'  │      t = {m["t_stat"]:>8.3f}   '
        f'p = {m["p_t"]:.2e}  {sig_marker(m["p_t"])}              │',
        f'  │    Wilcoxon signed-rank                                  │',
    ]
    if m['p_w'] is not None:
        lines.append(
        f'  │      W = {m["w_stat"]:>8.1f}   p = {m["p_w"]:.2e}'
        f'  {sig_marker(m["p_w"])}              │')
    else:
        lines.append(
        f'  │      (N < 10 or scipy unavailable)                       │')
    lines.append(
        f'  └──────────────────────────────────────────────────────────┘')

    for ln in lines:
        print(ln)
    return m


# ── filtered vs unfiltered side-by-side ──────────────────────────
def report_comparison(m_all, m_filt):
    heading('FILTERED vs UNFILTERED COMPARISON')
    if not m_all or not m_filt or m_all.get('n', 0) == 0:
        print('  Insufficient data.')
        return
    if m_filt.get('n', 0) == 0:
        print('  No guidance-valid runs found. Cannot compare.')
        return

    def yn(p): return 'YES ✓' if p < 0.05 else 'NO  ✗'

    print(f'''
  ┌────────────────────────┬──────────────┬──────────────┐
  │  Metric                │  Unfiltered  │   Filtered   │
  ├────────────────────────┼──────────────┼──────────────┤
  │  N                     │  {m_all["n"]:>10}  │  {m_filt["n"]:>10}  │
  │  Corr mean miss (m)    │  {m_all["c_mean"]:>10.1f}  │  {m_filt["c_mean"]:>10.1f}  │
  │  Uncorr mean miss (m)  │  {m_all["u_mean"]:>10.1f}  │  {m_filt["u_mean"]:>10.1f}  │
  │  Mean improvement (%)  │  {m_all["i_mean"]:>+10.1f}  │  {m_filt["i_mean"]:>+10.1f}  │
  │  Median improv. (%)    │  {m_all["i_med"]:>+10.1f}  │  {m_filt["i_med"]:>+10.1f}  │
  │  Win rate (%)          │  {m_all["wr"]:>10.1f}  │  {m_filt["wr"]:>10.1f}  │
  │  Cohen d               │  {m_all["d"]:>+10.3f}  │  {m_filt["d"]:>+10.3f}  │
  │  p-value (t-test)      │  {m_all["p_t"]:>10.2e}  │  {m_filt["p_t"]:>10.2e}  │
  │  Significant?          │  {yn(m_all["p_t"]):>10}  │  {yn(m_filt["p_t"]):>10}  │
  └────────────────────────┴──────────────┴──────────────┘''')

    sa = m_all['p_t'] < 0.05
    sf = m_filt['p_t'] < 0.05
    print()
    if sf and not sa:
        print('  → Filtering reveals a significant effect masked by'
              ' guidance failures.')
        print('    Correction helps when the missile can reach the target.')
    elif sf and sa:
        print('  → Both analyses show significance.'
              ' The effect is robust.')
    elif not sf and not sa:
        print('  → Neither analysis shows significance.'
              ' Effect is too small or')
        print('    too variable to distinguish from noise.')
    elif not sf and sa:
        print('  → Unfiltered is significant but filtered is not.'
              ' The apparent')
        print('    effect may be driven by large-error outlier compression.')


# ── paper-ready table ────────────────────────────────────────────
def report_paper_table(results):
    heading('PAPER-READY TABLE  (Markdown, filtered)')
    print()
    print('  | Category | N | Corr CEP | Uncorr CEP | '
          'Mean Δ% | Win% | p(t) | Sig |')
    print('  |---|---|---|---|---|---|---|---|')
    for cid, cname in CAT_NAMES.items():
        rows = _catrows(cid, results)
        m = compute_metrics(rows, miss_cap=GUIDANCE_VALID_M)
        if m['n'] == 0:
            continue
        short = cname.split('—')[-1].strip() if '—' in cname else cname
        sg = sig_marker(m['p_t'])
        print(f'  | {short} | {m["n"]} | {m["c_med"]:.1f} '
              f'| {m["u_med"]:.1f} | {m["i_mean"]:.1f} '
              f'| {m["wr"]:.1f} | {m["p_t"]:.4f} | {sg.strip()} |')
    m = compute_metrics(results, miss_cap=GUIDANCE_VALID_M)
    if m['n']:
        sg = sig_marker(m['p_t'])
        print(f'  | **Overall** | **{m["n"]}** | **{m["c_med"]:.1f}** '
              f'| **{m["u_med"]:.1f}** | **{m["i_mean"]:.1f}** '
              f'| **{m["wr"]:.1f}** | **{m["p_t"]:.4f}** | {sg.strip()} |')

    print()
    print('  Per-missile (filtered):')
    print()
    print('  | Missile | N | Corr | Uncorr | Δ% | d | p | Sig |')
    print('  |---|---|---|---|---|---|---|---|')
    for mt in MISSILES:
        rows = [r for r in results if r[2]['mt'] == mt]
        m = compute_metrics(rows, miss_cap=GUIDANCE_VALID_M)
        if m['n'] < 3:
            continue
        sg = sig_marker(m['p_t'])
        print(f'  | {MISSILES[mt]["name"]} | {m["n"]} '
              f'| {m["c_mean"]:.1f} | {m["u_mean"]:.1f} '
              f'| {m["i_mean"]:.1f} | {m["d"]:.3f} '
              f'| {m["p_t"]:.4f} | {sg.strip()} |')


# ── honest assessment ────────────────────────────────────────────
def report_honest_assessment(results):
    heading('HONEST ASSESSMENT')
    m_all  = compute_metrics(results)
    m_filt = compute_metrics(results, miss_cap=GUIDANCE_VALID_M)
    v, f, t = guidance_counts(results)

    print(f'\n  KEY FINDINGS:\n')

    # Overall significance
    if m_filt.get('n', 0) >= 3 and m_filt['p_t'] < 0.05:
        d_word = ('negligible' if abs(m_filt['d']) < 0.2 else
                  'small' if abs(m_filt['d']) < 0.5 else
                  'medium' if abs(m_filt['d']) < 0.8 else 'large')
        print(f'  ✓ PSTNet correction is statistically significant '
              f'among guidance-valid')
        print(f'    runs  (p = {m_filt["p_t"]:.4f}, '
              f'd = {m_filt["d"]:.3f} [{d_word}]).')
    else:
        print(f'  ✗ PSTNet correction is NOT statistically significant '
              f'even after')
        p_str = (f'{m_filt["p_t"]:.4f}' if m_filt.get('n', 0) >= 3
                 else 'N/A (N<3)')
        print(f'    filtering out guidance failures  (p = {p_str}).')

    # Guidance failures
    if f > 0:
        print(f'\n  ⚠ {f} of {v+f+t} runs ({f/(v+f+t)*100:.0f}%) are '
              f'guidance failures (miss > {GUIDANCE_VALID_M:.0f} m).')
        print(f'    These inflate variance and dilute the correction signal.')

    # Per-missile
    print(f'\n  PER-MISSILE:')
    for mt in MISSILES:
        rows = [r for r in results if r[2]['mt'] == mt]
        m = compute_metrics(rows, miss_cap=GUIDANCE_VALID_M)
        vm, fm, _ = guidance_counts(rows)
        if m.get('n', 0) < 3:
            print(f'    {MISSILES[mt]["name"]:<22} '
                  f'only {m.get("n",0)} guidance-valid '
                  f'({fm} failures) — cannot evaluate')
            continue
        word = 'SIGNIFICANT' if m['p_t'] < 0.05 else 'not significant'
        d_w = ('negligible' if abs(m['d']) < 0.2 else
               'small' if abs(m['d']) < 0.5 else
               'medium' if abs(m['d']) < 0.8 else 'large')
        print(f'    {MISSILES[mt]["name"]:<22} '
              f'Δ={m["i_mean"]:>+5.1f}%  d={m["d"]:>+.3f} ({d_w})'
              f'  p={m["p_t"]:.4f} ({word})'
              f'  win {m["wr"]:.0f}%')

    # Recommendations
    if m_filt.get('n', 0) >= 3 and m_filt['p_t'] < 0.05:
        print(f'\n  RECOMMENDATION:')
        print(f'    The correction provides a modest but real improvement')
        print(f'    when guidance is functional.  Practical significance')
        print(f'    depends on whether {m_filt["i_mean"]:.1f}% mean '
              f'accuracy gain justifies')
        print(f'    the additional complexity.')
    else:
        print(f'\n  RECOMMENDATION:')
        print(f'    The correction does not demonstrate significance.')
        print(f'    Consider:')
        print(f'      (a) More diverse / larger training data')
        print(f'      (b) Higher correction magnitude caps')
        print(f'      (c) Regime gate with physically-derived inputs')
        print(f'      (d) Separate tuning per missile / altitude band')


# =====================================================================
#  JSON Export
# =====================================================================
def export_json(path, results, m_all, m_filt, elapsed, mode):
    records = []
    for r in results:
        cr, ur, info = r[0], r[1], r[2]
        extra = r[3] if len(r) > 3 else {}
        c_ok = cr['status'] != 'timeout'
        u_ok = ur['status'] != 'timeout'
        imp = None
        if c_ok and u_ok and ur['miss_distance_m'] > 0.01:
            imp = (1.0 - cr['miss_distance_m']
                   / ur['miss_distance_m']) * 100.0
        gv = u_ok and ur['miss_distance_m'] <= GUIDANCE_VALID_M
        rec = dict(
            category=info['cat'], group=info['grp'], label=info['lbl'],
            missile=info['mt'], altitude=info['alt'], seed=info['seed'],
            range_km=math.hypot(info['t'][0] - info['s'][0],
                                info['t'][1] - info['s'][1]),
            corr_miss=cr['miss_distance_m'] if c_ok else None,
            uncorr_miss=ur['miss_distance_m'] if u_ok else None,
            improvement_pct=imp, guidance_valid=gv,
            corr_time=cr.get('time'), uncorr_time=ur.get('time'),
            avg_turbulence=cr.get('avg_turbulence'),
            corr_max_xtrack=cr.get('max_cross_track_m'),
            uncorr_max_xtrack=ur.get('max_cross_track_m'),
            regime_weights=extra.get('regime_weights', {}),
        )
        records.append(rec)
    blob = dict(
        generated=datetime.now().isoformat(), mode=mode,
        total_pairs=len(results), elapsed_s=elapsed,
        guidance_threshold_m=GUIDANCE_VALID_M,
        overall_unfiltered=m_all, overall_filtered=m_filt,
        results=records,
    )
    with open(path, 'w') as fh:
        json.dump(blob, fh, indent=2, default=str)
    print(f'\n  Detailed results → {path}')


# =====================================================================
#  Main
# =====================================================================
def main():
    quick = '--quick' in sys.argv
    json_path = None
    if '--json' in sys.argv:
        idx = sys.argv.index('--json')
        if idx + 1 < len(sys.argv):
            json_path = sys.argv[idx + 1]

    mode  = 'QUICK' if quick else 'FULL'
    tests = build_suite(quick)

    n_cats     = len(set(t['cat'] for t in tests))
    n_configs  = len(set(t['grp'] for t in tests))
    n_missiles = len(set(t['mt']  for t in tests))

    print(SEP)
    print(f'  PSTNet — RIGOROUS TEST SUITE ({mode})')
    print(f'  {len(tests)} paired simulations  •  '
          f'{n_cats} categories  •  '
          f'{n_configs} configs  •  '
          f'{n_missiles} missile types')
    print(f'  Guidance-validity threshold: '
          f'{GUIDANCE_VALID_M:.0f} m')
    if HAS_SCIPY_STATS:
        print(f'  scipy.stats available — Wilcoxon test enabled')
    else:
        print(f'  scipy.stats not found — Wilcoxon test disabled')
    print(SEP)

    # ── Weather + Turbulence Field ────────────────────────────────
    print('\n  [1/3] NASA weather …')
    ws = WeatherService(lat=35.0, lon=-120.0)
    tf = TurbulenceField(ws)

    print('  [2/3] Base turbulence field …')
    tf.update()

    print('  [3/3] Retraining PSTNet with diverse profiles (12 locations) …')
    diverse = create_diverse_profiles(ws)
    tf.predictor = TurbulencePredictor()          # fresh weights
    tf.predictor.fit(diverse, epochs=300, lr=0.004)

    mi = tf.get_model_info()
    print(f'  Model: {mi["name"]}')
    n_raw = len(diverse) * len(ALTITUDE_LAYERS)
    print(f'  Params: {mi.get("total_params","?")}   '
          f'Trained: {mi["trained"]}   '
          + (f'Loss: {mi["final_loss"]:.6f}' if mi['final_loss'] else ''))
    print(f'  Training: {len(diverse)} profiles, '
          f'{n_raw} raw samples')

    # ── Turbulence profile for reference ──────────────────────────
    print('\n  Turbulence profile:')
    for a in sorted(tf.turb_by_alt):
        print(f'    {a:5.1f} km : {tf.turb_by_alt[a]*100:5.1f}%')

    print(f'\n  Regime gate weights:')
    for L in tf.profile:
        rw = tf.predictor.get_regime_weights(
            L['wind_speed'], L['temperature'], L['density'],
            L['richardson'], L['altitude'], L['pressure'])
        dom = max(rw, key=rw.get)
        ws_str = '  '.join(f'{k[:4]}={v:.2f}' for k, v in rw.items())
        print(f'    {L["altitude"]:5.1f} km  {ws_str}  → {dom}')

    # ── Run paired simulations ────────────────────────────────────
    print(f'\n  Running {len(tests)} paired trajectory simulations …\n')
    all_results = []
    t0 = time.time()

    for i, tc in enumerate(tests):
        pbar(i, len(tests), tag=tc['cat'])
        with mute():
            cr, ur, extra = run_pair(tc['mt'], tf,
                                     tc['s'], tc['t'],
                                     tc['alt'], tc['seed'])
        all_results.append((cr, ur, tc, extra))

    pbar(len(tests), len(tests), tag='DONE')
    elapsed = time.time() - t0
    timeouts = sum(1 for r in all_results
                   if r[0]['status'] == 'timeout'
                   or r[1]['status'] == 'timeout')
    print(f'\n\n  Completed in {elapsed:.1f}s  '
          f'({len(tests)} pairs, {timeouts} timeouts)\n')

    # ==============================================================
    #  PART 1  — Guidance Validity
    # ==============================================================
    report_guidance(all_results)

    # ==============================================================
    #  PART 2  — Unfiltered (all runs)
    # ==============================================================
    heading('SECTION 1: UNFILTERED RESULTS  (ALL RUNS)', ch='▓')
    print('  Includes guidance failures.  For reference only.\n')

    for cid in CAT_NAMES:
        report_category(cid, all_results, filt=False)

    m_all = report_overall(all_results, elapsed, filt=False)

    # ==============================================================
    #  PART 3  — Filtered (guidance-valid only)
    # ==============================================================
    heading(f'SECTION 2: FILTERED RESULTS  '
            f'(uncorr miss ≤ {GUIDANCE_VALID_M:.0f} m)', ch='▓')
    print('  Guidance failures removed.  '
          'This is the primary analysis.\n')

    for cid in CAT_NAMES:
        report_category(cid, all_results, filt=True)

    report_per_missile(all_results, filt=True)
    report_by_altitude(all_results, filt=True)
    report_by_range(all_results, filt=True)
    report_by_turbulence(all_results, filt=True)
    report_histogram(all_results, filt=True)
    report_extremes(all_results, n=5, filt=True)
    report_consistency(all_results, filt=True)

    m_filt = report_overall(all_results, elapsed, filt=True)

    # ==============================================================
    #  PART 4  — Comparative / Analytical
    # ==============================================================
    if m_all and m_filt:
        report_comparison(m_all, m_filt)
    report_regime_analysis(all_results)
    report_paper_table(all_results)
    report_honest_assessment(all_results)

    # ── JSON export ───────────────────────────────────────────────
    if json_path and m_all and m_all.get('n', 0) > 0:
        export_json(json_path, all_results,
                    m_all, m_filt or {}, elapsed, mode)

    print(f'\n{SEP}')
    print(f'  Test suite complete.')
    print(SEP)


if __name__ == '__main__':
    main()