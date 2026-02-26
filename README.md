<p align="center">
  <img src="https://img.shields.io/badge/PSTNet-Physically--Structured%20Turbulence%20Network-0d1117?style=for-the-badge&labelColor=161b22" alt="PSTNet"/>
</p>

<h1 align="center">PSTNet</h1>
<h3 align="center">Physically-Structured Turbulence Network</h3>

<p align="center">
  <em>Real-time atmospheric turbulence estimation via physics-embedded neural architecture</em>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" alt="MIT License"/></a>
  <img src="https://img.shields.io/badge/Parameters-552-8b5cf6?style=flat-square" alt="552 params"/>
  <img src="https://img.shields.io/badge/Tests-340%20Paired%20Sims-3b82f6?style=flat-square" alt="340 Paired Sims"/>
  <img src="https://img.shields.io/badge/Inference-%3C1ms-f59e0b?style=flat-square" alt="<1ms inference"/>
</p>

<p align="center">
  <strong>+2.8% Mean Improvement</strong> &middot; <strong>Cohen's <em>d</em> = 0.408</strong> &middot; <strong><em>p</em> = 1.96 &times; 10<sup>-10</sup></strong> &middot; <strong>340 Paired Simulations</strong>
</p>

---

## Abstract

For decades, estimating atmospheric turbulence at a specific altitude, in a specific region, in real time has required massive numerical weather prediction infrastructure — multi-hour update cycles, heavy compute, framework dependencies. PSTNet demonstrates that this is no longer necessary.

**PSTNet** (Physically-Structured Turbulence Network) is a lightweight neural architecture that embeds atmospheric physics directly into the network structure — not just the loss function. By fusing an analytical Monin-Obukhov backbone with regime-gated expert sub-networks under Kolmogorov spectral constraints, PSTNet computes turbulence intensity at any flight level (FL100 to FL450), anywhere in the world, from a single NASA POWER weather pull — in seconds, on laptop-class hardware.

The architecture achieves statistically significant accuracy gains validated across 340 paired Monte Carlo simulations spanning supersonic (*d* = 0.813, *p* < 0.001), high-supersonic, and hypersonic (*d* = 1.027, *p* < 0.001) regimes, with only **552 learnable parameters** — two orders of magnitude fewer than competing deep learning approaches.

---

## Key Idea

> Physics does the heavy lifting. The network refines what theory cannot close analytically.

PSTNet is not "AI replacing physics." It is **physics-structured intelligence** — an architecture where every layer, gate, and constraint has a physical interpretation:

| Component | Role | Parameters |
|:---|:---|:---:|
| Analytical Backbone | Monin-Obukhov TKE &rarr; correction mapping | **0** |
| Regime Gate (6&rarr;8&rarr;4) | Stability-class softmax routing | 84 |
| 4 Expert Sub-networks (6&rarr;8&rarr;3) | Regime-specialized residual refinement | 356 |
| FiLM Conditioning (&rho;&rarr;&gamma;,&beta;) | Density-modulated feature scaling | 112 |
| Kolmogorov Spectral Constraint | &epsilon;<sup>1/3</sup> cascade output transform | **0** |
| **Total** | | **552** |

---

## Architecture

```
                            Input: [wind, temp, density, Ri, alt, pres]
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
              │ Analytical │       │  Regime   │       │  4 Expert │
              │  Backbone  │       │   Gate    │       │ Networks  │
              │  (MO-TKE)  │       │ (softmax) │       │ + FiLM(ρ) │
              │  0 params  │       │  84 par.  │       │  468 par. │
              └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
                    │                    │                    │
                    │              ┌─────▼─────┐             │
                    │              │  Gated    │◄────────────┘
                    │              │  Mixture  │
                    │              └─────┬─────┘
                    │                    │
                    │          ┌─────────▼─────────┐
                    │          │    Kolmogorov      │
                    │          │ Spectral Constraint│
                    │          │   ε^{1/3} scaling  │
                    │          └─────────┬─────────┘
                    │                    │
                    └────────┬───────────┘
                             │
                      ┌──────▼──────┐
                      │  backbone + │
                      │  residual   │
                      └──────┬──────┘
                             │
                     Output: [strength, reliability, drift_scale]
```

### Stability Regimes

The Regime Gate classifies atmospheric state into four expert domains:

| Regime | Condition | Altitude Band | Physics |
|:---|:---|:---|:---|
| **Convective** | Ri < &minus;0.25 | Boundary layer (< 2 km) | Buoyancy-driven TKE production |
| **Neutral** | &minus;0.25 &le; Ri &le; 0.25 | Mixed | Shear-dominated, log-law profile |
| **Stable** | Ri > 0.25 | Free troposphere | Gravity-wave damping |
| **Stratospheric** | alt > 20 km | Stratosphere/mesosphere | Quasi-laminar, residual CAT |

### Physics Embedding

- **Monin-Obukhov Backbone**: Zero-parameter analytical TKE estimation using similarity theory across four atmospheric layers — boundary layer, free troposphere, tropopause/jet stream, and stratosphere. The backbone guarantees outputs remain close to physics-based estimates.

- **FiLM Density Conditioning**: Feature-wise Linear Modulation uses air density ratio (&rho;/&rho;<sub>0</sub>) to modulate expert hidden states via learned &gamma; and &beta; parameters. This captures aerodynamic authority variation with altitude without explicit altitude encoding.

- **Kolmogorov Spectral Constraint**: The primary output channel applies &epsilon;<sup>1/3</sup> scaling (turbulent energy cascade law) to the neural residual, ensuring the learned corrections respect the universal &minus;5/3 power spectrum of homogeneous isotropic turbulence.

---

## Benchmark Results

### Overall Performance (340 paired sims, 1000 m guidance validity threshold)

| Model | Params | Corr Miss (m) | &Delta;% | Win% | *d* | *p* | Sig |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| **PSTNet (ours)** | **552** | **—** | **+2.8** | **78** | **+0.408** | **1.96e-10** | *** |
| Vanilla MLP | 627 | — | +1.4 | 62 | +0.185 | 2.1e-03 | ** |
| Deep MLP (10&times;) | 6,819 | — | +1.1 | 58 | +0.142 | 1.8e-02 | * |
| GBT Ensemble | ~9,000 nodes | — | +0.9 | 55 | +0.108 | 6.4e-02 | ns |
| Dryden Classical | 0 | — | +0.6 | 51 | +0.071 | 1.9e-01 | ns |

### Per-Regime Effect Sizes

| Regime | Cohen's *d* | *p*-value | Effect Size | Interpretation |
|:---|:---:|:---:|:---:|:---|
| Hypersonic (M 8.0) | **1.027** | < 0.001 | Large | Strong correction signal at stratospheric cruise |
| Supersonic (M 2.8) | **0.813** | < 0.001 | Large | Boundary-layer & jet-stream compensation |
| High Supersonic (M 4.5) | 0.456 | < 0.01 | Medium | Tropopause transition regime |

### Statistical Validation

- **Paired *t*-test**: *p* = 1.96 &times; 10<sup>-10</sup> (highly significant)
- **Wilcoxon signed-rank**: Non-parametric confirmation
- **Friedman test**: Multi-model ranking with Nemenyi post-hoc critical difference
- **Bootstrap 95% CI**: 5,000 resamples
- **Head-to-head**: PSTNet vs each baseline with paired effect sizes

---

## System Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PSTNet System                                │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  NASA    │───▶│   Weather    │───▶│  Turbulence  │               │
│  │  POWER   │    │   Service    │    │    Field     │               │
│  │  API     │    │ (weather_api)│    │ (turb_model) │               │
│  └──────────┘    └──────────────┘    └──────┬───────┘               │
│                                             │                        │
│                                     ┌───────▼───────┐               │
│                                     │    PSTNet     │               │
│                                     │  Predictor    │               │
│                                     │  (552 params) │               │
│                                     └───────┬───────┘               │
│                                             │                        │
│       ┌─────────────────────────────────────┼──────────────┐        │
│       │                                     │              │        │
│  ┌────▼────┐    ┌──────────────┐    ┌──────▼───────┐      │        │
│  │ Globe   │    │  Trajectory  │    │  Validation  │      │        │
│  │ Viz     │    │  Simulator   │    │  Suite       │      │        │
│  │(main.py)│    │(trajectory.py│    │ (tests.py)   │      │        │
│  └─────────┘    └──────────────┘    └──────────────┘      │        │
└──────────────────────────────────────────────────────────────────────┘
```

### Modules

| File | Purpose | Lines |
|:---|:---|---:|
| `turbulence_model.py` | PSTNet architecture — backbone, gate, experts, FiLM, forward/backward | ~650 |
| `trajectory.py` | 6-DoF trajectory simulator with drag, gravity, PN guidance, ML correction | ~320 |
| `weather_api.py` | NASA POWER API integration with ISA vertical profile generation | ~113 |
| `config.py` | Physical constants, vehicle specifications, scenario definitions | ~75 |
| `tests.py` | 340-pair comparative validation suite with 5 models & full statistics | ~1530 |
| `main.py` | Flask web app — turbulence globe, trajectory visualizer, dashboard | ~890 |

---

## Validation Suite

The test suite (`tests.py`) compares PSTNet against four baselines under identical conditions:

| Baseline | Type | Parameters | Purpose |
|:---|:---|---:|:---|
| Vanilla MLP | 6&rarr;24&rarr;16&rarr;3 | 627 | Ablation — same budget, no regime gating |
| Deep MLP | 6&rarr;64&rarr;64&rarr;32&rarr;3 | 6,819 | Capacity test — 10&times; parameters |
| GBT Ensemble | 3&times;200 trees | ~9,000 nodes | Non-neural ML baseline |
| Dryden (MIL-HDBK-1797A) | Analytical | 0 | Classical atmospheric model |

### Test Categories (340 pairs &times; 5 models = 1,700 simulations)

| Cat. | Tests | Description |
|:---|---:|:---|
| A | 15 | Validated standard scenarios (HIGH_ALT, STRAT) |
| B | 25 | Optimal altitude envelope (5 altitudes &times; 5 seeds) |
| C | 25 | Effective range band (60&ndash;180 km) |
| D | 35 | Lateral engagement (&minus;30 to +30 km offsets) |
| E | 15 | Edge cases (3&ndash;22 km altitudes) |
| F | 20 | Monte Carlo hypersonic validation |

### Operational Envelope

| Vehicle | Mach | Altitude | Range |
|:---|:---:|:---|:---|
| Supersonic | 2.8 | 3&ndash;15 km | 60&ndash;120 km |
| High Supersonic | 4.5 | 12&ndash;22 km | Select geometries |
| Hypersonic Glide | 8.0 | 18&ndash;25 km | 120&ndash;180 km |

---

## Quick Start

### Requirements

- Python 3.10+
- No GPU required
- No deep learning framework dependencies

### Installation

```bash
git clone https://github.com/BorisKriuk/PSTNet.git
cd PSTNet

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Run Validation Suite

```bash
# Full suite (340 tests × 5 models = 1,700 paired sims)
python tests.py

# Quick smoke test (20 tests)
python tests.py --quick

# Export detailed results to JSON
python tests.py --json results.json
```

### Run Web Interface

```bash
python main.py
# → http://localhost:5000          Dashboard
# → http://localhost:5000/globe    Turbulence globe visualization
# → http://localhost:5000/trajectory   3D trajectory simulator
```

### Dependencies

```
flask==3.0.0
numpy==1.24.0
scipy==1.11.0
scikit-learn==1.3.0
```

No PyTorch. No TensorFlow. No CUDA. Pure NumPy + SciPy.

---

## How It Works

### Training

PSTNet trains on vertically-resolved atmospheric profiles from 12 geographically diverse locations. Each profile provides `[wind, temperature, density, Richardson number, altitude, pressure]` at 11 altitude layers (0&ndash;30 km). Data augmentation (8&times; Gaussian perturbation + 3&times; interpolation) expands the training set. Training uses momentum SGD with cosine-decayed learning rate and regime supervision warm-up.

### Inference

A single forward pass:
1. **Backbone** computes physics-based correction from Monin-Obukhov TKE
2. **Regime Gate** classifies stability state &rarr; softmax over 4 experts
3. **Experts + FiLM** produce density-conditioned residual corrections
4. **Spectral constraint** transforms output channel 0 via &epsilon;<sup>1/3</sup> scaling
5. **Output** = backbone + constrained residual (clipped to physical bounds)

Inference latency: **< 1 ms** per atmospheric query.

### Real-Time Weather

PSTNet pulls live meteorological data from the [NASA POWER API](https://power.larc.nasa.gov/) — surface wind speed, temperature, pressure, humidity, precipitation, and cloud cover — and constructs a full vertical atmospheric profile using ISA lapse rates calibrated to measured surface conditions.

---

## Key Design Decisions

1. **Physics in the structure, not just the loss.** The analytical backbone ensures every prediction starts from a physically grounded estimate. The network only needs to learn the residual — what theory cannot close analytically.

2. **Regime gating over monolithic networks.** Atmospheric turbulence is fundamentally multi-regime (convective, neutral, stable, stratospheric). A single MLP cannot efficiently partition this space; four specialized experts with soft gating can.

3. **FiLM over feature concatenation.** Density modulates aerodynamic authority multiplicatively, not additively. Feature-wise Linear Modulation captures this physical relationship with minimal parameters.

4. **Kolmogorov constraint over unconstrained outputs.** The &minus;5/3 spectral law is universal for homogeneous isotropic turbulence. Hard-wiring &epsilon;<sup>1/3</sup> scaling into the output prevents the network from producing spectrally inconsistent corrections.

5. **552 parameters over thousands.** Fewer parameters mean faster training, less overfitting, and deployment on any hardware. PSTNet achieves higher effect sizes than a 6,819-parameter Deep MLP.

---

## Citation

```bibtex
@software{kriuk2026pstnet,
  author    = {Kriuk, Boris},
  title     = {{PSTNet}: Physically-Structured Turbulence Network for
               Real-Time Atmospheric Turbulence Estimation},
  year      = {2026},
  url       = {https://github.com/BorisKriuk/PSTNet},
  note      = {552-parameter regime-gated physics-informed architecture}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
