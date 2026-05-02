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
  <strong>Cohen's <em>d</em> = 0.408</strong> &middot; <strong><em>p</em> = 1.96 &times; 10<sup>-10</sup></strong> &middot; <strong>340 Paired Simulations</strong> &middot; <strong>552 Parameters</strong>
</p>

---

## Abstract

Estimating atmospheric turbulence at a specific altitude, in a specific region, in real time has historically required massive numerical weather prediction infrastructure. PSTNet demonstrates that this is no longer necessary.

**PSTNet** embeds atmospheric physics directly into the network structure — not just the loss function. An analytical Monin-Obukhov backbone, regime-gated expert sub-networks, and Kolmogorov spectral constraints compute turbulence intensity at any flight level (FL100 to FL450), anywhere in the world, from a single NASA POWER weather pull — in seconds, on laptop-class hardware, with only **552 learnable parameters**.

Validated across 340 paired Monte Carlo simulations: supersonic (*d* = 0.813, *p* < 0.001), high-supersonic, and hypersonic (*d* = 1.027, *p* < 0.001) regimes. No PyTorch. No TensorFlow. No GPU. Pure NumPy.

---

## Architecture

> Physics does the heavy lifting. The network refines what theory cannot close analytically.

| Component | Role | Parameters |
|:---|:---|:---:|
| Analytical Backbone | Monin-Obukhov TKE &rarr; correction mapping | **0** |
| Regime Gate (6&rarr;8&rarr;4) | Stability-class softmax routing | 84 |
| 4 Expert Sub-networks (6&rarr;8&rarr;3) | Regime-specialized residual refinement | 356 |
| FiLM Conditioning (&rho;&rarr;&gamma;,&beta;) | Density-modulated feature scaling | 112 |
| Kolmogorov Spectral Constraint | &epsilon;<sup>1/3</sup> cascade output transform | **0** |
| **Total** | | **552** |

```
                            Input: [wind, temp, density, Ri, alt, pres]
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
              │ Analytical│       │  Regime   │       │  4 Expert │
              │  Backbone │       │   Gate    │       │ Networks  │
              │  (MO-TKE) │       │ (softmax) │       │ + FiLM(ρ) │
              │  0 params │       │  84 par.  │       │  468 par. │
              └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
                    │                    │                  │
                    │              ┌─────▼─────┐            │
                    │              │  Gated    │◄───────────┘
                    │              │  Mixture  │
                    │              └─────┬─────┘
                    │                    │
                    │          ┌─────────▼─────────┐
                    │          │    Kolmogorov     │
                    │          │ Spectral Constr.  │
                    │          │   ε^{1/3} scaling │
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

The Regime Gate routes inputs to four stability-regime experts:

| Regime | Condition | Physics |
|:---|:---|:---|
| **Convective** | Ri < &minus;0.25 | Buoyancy-driven TKE, boundary layer |
| **Neutral** | &minus;0.25 &le; Ri &le; 0.25 | Shear-dominated, log-law profile |
| **Stable** | Ri > 0.25 | Gravity-wave damping, free troposphere |
| **Stratospheric** | alt > 20 km | Quasi-laminar, residual CAT |

---

## Results

### Overall (340 paired sims, 1000 m guidance threshold)

| Model | Params | &Delta;% | Win% | *d* | *p* | Sig |
|:---|---:|---:|---:|---:|---:|:---:|
| **PSTNet (ours)** | **552** | **+2.8** | **78** | **+0.408** | **1.96e-10** | *** |
| Vanilla MLP | 627 | +1.4 | 62 | +0.185 | 2.1e-03 | ** |
| Deep MLP (10&times;) | 6,819 | +1.1 | 58 | +0.142 | 1.8e-02 | * |
| GBT Ensemble | ~9,000 | +0.9 | 55 | +0.108 | 6.4e-02 | ns |
| Dryden Classical | 0 | +0.6 | 51 | +0.071 | 1.9e-01 | ns |

### Per-Regime Effect Sizes

| Regime | Cohen's *d* | *p*-value | Effect |
|:---|:---:|:---:|:---:|
| Hypersonic (M 8.0) | **1.027** | < 0.001 | Large |
| Supersonic (M 2.8) | **0.813** | < 0.001 | Large |
| High Supersonic (M 4.5) | 0.456 | < 0.01 | Medium |

Statistical validation: paired *t*-test, Wilcoxon signed-rank, Friedman + Nemenyi post-hoc, bootstrap 95% CI (5,000 resamples).

---

## Quick Start

```bash
git clone https://github.com/BorisKriuk/PSTNet.git
cd PSTNet
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Validation suite (340 tests × 5 models)
python tests.py

# Publication figures (10 figures → figures/)
python visualize.py

# Web interface
python main.py
# → localhost:5000/globe       Turbulence globe
# → localhost:5000/trajectory  3D trajectory simulator
```

Dependencies: `flask`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`. No GPU. No deep learning frameworks.

---

## Project Structure

| File | Purpose |
|:---|:---|
| `turbulence_model.py` | PSTNet architecture — backbone, gate, experts, FiLM, forward/backward |
| `trajectory.py` | 6-DoF trajectory simulator with drag, gravity, PN guidance, ML correction |
| `weather_api.py` | NASA POWER API integration with ISA vertical profile generation |
| `config.py` | Physical constants, vehicle specs, scenario definitions |
| `tests.py` | 340-pair comparative validation suite (5 models, 6 categories, full statistics) |
| `visualize.py` | 10 publication-quality figures (PDF + PNG) |
| `main.py` | Flask web app — turbulence globe, trajectory visualizer |

---

## Citation

```bibtex
@software{kriuk2026pstnet,
  author    = {Kriuk, Boris and Kriuk, Fedor},
  title     = {{PSTNet}: Physically-Structured Turbulence Network for
               Real-Time Atmospheric Turbulence Estimation},
  year      = {2026},
  url       = {https://github.com/BorisKriuk/PSTNet},
  note      = {552-parameter regime-gated physics-informed architecture}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
