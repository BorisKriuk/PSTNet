# 🎯 PSTNet

### Predictive Strike Trajectory Network

*A physics-informed neural network framework for real-time missile intercept trajectory optimization*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0-purple?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-340_Paired_Sims-blue?style=flat-square)]()

---

**+2.8% Mean Improvement** · **Cohen's d = 0.408** · **p = 1.96e-10** · **340 Paired Simulations**

---

## 📖 Overview

PSTNet replaces static trajectory look-up tables with a lightweight neural correction layer trained on diverse atmospheric profiles. The network ingests real-time weather telemetry — wind vectors, pressure, temperature, humidity — and outputs per-timestep trajectory adjustments that respect aerodynamic and structural constraints.

By fusing physics-based guidance laws with learned turbulence compensation, PSTNet delivers statistically significant accuracy gains for **supersonic** (d = 0.813) and **hypersonic** (d = 1.027) engagement scenarios, validated across 340 paired Monte Carlo simulations spanning 51 unique configurations.

---

## ⚡ Key Features

### 🌦️ Live Weather Integration
Fetches real-time atmospheric data via `weather_api.py` — wind vectors, pressure gradients, temperature, and humidity — for location-specific trajectory correction.

### 🌪️ Multi-Regime Turbulence
Von Kármán and Dryden spectral models generate realistic turbulence fields across altitude bands. Supports low-level shear, jet-stream, and boundary-layer turbulence.

### 🧠 Neural Correction Layer
Lightweight 3-layer MLP learns residual trajectory corrections on top of physics-based guidance, maintaining real-time inference at <1 ms per timestep.

### 🛡️ Constraint Enforcement
Hard aerodynamic and structural limits (max-g, dynamic pressure, angle-of-attack) are enforced post-correction to guarantee physically valid trajectories.

### 🎯 Multi-Profile Training
Trains across 12 geographically diverse atmospheric profiles to prevent overfitting to a single weather regime, improving generalization to unseen conditions.

### 📊 Statistical Validation
340 paired simulations across 6 categories and 51 configurations with guidance validity filtering (1000 m threshold) for rigorous hypothesis testing.

---

## 📈 Benchmark Results

Filtered analysis using 1000 m guidance validity threshold. Effect sizes reported as Cohen's d from paired-sample t-tests.

| Missile Category | Cohen's d | p-value | Mean Δ Miss | Effect Size |
|:---|:---:|:---:|:---:|:---:|
| **Hypersonic** | **1.027** | < 0.001 | −4.2 m | 🟢 Large |
| **Supersonic** | **0.813** | < 0.001 | −3.1 m | 🟢 Large |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BorisKriuk/PSTNet.git
cd PSTNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run Main <1 or 2>
python main<1,2>.py
```

## 📄 License & Citation
# MIT License.

@software{kriuk2026pstnet,
  author  = {Kriuk, Boris},
  title   = {PSTNet: Predictive Strike Trajectory Network},
  year    = {2026},
  url     = {https://github.com/BorisKriuk/PSTNet}
}
