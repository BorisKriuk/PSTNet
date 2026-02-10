# turbulence_model.py
"""PSTNet — Physically-Structured Turbulence Network

A novel lightweight neural architecture for universal atmospheric turbulence
estimation, where physics is embedded in the network STRUCTURE — not just the
loss function.

Architectural innovations:
  1. Analytical Backbone      — Monin-Obukhov base (0 learnable params)
  2. Regime-Gated MoE         — 4 stability-regime expert sub-networks
  3. FiLM Density Conditioning — ρ modulates expert hidden states
  4. Kolmogorov Spectral Constraint — output derived via ε^{1/3} scaling

Total learnable parameters: ~552
"""

import math
import numpy as np
from config import ALTITUDE_LAYERS, AIR_DENSITY_SEA


# =====================================================================
#  PSTNet  –  core predictor
# =====================================================================
class TurbulencePredictor:
    """PSTNet-based turbulence predictor.

    Drop-in replacement for the original 6→16→3 MLP, same public API:
        .fit(profiles)
        .predict(wind, temp, density, ri, alt, pres) → dict
        .physics_turbulence(alt, wind, ri, density)   (static)
        .trained, .loss_history
    """

    REGIME_NAMES = ['convective', 'neutral', 'stable', 'stratospheric']
    N_EX = 4           # number of regime experts
    GH   = 8           # gate hidden dim
    EH   = 8           # expert hidden dim
    NI   = 6           # input features
    NO   = 3           # output features

    # ---- init --------------------------------------------------------
    def __init__(self):
        np.random.seed(42)
        H = self.EH

        # Regime Gate  6 → 8 → 4
        self.Wg1 = np.random.randn(self.NI, self.GH) * np.sqrt(2 / self.NI)
        self.bg1 = np.zeros(self.GH)
        self.Wg2 = np.random.randn(self.GH, self.N_EX) * np.sqrt(2 / self.GH)
        self.bg2 = np.zeros(self.N_EX)

        # Experts  4 × (6 → 8 → 3)
        self.We1 = [np.random.randn(self.NI, H) * np.sqrt(2 / self.NI)
                    for _ in range(self.N_EX)]
        self.be1 = [np.zeros(H) for _ in range(self.N_EX)]
        self.We2 = [np.random.randn(H, self.NO) * np.sqrt(2 / H)
                    for _ in range(self.N_EX)]
        self.be2 = [np.zeros(self.NO) for _ in range(self.N_EX)]

        # FiLM  density → (γ, β) per expert   1 → 2H
        self.Wf = [np.random.randn(1, 2 * H) * 0.05
                   for _ in range(self.N_EX)]
        self.bf = [np.concatenate([np.ones(H), np.zeros(H)])
                   for _ in range(self.N_EX)]

        self.mean = self.std = None
        self.trained = False
        self.loss_history = []

    # ---- activations -------------------------------------------------
    @staticmethod
    def _tanh(x):
        return np.tanh(np.clip(x, -10, 10))

    @staticmethod
    def _softmax(x):
        s = x - x.max(axis=-1, keepdims=True)
        e = np.exp(s)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-10)

    # ==================================================================
    #  COMPONENT 1 — Analytical Backbone  (zero learnable parameters)
    # ==================================================================
    def analytical_backbone(self, raw):
        """Monin-Obukhov + altitude-regime base prediction.

        Provides a guaranteed-physical base that the neural component
        only needs to *correct*, not reproduce from scratch.

        Parameters
        ----------
        raw : ndarray (N, 6)  [wind, temp, density, Ri, alt, pres]

        Returns
        -------
        base : ndarray (N, 3) [strength, reliability, drift_scale]
        """
        wind, temp, density, ri, alt, pres = (raw[:, i] for i in range(6))
        N   = len(wind)
        base = np.zeros((N, 3))
        rr  = density / AIR_DENSITY_SEA

        for j in range(N):
            a, w, r = float(alt[j]), float(wind[j]), float(ri[j])
            z = max(a * 1000.0, 10.0)                     # metres AGL
            u_star = max(0.4 * w / (np.log(z / 0.1 + 1) + 1e-6), 0.01)

            # ------ TKE by atmospheric regime ------
            if a < 2:                                       # boundary layer
                phi = (1 - 16 * r) ** -0.25 if r < 0 else 1 + 5 * min(r, 2.0)
                tke = u_star ** 2 / max(0.3 * phi, 0.01)
            elif a < 12:                                    # free troposphere
                tke = max(0.02 * w / max(a, 0.5), 1e-4) * np.exp(-(a - 2) / 10)
            elif a < 20:                                    # tropopause / jet
                tke = max(5e-3 * w * np.exp(-abs(a - 12) / 5), 1e-5)
            else:                                           # stratosphere
                tke = max(1e-3 * np.exp(-(a - 20) / 15), 1e-6)

            ti = np.sqrt(max(tke, 1e-8))
            base[j, 0] = np.clip(0.15 + 0.40 * ti * rr[j] ** 0.3,  0.10, 0.80)
            base[j, 1] = np.clip(0.40 * rr[j] / (1 + 2 * ti),      0.15, 0.85)
            base[j, 2] = np.clip(0.20 * rr[j] + 0.05,              0.05, 0.55)
        return base

    # ==================================================================
    #  FORWARD PASS
    # ==================================================================
    def forward(self, Xn, density, raw):
        """
        output = backbone(raw) + spectral_constrained_residual(neural(Xn, ρ))

        Returns  (output, cache_dict)
        """
        N = Xn.shape[0];  H = self.EH
        dn = (density / AIR_DENSITY_SEA).reshape(-1, 1)    # FiLM input
        rr = density / AIR_DENSITY_SEA                      # for spectral
        phys = self.analytical_backbone(raw)

        # ---- Component 2: Regime Gate --------------------------------
        gp = Xn @ self.Wg1 + self.bg1;    gh = self._tanh(gp)
        gl = gh @ self.Wg2 + self.bg2;    gates = self._softmax(gl)   # (N, 4)

        # ---- Component 3: Experts + FiLM -----------------------------
        e_out, ec = [], []
        for i in range(self.N_EX):
            ep = Xn @ self.We1[i] + self.be1[i];   eh = self._tanh(ep)
            fr = dn @ self.Wf[i]  + self.bf[i]                   # (N, 2H)
            gam, bet = fr[:, :H], fr[:, H:]
            mp = gam * eh + bet;                    mh = self._tanh(mp)
            eo = mh @ self.We2[i] + self.be2[i]                  # (N, 3)
            e_out.append(eo)
            ec.append(dict(ep=ep, eh=eh, gam=gam, bet=bet, mp=mp, mh=mh))

        # ---- Gated mixture -------------------------------------------
        mix = sum(gates[:, i:i+1] * e_out[i] for i in range(self.N_EX))

        # ---- Component 4: Spectral constraint (ch 0) + bounded (1,2) -
        t0 = self._tanh(mix[:, 0])
        sf = np.exp(t0 / 3.0) - 1.0            # Kolmogorov  ε^{1/3}  scaling
        sq = np.sqrt(np.maximum(rr, 1e-4))      # √(ρ/ρ₀) — aero authority
        res0 = 0.50 * sf * sq

        t1 = self._tanh(mix[:, 1]);   res1 = 0.15 * t1
        t2 = self._tanh(mix[:, 2]);   res2 = 0.10 * t2

        pre = phys + np.column_stack([res0, res1, res2])
        out = pre.copy()
        out[:, 0] = np.clip(out[:, 0], 0.10, 0.90)
        out[:, 1] = np.clip(out[:, 1], 0.10, 0.95)
        out[:, 2] = np.clip(out[:, 2], 0.05, 0.70)

        cache = dict(Xn=Xn, dn=dn, rr=rr, sq=sq, phys=phys,
                     gp=gp, gh=gh, gl=gl, gates=gates,
                     e_out=e_out, ec=ec, mix=mix,
                     t0=t0, sf=sf, t1=t1, t2=t2,
                     res0=res0, res1=res1, res2=res2,
                     pre=pre, out=out)
        return out, cache

    # ==================================================================
    #  BACKWARD PASS  (analytical gradients for all learnable params)
    # ==================================================================
    def backward(self, cache, Y):
        c = cache;  N = c['Xn'].shape[0];  H = self.EH
        d = 2.0 * (c['out'] - Y) / N

        # ---- output clip masks ----------------------------------------
        bounds = [(0.10, 0.90), (0.10, 0.95), (0.05, 0.70)]
        for ch, (lo, hi) in enumerate(bounds):
            mask = ((c['pre'][:, ch] > lo) & (c['pre'][:, ch] < hi)).astype(float)
            d[:, ch] *= mask

        # ---- channel 0  (spectral constraint) -------------------------
        d_t0   = d[:, 0] * 0.5 * np.exp(c['t0'] / 3.0) / 3.0 * c['sq']
        d_mix0 = d_t0 * (1.0 - c['t0'] ** 2)

        # ---- channels 1, 2  (bounded residual) -----------------------
        d_mix1 = d[:, 1] * 0.15 * (1.0 - c['t1'] ** 2)
        d_mix2 = d[:, 2] * 0.10 * (1.0 - c['t2'] ** 2)

        dm = np.column_stack([d_mix0, d_mix1, d_mix2])

        # ---- gated mixture backward ----------------------------------
        deo = [c['gates'][:, i:i+1] * dm for i in range(self.N_EX)]
        dg_cols = [(c['e_out'][i] * dm).sum(axis=1) for i in range(self.N_EX)]
        dg = np.column_stack(dg_cols)                            # (N, 4)

        # softmax Jacobian
        g = c['gates']
        dl = g * (dg - (g * dg).sum(axis=1, keepdims=True))     # (N, 4)

        G = {}
        # ---- gate params ---------------------------------------------
        G['Wg2'] = c['gh'].T @ dl;        G['bg2'] = dl.sum(0)
        dgh = dl @ self.Wg2.T
        dgp = dgh * (1.0 - c['gh'] ** 2)
        G['Wg1'] = c['Xn'].T @ dgp;      G['bg1'] = dgp.sum(0)

        # ---- expert + FiLM params ------------------------------------
        for i in range(self.N_EX):
            e = c['ec'][i]
            G[f'We2_{i}'] = e['mh'].T @ deo[i]
            G[f'be2_{i}'] = deo[i].sum(0)
            dmh = deo[i] @ self.We2[i].T

            dmp = dmh * (1.0 - e['mh'] ** 2)
            dgv = dmp * e['eh']
            dbv = dmp
            deh = dmp * e['gam']

            dfr = np.concatenate([dgv, dbv], axis=1)
            G[f'Wf_{i}'] = c['dn'].T @ dfr
            G[f'bf_{i}'] = dfr.sum(0)

            dep = deh * (1.0 - e['eh'] ** 2)
            G[f'We1_{i}'] = c['Xn'].T @ dep
            G[f'be1_{i}'] = dep.sum(0)

        return G

    # ==================================================================
    #  TRAINING
    # ==================================================================
    def fit(self, profiles, epochs=300, lr=0.004):
        X, Y = [], []
        for prof in profiles:
            for L in prof:
                x = [L['wind_speed'], L['temperature'], L['density'],
                     L['richardson'], L['altitude'], L['pressure']]
                tii = self.physics_turbulence(
                    L['altitude'], L['wind_speed'],
                    L['richardson'], L['density'])
                dr = L['density'] / AIR_DENSITY_SEA
                # ── Correction-strength targets (tuned for ~15-25 %
                #    cancellation of deterministic turbulence) ─────────
                Y.append([
                    float(np.clip(0.40 + 0.35 * tii + 0.20 * dr,
                                  0.25, 0.90)),
                    float(np.clip((0.45 + 0.45 * dr) / (1 + 1.0 * tii),
                                  0.20, 0.95)),
                    float(np.clip(0.35 * dr + 0.10,
                                  0.05, 0.70)),
                ])
                X.append(x)
        X, Y = np.array(X), np.array(Y)
        if len(X) < 2:
            return

        # ---- data augmentation ----------------------------------------
        rng = np.random.RandomState(123)
        Xa, Ya = [X], [Y]
        for _ in range(8):
            Xa.append(X + rng.randn(*X.shape) * 0.03 * X.std(0))
            Ya.append(Y.copy())
        for i in range(len(X) - 1):
            for f in (0.25, 0.50, 0.75):
                Xa.append(((1 - f) * X[i] + f * X[i + 1]).reshape(1, -1))
                Ya.append(((1 - f) * Y[i] + f * Y[i + 1]).reshape(1, -1))
        X = np.vstack(Xa);  Y = np.vstack(Ya)

        self.mean = X.mean(0);  self.std = X.std(0) + 1e-8
        Xn   = (X - self.mean) / self.std
        dens = X[:, 2]

        self.loss_history = []
        for ep in range(epochs):
            lr_t = lr * (1 - 0.5 * ep / epochs)
            out, cache = self.forward(Xn, dens, X)
            loss = float(np.mean((out - Y) ** 2))
            self.loss_history.append(loss)

            G = self.backward(cache, Y)
            for k in G:
                np.clip(G[k], -5, 5, out=G[k])

            self.Wg1 -= lr_t * G['Wg1'];  self.bg1 -= lr_t * G['bg1']
            self.Wg2 -= lr_t * G['Wg2'];  self.bg2 -= lr_t * G['bg2']
            for i in range(self.N_EX):
                self.We1[i] -= lr_t * G[f'We1_{i}']
                self.be1[i] -= lr_t * G[f'be1_{i}']
                self.We2[i] -= lr_t * G[f'We2_{i}']
                self.be2[i] -= lr_t * G[f'be2_{i}']
                self.Wf[i]  -= lr_t * G[f'Wf_{i}']
                self.bf[i]  -= lr_t * G[f'bf_{i}']

        self.trained = True
        print(f"PSTNet trained — loss {self.loss_history[-1]:.6f}  "
              f"({len(X)} samples, {epochs} epochs)")

    # ==================================================================
    #  INFERENCE
    # ==================================================================
    def predict(self, wind, temp, density, ri, alt, pres):
        if self.mean is None:
            return dict(correction_strength=0.4,
                        reliability=0.4, drift_scale=0.15)
        x = np.array([[wind, temp, density, ri, alt, pres]])
        xn = (x - self.mean) / self.std
        out, _ = self.forward(xn, np.array([density]), x)
        return dict(
            correction_strength=float(out[0, 0]),
            reliability=float(out[0, 1]),
            drift_scale=float(out[0, 2]),
        )

    # ==================================================================
    #  REGIME ANALYSIS  (for paper figures)
    # ==================================================================
    def get_regime_weights(self, wind, temp, density, ri, alt, pres):
        if self.mean is None:
            return {n: 0.25 for n in self.REGIME_NAMES}
        x  = np.array([[wind, temp, density, ri, alt, pres]])
        xn = (x - self.mean) / self.std
        gp = xn @ self.Wg1 + self.bg1
        gh = self._tanh(gp)
        gl = gh @ self.Wg2 + self.bg2
        gates = self._softmax(gl)[0]
        return dict(zip(self.REGIME_NAMES, gates.tolist()))

    # ==================================================================
    #  PHYSICS BASELINE
    # ==================================================================
    @staticmethod
    def physics_turbulence(alt, wind, ri, density):
        if alt < 0.5:
            b, w = 0.50, min(wind / 10, 1) * 0.30
            s = -0.20 if ri > 0.25 else 0.10
        elif alt < 2:
            b, w = 0.40, min(wind / 15, 1) * 0.25
            s = -0.15 if ri > 0.25 else 0.10
        elif alt < 5:
            b, w = 0.25, min(wind / 20, 1) * 0.15
            s = -0.10 if ri > 0.50 else 0.05
        elif alt < 12:
            b = 0.15
            w = min(wind / 30, 1) * 0.10 + 0.10 * np.exp(-((alt - 10) ** 2) / 4)
            s = -0.05 if ri > 1 else 0.02
        elif alt < 20:
            b, w = 0.08, min(wind / 40, 1) * 0.05
            s = -0.03
        else:
            b = 0.03
            w = min(wind / 50, 1) * 0.02 * np.exp(-(alt - 20) / 10)
            s = -0.02
        return float(np.clip(b + w + s, 0.01, 0.80))


# =====================================================================
#  TurbulenceField
# =====================================================================
class TurbulenceField:
    def __init__(self, weather_service):
        self.weather = weather_service
        self.predictor = TurbulencePredictor()
        self.profile = None
        self.turb_by_alt = {}
        self.ready = False

    def update(self):
        print("Building turbulence field …")
        self.profile = self.weather.get_vertical_profile(ALTITUDE_LAYERS)
        for L in self.profile:
            print(f"  {L['altitude']:5.1f} km  wind={L['wind_speed']:5.1f} m/s "
                  f"T={L['temperature']:.0f} K  ρ={L['density']:.4f}")

        self.predictor.fit([self.profile])

        self.turb_by_alt = {}
        for L in self.profile:
            self.turb_by_alt[L['altitude']] = self.predictor.physics_turbulence(
                L['altitude'], L['wind_speed'], L['richardson'], L['density'])
        print("Turbulence:")
        for a in sorted(self.turb_by_alt):
            print(f"  {a:5.1f} km : {self.turb_by_alt[a] * 100:5.1f}%")

        print("Regime gate weights:")
        for L in self.profile:
            rw = self.predictor.get_regime_weights(
                L['wind_speed'], L['temperature'], L['density'],
                L['richardson'], L['altitude'], L['pressure'])
            dominant = max(rw, key=rw.get)
            weights = '  '.join(f"{k[:4]}={v:.2f}" for k, v in rw.items())
            print(f"  {L['altitude']:5.1f} km  {weights}  → {dominant}")

        self.ready = True

    # ---- interpolated look-up ----------------------------------------
    def get_at(self, altitude):
        if not self.ready:
            self.update()
        below, above = self.profile[0], self.profile[-1]
        for i in range(len(self.profile) - 1):
            if self.profile[i]['altitude'] <= altitude <= self.profile[i + 1]['altitude']:
                below, above = self.profile[i], self.profile[i + 1]
                break
        span = above['altitude'] - below['altitude']
        f = (altitude - below['altitude']) / span if span > 0 else 0.0
        f = np.clip(f, 0, 1)
        layer = {k: below[k] * (1 - f) + above[k] * f
                 for k in ('wind_speed', 'temperature', 'density',
                           'richardson', 'pressure')}
        layer['altitude'] = altitude
        tii = self.predictor.physics_turbulence(
            altitude, layer['wind_speed'], layer['richardson'], layer['density'])
        return tii, layer

    # ---- ML correction (kept for backward compat) --------------------
    def ml_correction(self, altitude, turb_pert, wind_pert, layer):
        if not self.predictor.trained:
            return np.zeros(3), 0.0

        p = self.predictor.predict(
            layer['wind_speed'], layer['temperature'], layer['density'],
            layer['richardson'], altitude, layer['pressure'])

        dr = layer['density'] / AIR_DENSITY_SEA
        density_authority = min(dr, 1.0) ** 0.5

        turb_corr = (-turb_pert
                     * p['correction_strength']
                     * p['reliability']
                     * density_authority)

        drift_corr = -wind_pert * p['drift_scale'] * p['reliability']

        corr = turb_corr + drift_corr
        conf = float(p['correction_strength'] * p['reliability'] * density_authority)
        return corr, conf

    # ---- spatial turbulence direction (deterministic) -----------------
    @staticmethod
    def _spatial_turb_direction(x, y, alt_km):
        """Multi-harmonic spatial turbulence direction field.

        Returns a unit-length 3-D direction that varies smoothly in
        space, representing the large-scale (predictable) turbulence
        pattern.  The small-scale (unpredictable) component is added
        as stochastic jitter by the caller.
        """
        p1 = x * 0.07 + y * 0.11 + alt_km * 0.23
        p2 = x * 0.13 + y * 0.07 + alt_km * 0.17
        p3 = x * 0.09 + y * 0.15 + alt_km * 0.11

        dx = math.sin(p1) * 0.7 + math.sin(p1 * 2.3 + 1.1) * 0.3
        dy = math.sin(p2) * 0.7 + math.cos(p2 * 1.7 + 0.7) * 0.3
        dz = math.sin(p3) * 0.5 + math.cos(p3 * 2.1 + 1.3) * 0.2

        vec = np.array([dx, dy, dz])
        n = np.linalg.norm(vec)
        if n > 1e-10:
            vec /= n
        return vec

    # ---- sample: turbulence vector at a world position ---------------
    def sample(self, x, y, alt_km):
        """Return (direction_vector, intensity) at (x, y, alt_km).

        ``direction_vector`` has roughly unit magnitude and captures
        the deterministic spatial pattern plus a small wind-driven
        bias.  The caller multiplies by ``pert_scale`` (which includes
        intensity) to obtain an acceleration.
        """
        tii, layer = self.get_at(alt_km)

        turb_dir = self._spatial_turb_direction(x, y, alt_km)

        # Consistent wind-driven bias (always present, correctable)
        wind = layer['wind_speed']
        wind_bias = np.array([0.15, 0.05, -0.02]) * (wind / 20.0)

        return turb_dir + wind_bias, tii

    # ---- get_correction: ML-predicted correction vector --------------
    def get_correction(self, x, y, alt_km, vx, vy, vz):
        """Return a correction vector in the same scale as ``sample()``
        output.

        The caller multiplies by ``pert_scale * confidence`` to obtain
        an acceleration that partially cancels the deterministic
        turbulence component.
        """
        tii, layer = self.get_at(alt_km)
        if not self.predictor.trained:
            return np.zeros(3)

        p = self.predictor.predict(
            layer['wind_speed'], layer['temperature'], layer['density'],
            layer['richardson'], alt_km, layer['pressure'])

        turb_dir = self._spatial_turb_direction(x, y, alt_km)

        wind = layer['wind_speed']
        wind_bias = np.array([0.15, 0.05, -0.02]) * (wind / 20.0)

        # Fraction of deterministic turbulence to cancel
        corr_frac = p['correction_strength'] * p['reliability']

        correction = -(turb_dir * corr_frac
                       + wind_bias * p['drift_scale'])
        return correction

    # ---- serialisation helpers ---------------------------------------
    def get_altitude_turbulence_profile(self):
        if not self.turb_by_alt:
            self.update()
        return [dict(altitude=a, turbulence=self.turb_by_alt[a])
                for a in sorted(self.turb_by_alt, reverse=True)]

    def get_model_info(self):
        p = self.predictor
        total_params = (
            p.Wg1.size + p.bg1.size + p.Wg2.size + p.bg2.size
            + sum(w.size for w in p.We1) + sum(b.size for b in p.be1)
            + sum(w.size for w in p.We2) + sum(b.size for b in p.be2)
            + sum(w.size for w in p.Wf) + sum(b.size for b in p.bf)
        )
        return dict(
            name='PSTNet (Physically-Structured Turbulence Net)',
            architecture='4-regime gated MoE + backbone + spectral constraint',
            total_params=int(total_params),
            trained=p.trained,
            final_loss=(p.loss_history[-1]
                        if p.loss_history else None),
        )