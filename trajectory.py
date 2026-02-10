"""trajectory.py  —  6-DoF Missile Trajectory with PSTNet ML Correction

Unit conventions
----------------
    Position      : km   (x, y, z)
    Velocity      : m/s  (vx, vy, vz)
    Acceleration  : m/s²
    Time          : s

Position update:  pos_km += vel_ms · dt / 1000
"""

import math
import numpy as np
from config import (MISSILES, SPEED_OF_SOUND_SEA, AIR_DENSITY_SEA,
                    GRAVITY, EARTH_RADIUS)

# Turbulence perturbation scale: intensity × GRAVITY × PERT_K
PERT_K = 5.0


class MissileTrajectory:
    """Simulate one cruise-missile trajectory with optional ML correction."""

    # ── constructor ───────────────────────────────────────────────
    def __init__(self, missile_type, turb_field, use_correction=True,
                 pert_seed=42):
        cfg = MISSILES[missile_type]
        self.missile_type   = missile_type
        self.name           = cfg['name']
        self.mach           = cfg['speed']
        self.speed          = cfg['speed'] * SPEED_OF_SOUND_SEA * 1000.0  # m/s
        self.drag_coeff     = cfg['drag_coeff']
        self.mass           = cfg['mass']
        self.ref_area       = cfg['ref_area']
        self.max_g          = cfg['max_g_turn']
        self.cruise_alt     = cfg['cruise_altitude']   # km
        self.dive_angle     = cfg['dive_angle']         # degrees
        self.guidance       = cfg['guidance']
        self.turb_field     = turb_field
        self.use_correction = use_correction
        self.rng            = np.random.RandomState(pert_seed)

        # state
        self.pos       = None          # km
        self.vel       = None          # m/s
        self.target    = None          # km  (2-D ground point)
        self.phase     = None          # 'cruise' | 'terminal'
        self.time      = 0.0
        self.launched  = False

        # logging  (compatible with main.py / tests.py)
        self.trajectory      = []
        self.cross_track_log = []
        self.altitude_log    = []
        self.turbulence_log  = []
        self.correction_log  = []
        self.position        = None
        self.impact_error    = None
        self._launch_2d      = None

        # descent planning  (set in launch())
        self.total_range  = 0.0
        self.effective_dr = 0.0

    # ── launch ────────────────────────────────────────────────────
    def launch(self, start_xy, target_xy, launch_alt):
        sx, sy = start_xy
        tx, ty = target_xy
        self.target     = np.array([tx, ty], dtype=float)
        self.pos        = np.array([sx, sy, launch_alt], dtype=float)
        self.cruise_alt = launch_alt

        dx, dy = tx - sx, ty - sy
        rng = math.hypot(dx, dy)                       # km
        ux, uy = dx / max(rng, 1e-6), dy / max(rng, 1e-6)

        self.vel = np.array([ux * self.speed,
                             uy * self.speed,
                             0.0], dtype=float)         # m/s, level
        self.phase      = 'cruise'
        self.time       = 0.0
        self.launched   = True
        self._launch_2d = np.array([sx, sy], dtype=float)
        self.total_range = rng

        # ---- effective dive range with kinematic buffer ----------
        dr_geom   = self.cruise_alt / max(
            math.tan(math.radians(self.dive_angle)), 0.1)
        max_turn  = self.max_g * GRAVITY / (self.speed + 1e-6)   # rad/s
        req_angle = math.atan2(self.cruise_alt, dr_geom)          # rad
        turn_time = req_angle / (max_turn + 1e-6)                 # s
        buffer_km = self.speed * turn_time / 1000.0               # km
        self.effective_dr = min(dr_geom + buffer_km * 2.0,
                                rng * 0.90)

        # clear logs
        self.trajectory      = []
        self.cross_track_log = []
        self.altitude_log    = []
        self.turbulence_log  = []
        self.correction_log  = []
        self.position        = self.pos.copy()
        self.impact_error    = None

    # ── helpers ───────────────────────────────────────────────────
    @staticmethod
    def _air_density(alt_km):
        return AIR_DENSITY_SEA * math.exp(-max(alt_km, 0) / 8.5)

    def _compute_cross_track(self, pos2d):
        """Signed cross-track error in metres."""
        line = self.target - self._launch_2d
        ll = np.linalg.norm(line)
        if ll < 1e-10:
            return 0.0
        lu = line / ll
        perp = np.array([-lu[1], lu[0]])
        return float(np.dot(pos2d - self._launch_2d, perp) * 1000.0)

    # ── correction confidence gate ────────────────────────────────
    def _correction_confidence(self, turb_intensity, alt_km):
        # turbulence factor  (lower confidence in heavy turbulence)
        tf = 1.0 / (1.0 + math.exp((turb_intensity - 0.35) / 0.12))
        tf = max(tf, 0.10)
        # altitude factor
        if alt_km < 0.3:
            af = 0.20
        elif alt_km < 3.0:
            af = 0.20 + 0.80 * (alt_km - 0.3) / 2.7
        else:
            af = 1.0
        # speed factor
        m = self.mach
        if m < 0.6:
            sf = 0.35
        elif m < 1.5:
            sf = 0.35 + 0.65 * (m - 0.6) / 0.9
        else:
            sf = 1.0
        return tf * af * sf

    # ── main integration step ─────────────────────────────────────
    def step(self, dt):
        if not self.launched:
            return None

        pos, vel = self.pos.copy(), self.vel.copy()
        alt_km = max(pos[2], 0.0)
        spd    = np.linalg.norm(vel) + 1e-6                # m/s
        uvel   = vel / spd

        # ---- drag ------------------------------------------------
        rho    = self._air_density(alt_km)
        drag_N = 0.5 * rho * spd**2 * self.drag_coeff * self.ref_area
        a_drag = -(drag_N / self.mass) * uvel

        # ---- gravity ---------------------------------------------
        a_grav = np.array([0.0, 0.0, -GRAVITY])

        # ---- range & phase check ---------------------------------
        tgt3d   = np.array([self.target[0], self.target[1], 0.0])
        r_vec   = tgt3d - pos                               # km
        r_horiz = math.hypot(r_vec[0], r_vec[1])            # km
        v_horiz = math.hypot(vel[0], vel[1]) + 1e-6         # m/s

        if self.phase == 'cruise' and r_horiz < self.effective_dr + 0.5:
            self.phase = 'terminal'

        # ---- altitude controller ---------------------------------
        a_alt = np.zeros(3)
        if self.phase == 'cruise':
            # PD + gravity feed-forward  ── hold cruise_alt
            alt_err_m = (self.cruise_alt - pos[2]) * 1000.0   # m
            az = GRAVITY + 1.0 * alt_err_m - 3.0 * vel[2]
            az = np.clip(az,
                         -self.max_g * GRAVITY * 0.3,
                          self.max_g * GRAVITY * 0.5)
            a_alt[2] = az
        else:
            # t-go descent-rate controller
            t_go       = max(r_horiz * 1000.0 / v_horiz, 0.3)
            desired_vz = -(pos[2] * 1000.0) / t_go          # m/s
            vz_err     = vel[2] - desired_vz
            az = GRAVITY - 3.0 * vz_err
            az = np.clip(az,
                         -self.max_g * GRAVITY * 0.8,
                          self.max_g * GRAVITY * 0.8)
            a_alt[2] = az

        # ---- turbulence ------------------------------------------
        turb_vec, intensity = self.turb_field.sample(
            pos[0], pos[1], alt_km)
        pert_scale = intensity * GRAVITY * PERT_K
        a_turb  = turb_vec * pert_scale
        a_turb += self.rng.randn(3) * pert_scale * 0.25    # jitter

        # ---- ML correction ---------------------------------------
        a_corr   = np.zeros(3)
        corr_mag = 0.0
        if self.use_correction:
            raw  = self.turb_field.get_correction(
                pos[0], pos[1], alt_km,
                vel[0], vel[1], vel[2])
            conf = self._correction_confidence(intensity, alt_km)
            correction = np.array(raw) * pert_scale * conf

            # cap at 60 % of instantaneous turbulence or 0.5 g
            turb_mag = np.linalg.norm(a_turb) + 1e-12
            cm       = np.linalg.norm(correction) + 1e-12
            cap      = min(turb_mag * 0.6, 0.5 * GRAVITY)
            if cm > cap:
                correction *= cap / cm

            a_corr   = correction
            corr_mag = float(np.linalg.norm(a_corr))

        # ---- horizontal proportional navigation ------------------
        a_guid = self._guidance_horiz(r_vec, r_horiz, spd, uvel)

        # ---- Euler integration -----------------------------------
        a_total = a_drag + a_grav + a_alt + a_turb + a_corr + a_guid
        vel_new = vel + a_total * dt                        # m/s

        # implicit thrust: clamp speed to ±15 % of cruise
        spd_new = np.linalg.norm(vel_new) + 1e-6
        spd_clm = np.clip(spd_new,
                          self.speed * 0.85,
                          self.speed * 1.15)
        vel_new = vel_new / spd_new * spd_clm

        pos_new = pos + vel_new * dt / 1000.0               # km

        # ---- update state ----------------------------------------
        self.vel  = vel_new
        self.pos  = pos_new
        self.time += dt

        # ---- logging ---------------------------------------------
        self.position = pos_new.copy()
        self.trajectory.append(
            [float(pos_new[0]), float(pos_new[1]), float(pos_new[2])])
        self.altitude_log.append(float(pos_new[2]))
        self.turbulence_log.append(float(intensity))
        self.correction_log.append(corr_mag)
        self.cross_track_log.append(
            self._compute_cross_track(pos_new[:2]))

        # ---- impact detection with interpolation -----------------
        if pos_new[2] <= 0.0:
            # interpolate to exact ground-plane crossing
            if pos[2] > 0.0:
                frac = pos[2] / max(pos[2] - pos_new[2], 1e-12)
                frac = min(frac, 1.0)
                impact_pos = pos + (pos_new - pos) * frac
            else:
                impact_pos = pos_new

            miss_m = float(
                np.linalg.norm(impact_pos[:2] - self.target) * 1000.0)
            self.impact_error = miss_m

            avg_t = (float(np.mean(self.turbulence_log))
                     if self.turbulence_log else 0.0)
            max_ct = (float(np.max(np.abs(self.cross_track_log)))
                      if self.cross_track_log else 0.0)
            return dict(
                status='impact',
                miss_distance_m=miss_m,
                time=self.time,
                avg_turbulence=avg_t,
                max_cross_track_m=max_ct,
                corrected=self.use_correction,
            )
        return None

    # ── horizontal proportional navigation ────────────────────────
    def _guidance_horiz(self, r_vec, r_horiz, spd, uvel):
        """Horizontal-only PN  (vertical handled by altitude controller)."""
        hd = np.array([r_vec[0], r_vec[1], 0.0])
        hd /= (np.linalg.norm(hd) + 1e-9)

        uh = np.array([uvel[0], uvel[1], 0.0])
        uhn = np.linalg.norm(uh)
        if uhn > 1e-9:
            uh /= uhn

        gain = 4.0 if r_horiz < 5.0 else 2.0
        a_cmd = (hd - uh) * spd * gain
        a_cmd[2] = 0.0                                  # no vertical

        am = np.linalg.norm(a_cmd) + 1e-9
        # g-budget share: 50 % cruise, 30 % terminal (rest for descent)
        share = 0.5 if self.phase == 'cruise' else 0.3
        mx = self.max_g * GRAVITY * share
        if am > mx:
            a_cmd = a_cmd / am * mx
        return a_cmd

    # ── convenience runner ────────────────────────────────────────
    def run(self, dt=None, max_time=3000):
        if dt is None:
            dt = min(0.5, 120.0 / self.speed)
        result = None
        while self.time < max_time:
            r = self.step(dt)
            if r is not None:
                result = r
                break
        if result is None:
            result = dict(
                status='timeout', miss_distance_m=1e6,
                time=self.time, avg_turbulence=0,
                max_cross_track_m=0, corrected=self.use_correction)
        history = dict(
            x=[p[0] for p in self.trajectory],
            y=[p[1] for p in self.trajectory],
            z=[p[2] for p in self.trajectory],
        )
        return result, history