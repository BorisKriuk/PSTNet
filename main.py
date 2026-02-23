#!/usr/bin/env python3
"""
main.py - PSTNet Unified Application
Physically-Structured Turbulence Network - Kriuk et al.

Combines:
  - 3D Missile Trajectory Simulation (Three.js)
  - Global Turbulence Globe (Pure Canvas 2D)

Port 80 - CORS enabled
"""

import math, time, os
import numpy as np
from flask import Flask, render_template_string, jsonify, request, make_response

from config import MISSILES, SCENARIOS, SPEED_OF_SOUND_SEA, ALTITUDE_LAYERS, AIR_DENSITY_SEA
from weather_api import WeatherService, FALLBACK_WEATHER
from turbulence_model import TurbulenceField, TurbulencePredictor
from trajectory import MissileTrajectory

app = Flask(__name__)


# ===================================================================
# CORS
# ===================================================================
@app.after_request
def add_cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return resp

@app.before_request
def preflight():
    if request.method == 'OPTIONS':
        r = make_response()
        r.headers['Access-Control-Allow-Origin'] = '*'
        r.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        return r


# ===================================================================
# SHARED STATE
# ===================================================================
weather_service = WeatherService(lat=35.0, lon=-120.0)
turb_field = TurbulenceField(weather_service)

sim = dict(
    missile_type='SUPERSONIC', launch_alt=10.0, target_range=120.0,
    launch_pos=[30.0, 100.0], target_pos=[150.0, 100.0],
    corrected=None, uncorrected=None, cr=None, ur=None,
    time=0.0, running=False,
)

def setup_sim(mt='SUPERSONIC', alt=10.0, rng=120.0):
    seed = np.random.randint(0, 2**31)
    sim.update(
        missile_type=mt, launch_alt=alt, target_range=rng,
        launch_pos=[30.0, 100.0], target_pos=[30.0+rng, 100.0],
        corrected=MissileTrajectory(mt, turb_field, True, seed),
        uncorrected=MissileTrajectory(mt, turb_field, False, seed),
        cr=None, ur=None, time=0.0, running=False,
    )

FLIGHT_LEVELS = [
    dict(name='FL100',alt_km=3.05,alt_ft=10000),
    dict(name='FL180',alt_km=5.49,alt_ft=18000),
    dict(name='FL240',alt_km=7.32,alt_ft=24000),
    dict(name='FL300',alt_km=9.14,alt_ft=30000),
    dict(name='FL340',alt_km=10.36,alt_ft=34000),
    dict(name='FL380',alt_km=11.58,alt_ft=38000),
    dict(name='FL410',alt_km=12.50,alt_ft=41000),
    dict(name='FL450',alt_km=13.72,alt_ft=45000),
]
GRID_N = 24
PRESETS = [
    dict(name='Central Asia \u2014 Himalaya',lat_min=18,lat_max=46,lon_min=58,lon_max=102),
    dict(name='North Atlantic Tracks',lat_min=40,lat_max=65,lon_min=-60,lon_max=-10),
    dict(name='North American Rockies',lat_min=30,lat_max=52,lon_min=-125,lon_max=-95),
    dict(name='European Alps',lat_min=40,lat_max=52,lon_min=-2,lon_max=22),
    dict(name='Andes Corridor',lat_min=-35,lat_max=10,lon_min=-80,lon_max=-60),
    dict(name='East Asian Jet Stream',lat_min=20,lat_max=45,lon_min=110,lon_max=150),
    dict(name='Horn of Africa',lat_min=-2,lat_max=18,lon_min=35,lon_max=55),
    dict(name='Southern Ocean',lat_min=-58,lat_max=-38,lon_min=20,lon_max=80),
]


class TurbulenceComputer:
    def __init__(self):
        self.predictor = TurbulencePredictor()
        self.trained = False

    @staticmethod
    def terrain_km(lat, lon):
        h = 0.0
        h = max(h,7.0*math.exp(-((lat-28.5)**2/1.2+(lon-86)**2/45)))
        h = max(h,4.8*math.exp(-((lat-33)**2/35+(lon-90)**2/120)))
        h = max(h,6.0*math.exp(-((lat-36)**2/2+(lon-76.5)**2/8)))
        h = max(h,4.5*math.exp(-((lat-35.5)**2/3+(lon-70)**2/15)))
        h = max(h,5.0*math.exp(-((lat-38.5)**2/2+(lon-73)**2/6)))
        h = max(h,4.0*math.exp(-((lat-42)**2/3+(lon-80)**2/60)))
        h = max(h,5.5*math.exp(-((lat-36)**2/1.5+(lon-82)**2/60)))
        h = max(h,3.5*math.exp(-((lat-40)**2/60+(lon+107)**2/8)))
        h = max(h,3.0*math.exp(-((lat-38)**2/40+(lon+120)**2/3)))
        h = max(h,5.0*math.exp(-((lat+15)**2/300+(lon+69)**2/4)))
        h = max(h,3.0*math.exp(-((lat+43)**2/40+(lon+71)**2/3)))
        h = max(h,2.5*math.exp(-((lat-24)**2/20+(lon+105)**2/5)))
        h = max(h,3.8*math.exp(-((lat-46.5)**2/1.5+(lon-10)**2/25)))
        h = max(h,4.0*math.exp(-((lat-42.5)**2/0.8+(lon-44)**2/12)))
        h = max(h,2.0*math.exp(-((lat-64)**2/30+(lon-14)**2/4)))
        h = max(h,2.5*math.exp(-((lat-42.5)**2/0.5+(lon-0)**2/5)))
        h = max(h,2.8*math.exp(-((lat-33)**2/2+(lon+2)**2/15)))
        h = max(h,3.0*math.exp(-((lat-9)**2/5+(lon-39)**2/5)))
        h = max(h,3.5*math.exp(-((lat+3)**2/0.5+(lon-37)**2/0.5)))
        h = max(h,2.5*math.exp(-((lat-36)**2/3+(lon-138)**2/2)))
        h = max(h,2.5*math.exp(-((lat+44)**2/2+(lon-170)**2/0.5)))
        h = max(h,1.5*math.exp(-((lat-57)**2/100+(lon-60)**2/2)))
        h = max(h,3.0*math.exp(-((lat-50)**2/2+(lon-87)**2/5)))
        return min(h, 8.5)

    def atm_state(self, lat, lon, alt_km, tkm, weather):
        w = weather
        T_sl = w['temperature_2m']-(lat-32.0)*0.3
        if alt_km < 11: temp = T_sl-6.5*alt_km
        elif alt_km < 20: temp = T_sl-6.5*11
        else: temp = T_sl-6.5*11+1.0*(alt_km-20)
        temp = max(temp,180.0)
        pres = max(w['pressure_surface']*math.exp(-alt_km/8.5),0.001)
        density = max((pres*1000)/(287.05*temp),1e-5)
        sw = w['wind_speed_10m']*(1+0.4*abs(lat-25)/20)
        if tkm > 1 and alt_km < tkm+5: sw *= (1+0.15*tkm/5)
        if alt_km < 1: wind = sw*(1+0.3*alt_km)
        elif alt_km < 10: wind = sw*1.3+(alt_km-1)*1.2
        elif alt_km < 15: wind = sw*1.3+9*1.2+6*max(0,1-abs(alt_km-12)/3)
        else: wind = max(sw*1.3+9*1.2-(alt_km-15)*0.6,1.0)
        wind = max(min(wind,80),0.5)
        ea = max(alt_km-tkm*0.3,0.1)
        dT = -6.5 if alt_km < 11 else (0.0 if alt_km < 20 else 1.0)
        ws = max(wind/max(ea,0.1),0.1)
        ri = max(min((9.81/temp)*(dT+9.8)/(ws**2+0.01),10),-5)
        if tkm > 1 and alt_km < tkm+5: ri *= max(0.3,1-0.4*tkm/5)
        return dict(wind_speed=wind,temperature=temp,density=density,richardson=ri,pressure=pres)

    @staticmethod
    def mountain_wave(alt_km, wind, ri, tkm, tgrad):
        if tkm < 0.5: return 0.0
        above = alt_km-tkm
        if above < -2: hf = 0.1
        elif above < 0: hf = 0.3+0.35*(above+2)/2
        elif above < 3: hf = 0.65+0.35*(1-above/3)
        elif above < 8: hf = 0.65*math.exp(-(above-3)/4)
        else: hf = 0.08*math.exp(-(above-8)/10)
        wf = min(wind/15,1.5); gf = min(tgrad/2.0,1.5)
        tf = min(tkm/4.0,1.5)
        sf = 1.0 if 0.1 < ri < 2.0 else (0.5 if ri > 2.0 else 0.7)
        return min(0.18*hf*wf*gf*tf*sf,0.40)

    def train(self, weather):
        samples = [(20,70),(20,90),(28,86),(33,90),(38,73),(42,80),(44,70),(44,95)]
        profiles = []
        for slat,slon in samples:
            tkm = self.terrain_km(slat,slon)
            p = []
            for alt in ALTITUDE_LAYERS:
                s = self.atm_state(slat,slon,alt,tkm,weather)
                s['altitude'] = alt; p.append(s)
            profiles.append(p)
        self.predictor.fit(profiles, epochs=200, lr=0.004)
        self.trained = True
        print('  PSTNet trained OK')

    def compute_region(self, lat_min, lat_max, lon_min, lon_max, weather):
        t0 = time.time()
        if not self.trained: self.train(weather)
        lats = np.linspace(lat_min,lat_max,GRID_N)
        lons = np.linspace(lon_min,lon_max,GRID_N)
        terrain = np.zeros((GRID_N,GRID_N))
        for i in range(GRID_N):
            for j in range(GRID_N):
                terrain[i,j] = self.terrain_km(lats[i],lons[j])
        dlat = lats[1]-lats[0] if GRID_N > 1 else 1.0
        dlon = lons[1]-lons[0] if GRID_N > 1 else 1.0
        gy,gx = np.gradient(terrain,dlat,dlon)
        tgrad = np.sqrt(gx**2+gy**2)
        levels = []
        for fl in FLIGHT_LEVELS:
            ak = fl['alt_km']
            tg = np.zeros((GRID_N,GRID_N))
            wg = np.zeros((GRID_N,GRID_N))
            rg = np.full((GRID_N,GRID_N),-1,dtype=int)
            for i in range(GRID_N):
                for j in range(GRID_N):
                    tkm = terrain[i,j]
                    if ak < tkm-0.5: tg[i,j] = -1; continue
                    s = self.atm_state(lats[i],lons[j],ak,tkm,weather)
                    ea = max(ak-tkm*0.3,0.1)
                    base = self.predictor.physics_turbulence(ea,s['wind_speed'],s['richardson'],s['density'])
                    mw = self.mountain_wave(ak,s['wind_speed'],s['richardson'],tkm,tgrad[i,j])
                    jet = 0
                    if 9 < ak < 14:
                        jet = 0.08*(s['wind_speed']/30)*math.exp(-((ak-11.5)**2)/3)
                    tg[i,j] = min(base+mw+jet,0.80)
                    wg[i,j] = s['wind_speed']
                    if self.trained:
                        rw = self.predictor.get_regime_weights(s['wind_speed'],s['temperature'],s['density'],s['richardson'],ak,s['pressure'])
                        ws_list = [rw.get('convective',.25),rw.get('neutral',.25),rw.get('stable',.25),rw.get('stratospheric',.25)]
                        rg[i,j] = int(np.argmax(ws_list))
            v = tg[tg >= 0]
            rc = {k:0 for k in range(4)}
            for rv in rg.flat:
                if rv >= 0: rc[rv] += 1
            st = dict(mean=round(float(v.mean()),3) if len(v) else 0,
                      max_val=round(float(v.max()),3) if len(v) else 0,
                      moderate=int(np.sum(v>0.20)),severe=int(np.sum(v>0.40)),
                      extreme=int(np.sum(v>0.60)),blocked=int(np.sum(tg<0)),
                      regime_conv=rc[0],regime_neut=rc[1],regime_stab=rc[2],regime_strt=rc[3])
            levels.append(dict(name=fl['name'],alt_km=fl['alt_km'],alt_ft=fl['alt_ft'],
                               turb=np.round(tg,3).tolist(),wind=np.round(wg,1).tolist(),
                               regime=rg.tolist(),stats=st))
        ct = time.time()-t0
        return dict(lats=np.round(lats,2).tolist(),lons=np.round(lons,2).tolist(),
                    terrain=np.round(terrain,2).tolist(),grid_n=GRID_N,
                    flight_levels=levels,compute_time=round(ct,1),
                    model_loss=round(self.predictor.loss_history[-1],6) if self.predictor.loss_history else None)


globe_computer = TurbulenceComputer()


# ===================================================================
# DASHBOARD HTML
# ===================================================================
DASH_HTML = r'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSTNet - Kriuk et al.</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;background:#000308;color:#8899aa;
 font-family:'Courier New',monospace;overflow-x:hidden;overflow-y:auto}
#bgCv{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none}
.wrap{min-height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
 padding:40px 20px;position:relative;z-index:1}
.logo{text-align:center;margin-bottom:40px}
.logo h1{color:#00dd88;font-size:36px;letter-spacing:8px;text-transform:uppercase;margin-bottom:4px;
 text-shadow:0 0 30px rgba(0,221,136,0.15)}
.logo .sub{color:#667788;font-size:13px;letter-spacing:3px;margin-bottom:14px}
.logo .auth{font-size:16px;margin-top:12px;letter-spacing:2px;font-weight:bold;
 background:linear-gradient(135deg,#00ccff,#00dd88);-webkit-background-clip:text;
 -webkit-text-fill-color:transparent;background-clip:text}
.divider{width:60px;height:1px;background:linear-gradient(90deg,transparent,#1a3540,transparent);margin:18px auto}
.desc{max-width:560px;margin:0 auto;text-align:center}
.desc h2{color:#aabbcc;font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px}
.desc p{color:#556677;font-size:10px;line-height:1.75;margin-bottom:8px}
.desc .hl{color:#00aa77}
.pills{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin-top:14px;margin-bottom:6px}
.pill{padding:3px 10px;border:1px solid #152030;border-radius:12px;font-size:8px;color:#557788;
 letter-spacing:0.5px}
.pill.a{border-color:#0a3530;color:#00aa77}
.cards{display:flex;gap:28px;flex-wrap:wrap;justify-content:center;margin-top:32px}
.card{display:flex;flex-direction:column;align-items:center;justify-content:center;
 width:230px;height:230px;border:1px solid #1a2540;border-radius:8px;
 background:rgba(4,8,16,.9);cursor:pointer;text-decoration:none;
 transition:all .25s;position:relative;overflow:hidden}
.card:hover{border-color:#00dd88;background:rgba(4,16,12,.95);transform:translateY(-3px);
 box-shadow:0 8px 30px rgba(0,221,136,.06)}
.card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
 background:linear-gradient(90deg,transparent,#00dd88,transparent);opacity:0;transition:opacity .25s}
.card:hover::after{opacity:.6}
.card svg{color:#00dd88;margin-bottom:16px;opacity:.7;transition:opacity .25s}
.card:hover svg{opacity:1}
.card h2{color:#aabbcc;font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px}
.card p{color:#556677;font-size:9px;text-align:center;padding:0 16px;line-height:1.5}
.foot{text-align:center;font-size:9px;color:#223344;margin-top:40px;padding-bottom:20px}
.foot span{color:#1a2540;margin:0 6px}
.ver{margin-top:4px;color:#1a2540;font-size:8px}
@media(max-width:600px){
 .logo h1{font-size:24px;letter-spacing:4px}
 .cards{flex-direction:column;align-items:center}
 .card{width:200px;height:200px}
 .desc p{font-size:9px}
}
</style></head><body>

<canvas id="bgCv"></canvas>

<div class="wrap">
 <div class="logo">
  <h1>PSTNet</h1>
  <div class="sub">Physically-Structured Turbulence Network</div>
  <div class="auth">Kriuk et al.</div>
 </div>

 <div class="divider"></div>

 <div class="desc">
  <h2>Architecture</h2>
  <p>A novel lightweight neural network where atmospheric physics is embedded directly into the
  <span class="hl">network structure</span>, not merely the loss function.
  Four stability-regime expert sub-networks are dynamically gated by a learned
  <span class="hl">Monin-Obukhov classifier</span>, while an analytical backbone provides
  zero-parameter TKE estimation that the neural residual refines.</p>
  <p>Output turbulence corrections are derived via
  <span class="hl">Kolmogorov spectral constraint</span>
  defining physically consistent energy cascade scaling across all atmospheric regimes
  from convective boundary layer through stratospheric cruise.</p>
  <div class="pills">
   <div class="pill a">4-Regime Gated MoE</div>
   <div class="pill a">FiLM Density Conditioning</div>
   <div class="pill a">Kolmogorov Spectral Output</div>
   <div class="pill">NASA POWER Weather</div>
  </div>
 </div>

 <div class="cards">
  <a href="/trajectory" class="card">
   <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>
   <h2>Trajectory</h2>
   <p>6-DoF missile flight simulation with real-time ML turbulence correction</p>
  </a>
  <a href="/globe" class="card">
   <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
   <h2>Turbulence Globe</h2>
   <p>Global atmospheric turbulence analysis across 8 flight levels</p>
  </a>
 </div>

 <div class="foot">
  PSTNet v1.0<span>|</span>NASA POWER Weather<span>|</span>Zero Framework Dependencies
  <div class="ver">Validated: Supersonic M2.8 &middot; High Supersonic M4.5 &middot; Hypersonic M8.0</div>
 </div>
</div>

<script>
(function(){
var cv=document.getElementById('bgCv'),c=cv.getContext('2d');
var W,H;

var EQS=[
 'E(k) = C\u2096 \u03B5\u00B2\u2033\u00B3 k\u207B\u2075\u2033\u00B3',
 '\u2202u/\u2202t + u\u00B7\u2207u = -\u2207p/\u03C1 + \u03BD\u2207\u00B2u',
 'Ri = (g/\u03B8)(\u2202\u03B8/\u2202z) / (\u2202u/\u2202z)\u00B2',
 'TKE = \u00BD(u\u2032\u00B2 + v\u2032\u00B2 + w\u2032\u00B2)',
 'L = -u*\u00B3\u03B8 / (\u03BAg w\u2032\u03B8\u2032)',
 '\u03C3\u00B2w = 1.7 \u03B5\u00B2\u2033\u00B3 z\u00B2\u2033\u00B3',
 '\u03C6m(\u03B6) = (1 - 16\u03B6)\u207B\u00B9\u2033\u2074',
 '\u03B5 = Ce TKE\u00B3\u2033\u00B2 / \u2113',
 'y\u0302 = \u03A3 gi(x) \u00B7 fi(x)',
 'S(k) \u221D k\u207B\u2075\u2033\u00B3 exp(-k/k\u03B7)',
 'Re = TKE\u00B2 / (\u03BD\u03B5)',
 'Km = \u2113\u00B2 |\u2202U/\u2202z| Fm(Ri)',
 'N\u00B2 = (g/\u03B80)(\u2202\u03B8\u0305/\u2202z)',
 'FiLM: h \u2192 \u03B3 \u2299 h + \u03B2',
 '\u2207 \u00B7 u = 0',
 'CD = f(Re, Ma, \u03B1)',
 '\u222B E(k) dk = TKE',
 'Fr = U / (N H)',
 '\u2202TKE/\u2202t = P - \u03B5 + T + D',
 'P = -u\u2032w\u2032 \u2202U/\u2202z',
 '\u03C4 = \u03C1 Cd |u| u',
 'dv/dt = F/m - g + Fturb',
 '\u03B1eff = \u03B1 + \u0394\u03B1turb(t)',
 'k\u03B7 = (\u03B5/\u03BD\u00B3)\u00BC'
];

var eqs=[],graphs=[],particles=[],lines=[];

function resize(){W=cv.width=innerWidth;H=cv.height=innerHeight;}

function init(){
 resize();
 eqs=[];graphs=[];particles=[];lines=[];

 for(var i=0;i<28;i++){
  eqs.push({
   text:EQS[i%EQS.length],
   x:Math.random()*W,
   y:Math.random()*H,
   vx:(Math.random()-0.5)*0.25,
   vy:-0.1-Math.random()*0.2,
   size:10+Math.random()*7,
   alpha:0.18+Math.random()*0.14,
   phase:Math.random()*6.283
  });
 }

 var gx=[W*0.07,W*0.93,W*0.05,W*0.95,W*0.14,W*0.86,W*0.5,W*0.3,W*0.7,W*0.12,W*0.88];
 var gy=[H*0.08,H*0.12,H*0.52,H*0.58,H*0.88,H*0.85,H*0.06,H*0.92,H*0.48,H*0.35,H*0.35];
 for(var i=0;i<11;i++){
  graphs.push({
   x:gx[i],y:gy[i],
   vx:(Math.random()-0.5)*0.1,
   vy:(Math.random()-0.5)*0.1,
   size:45+Math.random()*40,
   type:i%6,
   alpha:0.18+Math.random()*0.12,
   phase:Math.random()*6.283
  });
 }

 for(var i=0;i<65;i++){
  particles.push({
   x:Math.random()*W,y:Math.random()*H,
   vx:(Math.random()-0.5)*0.3,
   vy:(Math.random()-0.5)*0.3,
   r:1+Math.random()*2,
   alpha:0.12+Math.random()*0.14,
   phase:Math.random()*6.283
  });
 }

 for(var i=0;i<7;i++){
  lines.push({
   y:H*0.08+Math.random()*H*0.84,
   speed:0.1+Math.random()*0.15,
   alpha:0.08+Math.random()*0.07,
   dir:Math.random()>0.5?1:-1,
   offset:Math.random()*W
  });
 }
}

function frame(ts){
 requestAnimationFrame(frame);
 var t=ts*0.001;
 c.clearRect(0,0,W,H);

 /* ---- grid ---- */
 c.strokeStyle='rgba(0,221,136,0.06)';
 c.lineWidth=0.5;
 var gs=60;
 for(var x=gs;x<W;x+=gs){c.beginPath();c.moveTo(x,0);c.lineTo(x,H);c.stroke();}
 for(var y=gs;y<H;y+=gs){c.beginPath();c.moveTo(0,y);c.lineTo(W,y);c.stroke();}
 c.fillStyle='rgba(0,221,136,0.12)';
 for(var x=gs;x<W;x+=gs){for(var y=gs;y<H;y+=gs){
  c.beginPath();c.arc(x,y,1,0,6.283);c.fill();}}

 /* ---- horizontal scan lines ---- */
 for(var i=0;i<lines.length;i++){
  var ln=lines[i];
  var lx=(ln.offset+t*60*ln.dir*ln.speed)%W;
  if(lx<0)lx+=W;
  var grad=c.createLinearGradient(lx-250,0,lx+250,0);
  grad.addColorStop(0,'rgba(0,221,136,0)');
  grad.addColorStop(0.5,'rgba(0,221,136,'+ln.alpha+')');
  grad.addColorStop(1,'rgba(0,221,136,0)');
  c.strokeStyle=grad;c.lineWidth=0.8;c.globalAlpha=1;
  c.beginPath();c.moveTo(lx-250,ln.y);c.lineTo(lx+250,ln.y);c.stroke();
 }

 /* ---- floating equations ---- */
 for(var i=0;i<eqs.length;i++){
  var e=eqs[i];
  e.x+=e.vx;e.y+=e.vy;
  if(e.y<-30){e.y=H+20+Math.random()*60;e.x=Math.random()*W;}
  if(e.y>H+50){e.y=-20;e.x=Math.random()*W;}
  if(e.x<-300)e.x=W+50;
  if(e.x>W+300)e.x=-250;
  var a=e.alpha*(0.6+0.4*Math.sin(t*0.4+e.phase));
  c.globalAlpha=Math.max(0,a);
  c.font=e.size+'px "Courier New",monospace';
  c.fillStyle='#00dd88';
  c.fillText(e.text,e.x,e.y);
 }

 /* ---- mini graphs ---- */
 for(var i=0;i<graphs.length;i++){
  var g=graphs[i];
  g.x+=g.vx;g.y+=g.vy;
  if(g.x<20||g.x>W-20)g.vx*=-1;
  if(g.y<20||g.y>H-20)g.vy*=-1;
  var a=g.alpha*(0.6+0.4*Math.sin(t*0.32+g.phase));
  c.globalAlpha=Math.max(0,a);
  c.save();c.translate(g.x,g.y);
  var s=g.size;

  c.strokeStyle='#00bb77';c.lineWidth=1.2;
  c.beginPath();c.moveTo(0,0);c.lineTo(s*1.1,0);c.moveTo(0,0);c.lineTo(0,-s*1.1);c.stroke();
  c.fillStyle='#00bb77';
  c.beginPath();c.moveTo(s*1.1,0);c.lineTo(s*1.02,-3);c.lineTo(s*1.02,3);c.fill();
  c.beginPath();c.moveTo(0,-s*1.1);c.lineTo(-3,-s*1.02);c.lineTo(3,-s*1.02);c.fill();
  c.lineWidth=0.6;
  for(var ti=1;ti<=4;ti++){
   c.beginPath();c.moveTo(s*ti/4,3);c.lineTo(s*ti/4,-3);c.stroke();
   c.beginPath();c.moveTo(-3,-s*ti/4);c.lineTo(3,-s*ti/4);c.stroke();}

  c.lineWidth=1.6;c.beginPath();

  if(g.type===0){
   c.strokeStyle='#00ffaa';
   for(var j=1;j<50;j++){var k=j/50;var E2=0.65*Math.pow(k+0.02,-1.667)*0.08;E2=Math.min(E2,0.95);
    var x2=k*s,y2=-E2*s;j===1?c.moveTo(x2,y2):c.lineTo(x2,y2);}
   c.stroke();
   c.strokeStyle='rgba(0,200,100,0.5)';c.lineWidth=0.8;c.setLineDash([3,3]);c.beginPath();
   c.moveTo(s*0.12,-s*0.88);c.lineTo(s*0.92,-s*0.1);c.stroke();c.setLineDash([]);
   c.fillStyle='#00ffaa';c.font='8px monospace';
   c.fillText('k',s*0.98,14);c.fillText('E(k)',3,-s*0.95);
   c.fillText('-5/3',s*0.45,-s*0.5);

  }else if(g.type===1){
   c.strokeStyle='#00ccff';
   for(var j=0;j<40;j++){var z2=j/40;var u2=0.1+0.75*(1-Math.exp(-z2*3.5))+0.03*Math.sin(t*1.1+z2*10);
    j===0?c.moveTo(u2*s,-z2*s):c.lineTo(u2*s,-z2*s);}
   c.stroke();
   c.fillStyle='#00ccff';c.font='8px monospace';
   c.fillText('u(z)',s*0.65,14);c.fillText('z',3,-s*0.95);

  }else if(g.type===2){
   c.strokeStyle='#44ff88';
   for(var j=0;j<60;j++){var x2=j/60*s;
    var y2=-s*0.45+Math.sin(j*0.5+t*1.6)*s*0.16+Math.sin(j*1.8+t*0.9)*s*0.09
     +Math.sin(j*4.5+t*2.5)*s*0.04+(Math.random()-0.5)*s*0.02;
    j===0?c.moveTo(x2,y2):c.lineTo(x2,y2);}
   c.stroke();
   c.strokeStyle='rgba(0,200,100,0.4)';c.lineWidth=0.6;c.setLineDash([2,3]);
   c.beginPath();c.moveTo(0,-s*0.45);c.lineTo(s,-s*0.45);c.stroke();c.setLineDash([]);
   c.fillStyle='#44ff88';c.font='8px monospace';
   c.fillText('t',s*0.98,14);c.fillText("u'(t)",2,-s*0.88);

  }else if(g.type===3){
   c.strokeStyle='#ffaa44';
   for(var j=0;j<45;j++){var x2=j/45;
    var y2=1/(1+Math.exp(-(x2-0.5)*12+Math.sin(t*0.7)*1.8));
    j===0?c.moveTo(x2*s,-y2*s*0.88-s*0.04):c.lineTo(x2*s,-y2*s*0.88-s*0.04);}
   c.stroke();
   c.strokeStyle='#cc8844';c.lineWidth=1.2;c.beginPath();
   for(var j=0;j<45;j++){var x2=j/45;
    var y2=1/(1+Math.exp(-(x2-0.35)*10+Math.cos(t*0.5)*1.4));
    j===0?c.moveTo(x2*s,-y2*s*0.88-s*0.04):c.lineTo(x2*s,-y2*s*0.88-s*0.04);}
   c.stroke();
   c.fillStyle='#ffaa44';c.font='8px monospace';
   c.fillText('x',s*0.98,14);c.fillText('g(x)',2,-s*0.88);

  }else if(g.type===4){
   var layers=[3,5,4,2],pos=[];
   for(var li=0;li<layers.length;li++){pos.push([]);
    var lx2=s*(li+0.5)/layers.length;
    for(var ni=0;ni<layers[li];ni++){
     pos[li].push({x:lx2,y:-s*(ni+0.5)/Math.max(layers[li],1)});}}
   c.strokeStyle='rgba(0,200,150,0.5)';c.lineWidth=0.8;
   for(var li=0;li<layers.length-1;li++){
    for(var ai=0;ai<pos[li].length;ai++){
     for(var bi=0;bi<pos[li+1].length;bi++){
      c.beginPath();c.moveTo(pos[li][ai].x,pos[li][ai].y);
      c.lineTo(pos[li+1][bi].x,pos[li+1][bi].y);c.stroke();}}}
   c.fillStyle='#00ffaa';
   for(var li=0;li<layers.length;li++){
    for(var ni=0;ni<pos[li].length;ni++){
     var pr=3+1.2*Math.sin(t*1.8+li+ni*0.8);
     c.beginPath();c.arc(pos[li][ni].x,pos[li][ni].y,pr,0,6.283);c.fill();}}
   c.font='7px monospace';c.fillStyle='#00ddaa';
   c.fillText('in',pos[0][0].x-4,10);c.fillText('MoE',pos[3][0].x-8,10);

  }else{
   c.strokeStyle='#00aaff';
   for(var j=0;j<45;j++){var x2=j/45;
    var ri=x2*4-1;
    var turb=ri<0.25?0.8:0.8*Math.exp(-(ri-0.25)*3);
    turb+=0.02*Math.sin(t*1.3+j*0.8);
    turb=Math.max(0,Math.min(1,turb));
    j===0?c.moveTo(x2*s,-turb*s*0.85):c.lineTo(x2*s,-turb*s*0.85);}
   c.stroke();
   c.strokeStyle='rgba(255,80,80,0.55)';c.lineWidth=0.8;c.setLineDash([3,3]);
   c.beginPath();var rcx=0.3125*s;c.moveTo(rcx,0);c.lineTo(rcx,-s);c.stroke();c.setLineDash([]);
   c.fillStyle='#00aaff';c.font='8px monospace';
   c.fillText('Ri',s*0.95,14);c.fillText('turb',2,-s*0.88);
   c.fillStyle='rgba(255,80,80,0.7)';c.font='7px monospace';c.fillText('Ri_c',rcx-8,-s*0.92);
  }
  c.restore();
 }

 /* ---- particles ---- */
 for(var i=0;i<particles.length;i++){
  var p=particles[i];
  p.x+=p.vx;p.y+=p.vy;
  if(p.x<0||p.x>W)p.vx*=-1;
  if(p.y<0||p.y>H)p.vy*=-1;
  var a2=p.alpha*(0.65+0.35*Math.sin(t*0.6+p.phase));
  c.globalAlpha=Math.max(0,a2);
  c.fillStyle='#00dd88';
  c.beginPath();c.arc(p.x,p.y,p.r,0,6.283);c.fill();
 }

 /* ---- particle connections ---- */
 c.lineWidth=0.7;
 for(var i=0;i<particles.length;i++){
  for(var j=i+1;j<particles.length;j++){
   var dx=particles[i].x-particles[j].x,dy=particles[i].y-particles[j].y;
   var d=Math.sqrt(dx*dx+dy*dy);
   if(d<150){
    c.globalAlpha=0.12*(1-d/150);
    c.strokeStyle='#00dd88';
    c.beginPath();c.moveTo(particles[i].x,particles[i].y);
    c.lineTo(particles[j].x,particles[j].y);c.stroke();
   }
  }
 }

 c.globalAlpha=1;
}

init();
window.addEventListener('resize',function(){resize();init();});
requestAnimationFrame(frame);
})();
</script>
</body></html>'''

# ===================================================================
# TRAJECTORY HTML
# ===================================================================
TRAJ_HTML = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSTNet Trajectory - Kriuk et al.</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;overflow:hidden;font-family:'Courier New',monospace;color:#888}
#c3d{position:fixed;top:0;left:0;width:100vw;height:100vh}
.hdr{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(0,0,0,.88);
 border-bottom:1px solid #1a1a1a;padding:7px 16px;display:flex;
 justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;height:38px}
.hdr-l{display:flex;align-items:center;gap:0}
.bk{color:#555;font-size:9px;padding:4px 8px;border-right:1px solid #1a1a1a;
 text-decoration:none;display:flex;align-items:center;gap:3px;
 transition:color .15s;margin-right:10px;white-space:nowrap}
.bk:hover{color:#0f0}
.bk svg{width:8px;height:8px;flex-shrink:0}
.hdr h1{color:#0f0;font-size:11px;text-transform:uppercase;letter-spacing:2px;white-space:nowrap}
.hdr .cite{color:#00aacc;font-size:9px;font-style:italic;margin-left:6px;white-space:nowrap}
.hdr-r{display:flex;gap:14px;font-size:10px;flex-wrap:wrap}
.hdr-r .v{color:#0f0}
.dot{width:7px;height:7px;border-radius:50%;background:#333;display:inline-block;margin-right:4px}
.dot.on{background:#0f0;animation:bk 1s infinite}
@keyframes bk{50%{opacity:.3}}
.cp{position:fixed;top:42px;left:10px;z-index:100;
 background:rgba(4,8,14,.93);border:1px solid #162016;border-radius:4px;
 padding:12px;width:210px;max-height:calc(100vh - 55px);overflow-y:auto}
.cp::-webkit-scrollbar{width:3px}.cp::-webkit-scrollbar-thumb{background:#1a2a1a}
.sc{font-size:9px;color:#0f0;text-transform:uppercase;letter-spacing:1px;
 margin:10px 0 5px;padding-bottom:3px;border-bottom:1px solid #162016}
.sc:first-child{margin-top:0}
.rg{display:flex;flex-direction:column;gap:2px}
.rg label{font-size:10px;cursor:pointer;padding:3px 5px;border-radius:2px;
 display:flex;align-items:center;gap:5px;transition:background .15s}
.rg label:hover{background:rgba(0,255,0,.04)}
.rg input[type=radio]{accent-color:#0f0}
.sg{margin:5px 0}
.sg .sl{font-size:9px;color:#555;display:flex;justify-content:space-between}
.sg .sl span{color:#0f0}
.sg input[type=range]{width:100%;margin-top:3px;accent-color:#0f0;height:4px}
.btn{width:100%;padding:7px;margin-top:5px;border:1px solid #333;background:#111;
 color:#0f0;font-family:inherit;font-size:10px;text-transform:uppercase;
 cursor:pointer;letter-spacing:1px;transition:all .15s;display:flex;
 align-items:center;justify-content:center;gap:5px}
.btn:hover{background:#1a1a1a;border-color:#0f0}
.btn:disabled{opacity:.4;cursor:default}
.btn.go{background:#0a2a0a;border-color:#0a0}
.btn.rst{background:#2a1a0a;border-color:#f80;color:#f80}
.btn svg{width:10px;height:10px;fill:currentColor}
.ta{font-size:11px;color:#aa0;padding:3px 0}
.tr{display:flex;align-items:center;margin-bottom:2px;font-size:7px}
.tr .a{width:30px;color:#555}
.tr .bg{flex:1;height:7px;background:#111;margin:0 4px;border-radius:2px;overflow:hidden}
.tr .fl{height:100%;border-radius:2px;transition:width .3s}
.tr .vl{width:26px;text-align:right}
.rb{display:flex;align-items:center;margin-bottom:3px;font-size:9px}
.rb .rl{width:42px;color:#556;font-size:8px}
.rb .rbg{flex:1;height:6px;background:#111;border-radius:2px;overflow:hidden;margin:0 4px}
.rb .rfill{height:100%;border-radius:2px;transition:width .3s}
.rb .rv{width:28px;text-align:right;font-size:8px}
.sp{position:fixed;top:42px;right:10px;z-index:100;
 background:rgba(4,8,14,.93);border:1px solid #162016;border-radius:4px;
 padding:14px;width:240px;display:none;max-width:calc(100vw - 240px)}
.sp.show{display:block}
.sp h3{font-size:9px;color:#0f0;text-transform:uppercase;margin-bottom:8px;
 padding-bottom:3px;border-bottom:1px solid #162016}
.rw{display:flex;justify-content:space-between;padding:3px 0;font-size:10px;
 border-bottom:1px solid #0a0a0a}
.rw .lb{color:#555}.rw .vv{font-weight:bold}
.big{text-align:center;padding:10px;margin-top:8px;border:1px solid #162016;
 border-radius:3px;background:rgba(0,0,0,.3)}
.big .n{font-size:20px;font-weight:bold}
.big .l{font-size:8px;color:#555;margin-top:2px}
.lg{position:fixed;bottom:10px;left:10px;z-index:100;
 background:rgba(4,8,14,.85);border:1px solid #162016;border-radius:4px;
 padding:6px 12px;font-size:9px;display:flex;gap:14px;flex-wrap:wrap}
.lg .li{display:flex;align-items:center;gap:4px}
.lg .lc{width:16px;height:3px;border-radius:2px}
.inf{position:fixed;bottom:10px;right:10px;z-index:100;
 background:rgba(4,8,14,.8);border:1px solid #162016;border-radius:4px;
 padding:5px 10px;font-size:8px;color:#444}
#loadScr{position:fixed;inset:0;z-index:300;background:#000;display:flex;
 flex-direction:column;align-items:center;justify-content:center}
#loadScr .spinner{width:36px;height:36px;border:3px solid #1a2540;
 border-top-color:#0f0;border-radius:50%;animation:lsp .8s linear infinite}
@keyframes lsp{to{transform:rotate(360deg)}}
#loadScr .msg{color:#0f0;font-size:11px;margin-top:14px}
#loadScr .err{color:#f44;font-size:11px;margin-top:14px;display:none}
#loadScr .err a{color:#0cf}
@media(max-width:700px){
 .cp{width:170px;padding:8px;font-size:9px}
 .sp{width:200px;max-width:calc(100vw - 190px)}
 .hdr h1{font-size:9px}.lg{font-size:8px;gap:8px}.inf{display:none}
 .hdr .cite{display:none}}
@media(max-width:480px){.cp{width:140px;padding:6px}.hdr-r{display:none}}
</style></head><body>

<div id="loadScr">
 <div class="spinner"></div>
 <div class="msg">Loading 3D engine...</div>
 <div class="err" id="loadErr">3D engine failed to load.<br><a href="/">Back to Dashboard</a></div>
</div>

<canvas id="c3d"></canvas>

<div class="hdr">
 <div class="hdr-l">
  <a href="/" class="bk"><svg viewBox="0 0 8 10" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="6,1 2,5 6,9"/></svg>BACK</a>
  <h1>PSTNet Trajectory</h1><span class="cite">Kriuk et al.</span>
 </div>
 <div class="hdr-r">
  <div>Weather: <span class="v" id="hW">...</span></div>
  <div>ML: <span class="v" id="hM">init</span></div>
  <div>Time: <span class="v" id="hT">0.0 s</span></div>
  <div><span class="dot" id="hD"></span><span id="hS">Ready</span></div>
 </div>
</div>

<div class="cp">
 <div class="sc">Missile Type</div>
 <div class="rg" id="tG">
  <label><input type="radio" name="mt" value="SUPERSONIC" checked> Supersonic M 2.8</label>
  <label><input type="radio" name="mt" value="HIGH_SUPERSONIC"> Hi-Super M 4.5</label>
  <label><input type="radio" name="mt" value="HYPERSONIC"> Hypersonic M 8</label>
 </div>
 <div class="sc">Altitude</div>
 <div class="sg"><div class="sl">Launch altitude <span id="aV">10.0 km</span></div>
  <input type="range" id="aS" min="2" max="15" step="0.5" value="10"></div>
 <div class="sc">Range</div>
 <div class="sg"><div class="sl">Target range <span id="rV">120 km</span></div>
  <input type="range" id="rS" min="40" max="200" step="10" value="120"></div>
 <div class="sc">Turbulence at Alt</div>
 <div class="ta" id="tA">&mdash;</div>
 <div class="sc">Regime Gates</div>
 <div id="rgP">
  <div class="rb"><span class="rl">conv</span><div class="rbg"><div class="rfill" id="rg_conv" style="background:#f44;width:25%"></div></div><span class="rv" id="rv_conv">25%</span></div>
  <div class="rb"><span class="rl">neut</span><div class="rbg"><div class="rfill" id="rg_neut" style="background:#4a4;width:25%"></div></div><span class="rv" id="rv_neut">25%</span></div>
  <div class="rb"><span class="rl">stab</span><div class="rbg"><div class="rfill" id="rg_stab" style="background:#48f;width:25%"></div></div><span class="rv" id="rv_stab">25%</span></div>
  <div class="rb"><span class="rl">strat</span><div class="rbg"><div class="rfill" id="rg_stra" style="background:#c4f;width:25%"></div></div><span class="rv" id="rv_stra">25%</span></div>
 </div>
 <div style="font-size:8px;color:#333;margin-top:2px" id="rgD">&mdash;</div>
 <button class="btn go" id="bL">
  <svg viewBox="0 0 10 10"><polygon points="1,0 10,5 1,10"/></svg> LAUNCH
 </button>
 <button class="btn rst" id="bR">
  <svg viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 5a4 4 0 1 1 1.5 3"/><polyline points="1,5 1,8 4,8"/></svg>
  RESET
 </button>
 <button class="btn" id="bW" style="margin-top:3px;font-size:8px">REFRESH WEATHER</button>
 <div class="sc">Turbulence Profile</div>
 <div id="tP"></div>
 <div class="sc">PSTNet Model</div>
 <div id="mdI" style="font-size:9px">
  <div style="color:#c4f;margin-bottom:3px">PSTNet (4-regime MoE)</div>
  <div style="display:flex;justify-content:space-between"><span style="color:#555">Status</span><span class="v" id="mS">&mdash;</span></div>
  <div style="display:flex;justify-content:space-between"><span style="color:#555">Loss</span><span style="color:#0ff" id="mL">&mdash;</span></div>
  <div style="display:flex;justify-content:space-between"><span style="color:#555">Params</span><span style="color:#0ff" id="mP">552</span></div>
  <div style="display:flex;justify-content:space-between"><span style="color:#555">Arch</span><span style="color:#888" id="mA">&mdash;</span></div>
 </div>
</div>

<div class="sp" id="sP"><h3>Impact Results</h3><div id="sR"></div></div>

<div class="lg">
 <div class="li"><div class="lc" style="background:#0cf"></div> Corrected (ML)</div>
 <div class="li"><div class="lc" style="background:#f60"></div> Uncorrected</div>
 <div class="li"><div class="lc" style="background:#0f0"></div> Launch</div>
 <div class="li"><div class="lc" style="background:#f00"></div> Target</div>
</div>

<div class="inf">Alt 3x exaggerated &middot; Scroll zoom &middot; Drag orbit</div>

<script>
var _3ok=false,_3err=false;
function _3loaded(){_3ok=true;}
function _3failed(){_3err=true;document.getElementById('loadErr').style.display='block';
 document.querySelector('#loadScr .msg').style.display='none';
 document.querySelector('#loadScr .spinner').style.display='none';}
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js" onload="_3loaded()" onerror="_3failed()"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js" onerror="_3failed()"></script>
<script>
(function(){
var _wait=setInterval(function(){
 if(_3err){clearInterval(_wait);return;}
 if(!_3ok||typeof THREE==='undefined')return;
 clearInterval(_wait);document.getElementById('loadScr').style.display='none';startApp();
},100);

function startApp(){
var VS=3,CX=90,CZ=100;
var MDEF={SUPERSONIC:{aMin:2,aMax:15,aDef:10,rDef:120},HIGH_SUPERSONIC:{aMin:8,aMax:25,aDef:18,rDef:120},HYPERSONIC:{aMin:15,aMax:35,aDef:25,rDef:150}};
var scene,cam,ren,ctl,clk;var oG,oM,sGrp,lMk,tMk;
var mC,mU,lC,lU,aC,aU;var expl=[],shkI=0,simT0=0;
var running=false,cD=false,uD=false;var tProf=[];
var cfg={missile_type:'SUPERSONIC',launch_alt:10,target_range:120};
function tw(x,y,a){return new THREE.Vector3(x-CX,a*VS,-(y-CZ))}

function init(){
 clk=new THREE.Clock();scene=new THREE.Scene();scene.background=new THREE.Color(0x000811);
 scene.fog=new THREE.FogExp2(0x000811,0.0012);
 cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.1,3000);cam.position.set(0,40,80);
 ren=new THREE.WebGLRenderer({canvas:document.getElementById('c3d'),antialias:true});
 ren.setSize(innerWidth,innerHeight);ren.setPixelRatio(Math.min(devicePixelRatio,2));
 ctl=new THREE.OrbitControls(cam,ren.domElement);ctl.enableDamping=true;ctl.dampingFactor=0.06;
 ctl.maxPolarAngle=Math.PI/2-0.02;ctl.target.set(0,10,0);
 scene.add(new THREE.AmbientLight(0x112244,0.5));
 var sun=new THREE.DirectionalLight(0xffeedd,0.6);sun.position.set(60,100,40);scene.add(sun);
 scene.add(new THREE.HemisphereLight(0x001144,0x000000,0.25));
 mkOcean();mkGrid();mkStars();mkMoon();mkShip();
 mC=mkMsl(0x00ccff);mU=mkMsl(0xee6600);mC.visible=mU.visible=false;scene.add(mC);scene.add(mU);
 lC=mkLn(0x00ccff,0.85);lU=mkLn(0xee6600,0.55);scene.add(lC);scene.add(lU);
 aC=mkLn(0x00ccff,0.15);aU=mkLn(0xee6600,0.15);aC.visible=aU.visible=false;scene.add(aC);scene.add(aU);
 mkMarkers();
 window.addEventListener('resize',function(){cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);});
}
function mkOcean(){oG=new THREE.PlaneGeometry(3000,3000,60,60);oG.rotateX(-Math.PI/2);
 oM=new THREE.Mesh(oG,new THREE.MeshPhongMaterial({color:0x001828,specular:0x002233,shininess:25,transparent:true,opacity:0.92}));oM.position.y=-0.15;scene.add(oM);}
function mkGrid(){var g=new THREE.GridHelper(400,40,0x002800,0x001000);g.position.y=0.01;scene.add(g);}
function mkStars(){var n=2000,p=new Float32Array(n*3);for(var i=0;i<n;i++){p[i*3]=(Math.random()-0.5)*2000;p[i*3+1]=50+Math.random()*400;p[i*3+2]=(Math.random()-0.5)*2000;}
 var g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(p,3));
 scene.add(new THREE.Points(g,new THREE.PointsMaterial({color:0xffffff,size:0.3,transparent:true,opacity:0.45})));}
function mkMoon(){var m=new THREE.Mesh(new THREE.SphereGeometry(4,16,16),new THREE.MeshBasicMaterial({color:0xddeeff}));m.position.set(250,180,-400);scene.add(m);}
function mkShip(){sGrp=new THREE.Group();var m1=new THREE.MeshPhongMaterial({color:0x3a4a5a}),m2=new THREE.MeshPhongMaterial({color:0x4a5a6a}),m3=new THREE.MeshPhongMaterial({color:0x222a32});
 function bx(w,h,d,mt,px,py,pz){var me=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mt);me.position.set(px||0,py||0,pz||0);return me;}
 function cy(rt,rb,h,mt,px,py,pz){var me=new THREE.Mesh(new THREE.CylinderGeometry(rt,rb,h,8),mt);me.position.set(px||0,py||0,pz||0);return me;}
 sGrp.add(bx(5,.8,1.3,m1,0,.4,0));var bow=new THREE.Mesh(new THREE.ConeGeometry(.65,2,4),m1);bow.position.set(3.5,.4,0);bow.rotation.z=-Math.PI/2;sGrp.add(bow);
 sGrp.add(bx(2.2,1.2,.95,m2,-.3,1.4,0));sGrp.add(bx(1.2,.7,.7,m2,-.3,2.3,0));sGrp.add(cy(.15,.22,.65,m3,.4,1.9,0));sGrp.add(cy(.03,.03,2.2,m2,-.3,3.6,0));sGrp.add(bx(.5,.25,.04,m2,-.3,4.5,0));
 var rl=new THREE.PointLight(0xff0000,.4,5);rl.position.set(-.3,4.8,0);sGrp.add(rl);
 sGrp.add(new THREE.Mesh(new THREE.SphereGeometry(.06,6,6),new THREE.MeshBasicMaterial({color:0x00ff00})));sGrp.children[sGrp.children.length-1].position.set(-2.3,.9,.65);
 sGrp.add(new THREE.Mesh(new THREE.SphereGeometry(.06,6,6),new THREE.MeshBasicMaterial({color:0xff0000})));sGrp.children[sGrp.children.length-1].position.set(-2.3,.9,-.65);
 updShip();scene.add(sGrp);}
function updShip(){var p=tw(30+cfg.target_range,100,0);sGrp.position.set(p.x,0,p.z);sGrp.rotation.y=Math.PI*.7;}
function mkMsl(color){var g=new THREE.Group(),mt=new THREE.MeshPhongMaterial({color:color,emissive:color,emissiveIntensity:.3});
 var bd=new THREE.Mesh(new THREE.CylinderGeometry(.12,.12,1.8,8),mt);bd.rotation.x=Math.PI/2;g.add(bd);
 var ns=new THREE.Mesh(new THREE.ConeGeometry(.12,.5,8),mt);ns.rotation.x=-Math.PI/2;ns.position.z=-1.15;g.add(ns);
 var fm=new THREE.MeshPhongMaterial({color:0x444444});
 for(var i=0;i<4;i++){var f=new THREE.Mesh(new THREE.BoxGeometry(.35,.02,.25),fm);f.position.z=.7;var w=new THREE.Group();w.add(f);w.rotation.z=(Math.PI/2)*i;g.add(w);}
 g.add(new THREE.Mesh(new THREE.SphereGeometry(.16,8,8),new THREE.MeshBasicMaterial({color:0xff4400,transparent:true,opacity:.7})));g.children[g.children.length-1].position.z=1.05;
 var pl=new THREE.PointLight(color,.8,6);pl.position.z=1;g.add(pl);return g;}
function mkLn(color,op){return new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:color,transparent:true,opacity:op||.8}));}
function setLn(ln,pts){if(!pts||pts.length<2)return;var a=new Float32Array(pts.length*3);
 for(var i=0;i<pts.length;i++){var w=tw(pts[i][0],pts[i][1],pts[i][2]);a[i*3]=w.x;a[i*3+1]=w.y;a[i*3+2]=w.z;}
 var ng=new THREE.BufferGeometry();ng.setAttribute('position',new THREE.BufferAttribute(a,3));var og=ln.geometry;ln.geometry=ng;og.dispose();}
function setAl(ln,wp){var a=new Float32Array([wp.x,wp.y,wp.z,wp.x,0,wp.z]);var ng=new THREE.BufferGeometry();ng.setAttribute('position',new THREE.BufferAttribute(a,3));var og=ln.geometry;ln.geometry=ng;og.dispose();}
function oriM(m,tr){var n=tr.length;if(n<2)return;var a=tw(tr[n-2][0],tr[n-2][1],tr[n-2][2]),b=tw(tr[n-1][0],tr[n-1][1],tr[n-1][2]);m.lookAt(m.position.clone().add(b.clone().sub(a).normalize()));}
function mkMarkers(){lMk=new THREE.Group();
 lMk.add(new THREE.Mesh(new THREE.RingGeometry(1.5,2,32).rotateX(-Math.PI/2),new THREE.MeshBasicMaterial({color:0x00ff00,side:THREE.DoubleSide,transparent:true,opacity:.35})));
 lMk.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0),new THREE.Vector3(0,4,0)]),new THREE.LineBasicMaterial({color:0x00ff00,transparent:true,opacity:.2})));
 var lp=tw(30,100,0);lMk.position.set(lp.x,.05,lp.z);scene.add(lMk);
 tMk=new THREE.Group();
 tMk.add(new THREE.Mesh(new THREE.RingGeometry(1.5,2,32).rotateX(-Math.PI/2),new THREE.MeshBasicMaterial({color:0xff0000,side:THREE.DoubleSide,transparent:true,opacity:.3})));
 tMk.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-2,0,0),new THREE.Vector3(2,0,0)]),new THREE.LineBasicMaterial({color:0xff0000,transparent:true,opacity:.2})));
 tMk.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,-2),new THREE.Vector3(0,0,2)]),new THREE.LineBasicMaterial({color:0xff0000,transparent:true,opacity:.2})));
 var tp=tw(30+cfg.target_range,100,0);tMk.position.set(tp.x,.05,tp.z);scene.add(tMk);}
function boom(pos,color){var N=300,P=new Float32Array(N*3),C=new Float32Array(N*3),V=[];
 for(var i=0;i<N;i++){P[i*3]=pos.x;P[i*3+1]=pos.y;P[i*3+2]=pos.z;
  var th=Math.random()*6.283,ph=Math.acos(2*Math.random()-1),sp=1+Math.random()*5;
  V.push(Math.sin(ph)*Math.cos(th)*sp,Math.sin(ph)*Math.sin(th)*sp*0.5+2.5,Math.cos(ph)*sp);
  var mx=Math.random();C[i*3]=1;C[i*3+1]=.3+mx*.7;C[i*3+2]=mx*.2;}
 var g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(P,3));g.setAttribute('color',new THREE.BufferAttribute(C,3));
 var pts=new THREE.Points(g,new THREE.PointsMaterial({size:.45,vertexColors:true,transparent:true,opacity:1,blending:THREE.AdditiveBlending,depthWrite:false}));scene.add(pts);
 var fl=new THREE.PointLight(0xffaa44,15,80);fl.position.copy(pos);scene.add(fl);
 var sp2=new THREE.Mesh(new THREE.SphereGeometry(.8,12,12),new THREE.MeshBasicMaterial({color:0xffcc00,transparent:true,opacity:.85}));sp2.position.copy(pos);scene.add(sp2);
 var rg=new THREE.Mesh(new THREE.RingGeometry(.5,1.5,24).rotateX(-Math.PI/2),new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:.6,side:THREE.DoubleSide}));rg.position.set(pos.x,.1,pos.z);scene.add(rg);
 expl.push({pts:pts,V:V,age:0,mx:3.5,fl:fl,sp:sp2,rg:rg});shkI=1.5;}
function updExpl(dt){for(var i=expl.length-1;i>=0;i--){var e=expl[i];e.age+=dt;
 if(e.age>e.mx){scene.remove(e.pts);scene.remove(e.fl);scene.remove(e.sp);scene.remove(e.rg);e.pts.geometry.dispose();e.pts.material.dispose();e.sp.geometry.dispose();e.sp.material.dispose();e.rg.geometry.dispose();e.rg.material.dispose();expl.splice(i,1);continue;}
 var pa=e.pts.geometry.attributes.position;for(var j=0;j<pa.count;j++){pa.array[j*3]+=e.V[j*3]*dt;pa.array[j*3+1]+=e.V[j*3+1]*dt;pa.array[j*3+2]+=e.V[j*3+2]*dt;e.V[j*3+1]-=5*dt;}
 pa.needsUpdate=true;e.pts.material.opacity=Math.max(0,1-e.age/e.mx);e.fl.intensity=Math.max(0,15*(1-e.age/.3));
 var s=1+e.age*4;e.sp.scale.set(s,s,s);e.sp.material.opacity=Math.max(0,.85*(1-e.age/.5));
 var rs=1+e.age*6;e.rg.scale.set(rs,1,rs);e.rg.material.opacity=Math.max(0,.6*(1-e.age/2));}}

function rLoop(){requestAnimationFrame(rLoop);var dt=clk.getDelta(),t=clk.getElapsedTime();ctl.update();
 var op=oG.attributes.position;for(var i=0;i<op.count;i++){op.setY(i,Math.sin(op.getX(i)*.015+t*.4)*.1+Math.cos(op.getZ(i)*.02+t*.3)*.07);}op.needsUpdate=true;
 if(sGrp){sGrp.position.y=Math.sin(t*.5)*.06;sGrp.rotation.x=Math.sin(t*.3)*.006;}
 updExpl(dt);
 if(shkI>.01){scene.position.set((Math.random()-.5)*shkI,(Math.random()-.5)*shkI*.5,(Math.random()-.5)*shkI);shkI*=.92;}else{scene.position.set(0,0,0);shkI=0;}
 ren.render(scene,cam);}

function fetchW(){document.getElementById('hW').textContent='...';
 fetch('/api/weather').then(function(r){return r.json()}).then(function(r){
  document.getElementById('hW').textContent='OK';document.getElementById('hM').textContent='ready';
  tProf=r.turbulence_profile||[];updTurbUI();updTurbAt();updRegime();
  if(r.model_info){document.getElementById('mS').textContent=r.model_info.trained?'Trained':'...';
   if(r.model_info.final_loss!=null)document.getElementById('mL').textContent=r.model_info.final_loss.toFixed(6);
   if(r.model_info.total_params!=null)document.getElementById('mP').textContent=r.model_info.total_params;
   if(r.model_info.architecture)document.getElementById('mA').textContent=r.model_info.architecture;}
 }).catch(function(e){document.getElementById('hW').textContent='err';});}
function updTurbUI(){var h='';tProf.forEach(function(p){var v=Math.min(p.turbulence*100,100),c=v<20?'#0a0':v<40?'#aa0':'#c40';
 h+='<div class="tr"><span class="a">'+p.altitude+'</span><div class="bg"><div class="fl" style="width:'+v+'%;background:'+c+'"></div></div><span class="vl" style="color:'+c+'">'+v.toFixed(0)+'%</span></div>';});
 document.getElementById('tP').innerHTML=h;}
function updTurbAt(){var alt=parseFloat(document.getElementById('aS').value),turb=0;
 if(tProf.length){var best=tProf[0];for(var i=0;i<tProf.length;i++)if(Math.abs(tProf[i].altitude-alt)<Math.abs(best.altitude-alt))best=tProf[i];turb=best.turbulence;}
 var v=(turb*100).toFixed(0),c=turb<.2?'#0a0':turb<.4?'#aa0':'#c40';
 document.getElementById('tA').innerHTML='<span style="color:'+c+';font-weight:bold">'+v+'%</span> <span style="color:#555">at '+alt.toFixed(1)+' km</span>';}
function updRegime(){var alt=parseFloat(document.getElementById('aS').value);
 fetch('/api/regime?alt='+alt).then(function(r){return r.json()}).then(function(r){
  if(!r.weights)return;var w=r.weights;
  var names={convective:'conv',neutral:'neut',stable:'stab',stratospheric:'stra'};
  var colors={convective:'#f44',neutral:'#4a4',stable:'#48f',stratospheric:'#c4f'};
  var maxK='',maxV=0;
  for(var k in w){var short=names[k]||k.substr(0,4);var el=document.getElementById('rg_'+short),vl=document.getElementById('rv_'+short);
   if(el)el.style.width=(w[k]*100).toFixed(0)+'%';if(vl){vl.textContent=(w[k]*100).toFixed(0)+'%';vl.style.color=colors[k]||'#888';}
   if(w[k]>maxV){maxV=w[k];maxK=k;}}
  document.getElementById('rgD').innerHTML='dominant: <span style="color:'+(colors[maxK]||'#888')+'">'+maxK+'</span> ('+(maxV*100).toFixed(0)+'%)';
  if(r.correction){var cc=r.correction;document.getElementById('rgD').innerHTML+='<br><span style="color:#555">str='+cc.correction_strength.toFixed(2)+' rel='+cc.reliability.toFixed(2)+' dft='+cc.drift_scale.toFixed(2)+'</span>';}
 }).catch(function(){});}
function cfgSim(){cfg.missile_type=document.querySelector('input[name="mt"]:checked').value;cfg.launch_alt=parseFloat(document.getElementById('aS').value);cfg.target_range=parseFloat(document.getElementById('rS').value);
 return fetch('/api/traj/configure',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});}
function doLaunch(){document.getElementById('bL').disabled=true;
 cfgSim().then(function(){cD=uD=false;mC.visible=mU.visible=true;aC.visible=aU.visible=true;
  [lC,lU,aC,aU].forEach(function(l){var g=l.geometry;l.geometry=new THREE.BufferGeometry();g.dispose();});
  expl.forEach(function(e){scene.remove(e.pts);scene.remove(e.fl);scene.remove(e.sp);scene.remove(e.rg);});expl=[];
  document.getElementById('sP').classList.remove('show');updShip();
  var tp=tw(30+cfg.target_range,100,0);tMk.position.set(tp.x,.05,tp.z);
  cam.position.set(0,Math.max(25,cfg.launch_alt*VS*1.1),cfg.target_range*.55);ctl.target.set(0,cfg.launch_alt*VS*.25,0);ctl.update();
  return fetch('/api/traj/launch',{method:'POST'});
 }).then(function(){running=true;simT0=Date.now();
  document.getElementById('hS').textContent='Simulating...';document.getElementById('hD').classList.add('on');simLoop();
 }).catch(function(e){console.error(e);document.getElementById('bL').disabled=false;});}
function simLoop(){if(!running)return;
 if(Date.now()-simT0>90000){running=false;document.getElementById('hS').textContent='Timeout';document.getElementById('hD').classList.remove('on');document.getElementById('bL').disabled=false;return;}
 fetch('/api/traj/step',{method:'POST'}).then(function(){return fetch('/api/traj/state')}).then(function(r){return r.json()}).then(function(s){
  if(s.corrected.trajectory.length>1)setLn(lC,s.corrected.trajectory);
  if(s.uncorrected.trajectory.length>1)setLn(lU,s.uncorrected.trajectory);
  if(s.corrected.status==='active'&&s.corrected.trajectory.length>0){var l=s.corrected.trajectory[s.corrected.trajectory.length-1],w=tw(l[0],l[1],l[2]);mC.position.copy(w);oriM(mC,s.corrected.trajectory);setAl(aC,w);aC.visible=true;}
  if(s.uncorrected.status==='active'&&s.uncorrected.trajectory.length>0){var l2=s.uncorrected.trajectory[s.uncorrected.trajectory.length-1],w2=tw(l2[0],l2[1],l2[2]);mU.position.copy(w2);oriM(mU,s.uncorrected.trajectory);setAl(aU,w2);aU.visible=true;}
  if(s.corrected.status==='impact'&&!cD){cD=true;boom(tw.apply(null,s.corrected.trajectory[s.corrected.trajectory.length-1]),0x00ccff);mC.visible=false;aC.visible=false;}
  if(s.uncorrected.status==='impact'&&!uD){uD=true;boom(tw.apply(null,s.uncorrected.trajectory[s.uncorrected.trajectory.length-1]),0xff6600);mU.visible=false;aU.visible=false;}
  document.getElementById('hT').textContent=s.time.toFixed(1)+' s';
  if(!s.running){running=false;document.getElementById('hS').textContent='Complete';document.getElementById('hD').classList.remove('on');document.getElementById('bL').disabled=false;showRes(s);return;}
  setTimeout(simLoop,30);
 }).catch(function(e){console.error(e);running=false;document.getElementById('bL').disabled=false;});}
function showRes(s){var cr=s.corrected_result,ur=s.uncorrected_result;if(!cr||!ur)return;
 var cm=cr.miss_distance_m,um=ur.miss_distance_m,imp=um>0?((1-cm/um)*100).toFixed(1):'0',iv=parseFloat(imp);
 var ic=iv>0?'#0f0':'#f44',wn=cm<um?'ML Corrected':'Uncorrected',wc=cm<um?'#0cf':'#f60';
 document.getElementById('sR').innerHTML='<div class="rw"><span class="lb">Corrected (ML)</span><span class="vv" style="color:#0cf">'+cm.toFixed(1)+' m</span></div>'+
  '<div class="rw"><span class="lb">Uncorrected</span><span class="vv" style="color:#f60">'+um.toFixed(1)+' m</span></div>'+
  '<div class="rw"><span class="lb">Improvement</span><span class="vv" style="color:'+ic+'">'+(iv>0?'+':'')+imp+'%</span></div>'+
  '<div class="rw"><span class="lb">Avg Turbulence</span><span class="vv" style="color:#aa0">'+(cr.avg_turbulence*100).toFixed(0)+'%</span></div>'+
  '<div class="big"><div class="n" style="color:'+wc+'">'+wn+'</div><div class="l">closer to target</div></div>';
 document.getElementById('sP').classList.add('show');}
function doReset(){running=false;fetch('/api/traj/reset',{method:'POST'}).then(function(){
 mC.visible=mU.visible=false;aC.visible=aU.visible=false;
 [lC,lU,aC,aU].forEach(function(l){var g=l.geometry;l.geometry=new THREE.BufferGeometry();g.dispose();});
 expl.forEach(function(e){scene.remove(e.pts);scene.remove(e.fl);scene.remove(e.sp);scene.remove(e.rg);});expl=[];shkI=0;scene.position.set(0,0,0);
 document.getElementById('sP').classList.remove('show');document.getElementById('hS').textContent='Ready';
 document.getElementById('hD').classList.remove('on');document.getElementById('hT').textContent='0.0 s';document.getElementById('bL').disabled=false;});}

document.getElementById('bL').onclick=doLaunch;document.getElementById('bR').onclick=doReset;document.getElementById('bW').onclick=fetchW;
document.getElementById('aS').oninput=function(){document.getElementById('aV').textContent=this.value+' km';updTurbAt();updRegime();};
document.getElementById('rS').oninput=function(){document.getElementById('rV').textContent=this.value+' km';};
document.querySelectorAll('input[name="mt"]').forEach(function(r){r.onchange=function(){var d=MDEF[this.value],a=document.getElementById('aS');
 a.min=d.aMin;a.max=d.aMax;a.value=d.aDef;document.getElementById('aV').textContent=d.aDef+' km';
 var rs=document.getElementById('rS');rs.value=d.rDef;document.getElementById('rV').textContent=d.rDef+' km';updTurbAt();updRegime();};});

init();fetchW();rLoop();
}
})();
</script></body></html>'''


# ===================================================================
# GLOBE HTML
# ===================================================================
GLOBE_HTML = r'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSTNet Globe - Kriuk et al.</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000308;overflow:hidden;font-family:'Courier New',monospace;color:#8899aa}
#cv{display:block;cursor:grab}
#cv.dragging{cursor:grabbing}
#cv.selecting{cursor:crosshair}
.hdr{position:fixed;top:0;left:0;right:0;z-index:100;height:38px;
 background:rgba(4,8,16,.92);border-bottom:1px solid #1a2540;
 display:flex;align-items:center;padding:0 14px;gap:0}
.bk{color:#556;font-size:9px;padding:4px 8px;border-right:1px solid #1a2540;
 text-decoration:none;display:flex;align-items:center;gap:3px;
 transition:color .15s;margin-right:10px;white-space:nowrap}
.bk:hover{color:#00dd88}
.bk svg{width:8px;height:8px;flex-shrink:0}
.hdr h1{color:#00dd88;font-size:11px;text-transform:uppercase;letter-spacing:2px;white-space:nowrap}
.hdr .cite{color:#00aacc;font-size:9px;font-style:italic;margin-left:6px;white-space:nowrap}
.hdr-r{margin-left:auto;display:flex;gap:10px;font-size:10px}
.badge{padding:2px 7px;border-radius:3px;border:1px solid #1a2540;font-size:9px}
.badge.ok{border-color:#00aa66;color:#00dd88}
.cp{position:fixed;top:42px;left:8px;z-index:100;width:205px;
 background:rgba(4,8,16,.93);border:1px solid #1a2540;border-radius:4px;
 padding:10px;max-height:calc(100vh - 80px);overflow-y:auto}
.cp::-webkit-scrollbar{width:3px}.cp::-webkit-scrollbar-thumb{background:#1a2540}
.sc{font-size:9px;color:#00dd88;text-transform:uppercase;letter-spacing:1.2px;
 margin:10px 0 5px;padding-bottom:3px;border-bottom:1px solid #0d1520;font-weight:600}
.sc:first-child{margin-top:0}
.pr-b{width:100%;padding:5px 7px;margin-bottom:2px;border:1px solid #1a2540;
 background:#0a1018;color:#667788;font-family:inherit;font-size:9px;
 cursor:pointer;border-radius:3px;text-align:left;transition:all .15s}
.pr-b:hover{border-color:#334466;color:#aabbcc}
.pr-b.on{border-color:#00dd88;color:#00dd88;background:#081a14}
.sel-b{width:100%;padding:7px;margin-top:4px;border:1px solid #336;
 background:#0a0e1a;color:#6688cc;font-family:inherit;font-size:10px;
 cursor:pointer;border-radius:3px;text-align:center;transition:all .15s;
 text-transform:uppercase;letter-spacing:1px;display:flex;align-items:center;justify-content:center;gap:5px}
.sel-b:hover{border-color:#88aaff;color:#aaccff}
.sel-b.on{border-color:#ff8800;color:#ff8800;background:#1a120a;animation:pulse 1.2s infinite}
.sel-b svg{width:12px;height:12px}
@keyframes pulse{50%{opacity:.6}}
.go-b{width:100%;padding:8px;margin-top:6px;border:1px solid #00aa66;
 background:#081a14;color:#00dd88;font-family:inherit;font-size:11px;
 cursor:pointer;border-radius:3px;text-align:center;text-transform:uppercase;
 letter-spacing:1.5px;font-weight:bold;transition:all .15s;
 display:flex;align-items:center;justify-content:center;gap:5px}
.go-b:hover{background:#0a2a1a}.go-b:disabled{opacity:.3;cursor:default}
.go-b.computing{animation:pulse .8s infinite;color:#ffaa00;border-color:#ffaa00}
.go-b svg{width:11px;height:11px;fill:currentColor}
.fl-g{display:grid;grid-template-columns:1fr 1fr;gap:3px}
.fl-b{padding:5px;border:1px solid #1a2540;background:#0a1018;color:#556677;
 font-family:inherit;font-size:9px;cursor:pointer;border-radius:3px;text-align:center;transition:all .15s}
.fl-b:hover{border-color:#334466;color:#aabbcc}
.fl-b.on{border-color:#00dd88;color:#00dd88;background:#0a1a18}
.fl-b .ft{font-size:7px;color:#445566}.fl-b.on .ft{color:#00aa66}
.sr{display:flex;justify-content:space-between;padding:2px 0;font-size:9px;border-bottom:1px solid #080c14}
.sr .lb{color:#556677}.sr .vl{font-weight:600}
.sr .g{color:#00dd88}.sr .y{color:#ffaa00}.sr .r{color:#ff3344}.sr .c{color:#00aaff}
.sev-r{display:flex;align-items:center;gap:5px;padding:1px 0;font-size:8px}
.sev-d{width:8px;height:8px;border-radius:2px;flex-shrink:0}
.ftr{position:fixed;bottom:0;left:0;right:0;z-index:100;height:32px;
 background:rgba(4,8,16,.92);border-top:1px solid #1a2540;
 display:flex;align-items:center;padding:0 14px;font-size:9px;gap:12px}
.ftr .sp{color:#1a2540}.ftr .lb{color:#445566}.ftr .vl{font-weight:600}
.tv{padding:1px 5px;border-radius:2px;font-weight:700}
.ld{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);
 display:none;flex-direction:column;align-items:center;justify-content:center;z-index:200}
.spinner{width:36px;height:36px;border:3px solid #1a2540;border-top-color:#00dd88;
 border-radius:50%;animation:sp .9s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.ld .msg{margin-top:14px;font-size:11px;color:#00dd88}
.ld .sm{margin-top:5px;font-size:9px;color:#556677}
.sel-banner{position:fixed;top:44px;left:50%;transform:translateX(-50%);z-index:150;
 background:rgba(255,170,0,0.12);border:1px solid rgba(255,170,0,0.35);border-radius:4px;
 padding:8px 24px;font-size:11px;color:#ffaa00;display:none;pointer-events:none;
 letter-spacing:0.5px;white-space:nowrap}
.inf{position:fixed;bottom:38px;right:10px;z-index:100;
 background:rgba(4,8,16,.7);border:1px solid #1a2540;border-radius:4px;
 padding:4px 10px;font-size:8px;color:#445566}
@media(max-width:600px){.cp{width:160px;font-size:8px}.inf{display:none}.hdr .cite{display:none}}
</style></head><body>
<canvas id="cv"></canvas>

<div class="hdr">
 <a href="/" class="bk"><svg viewBox="0 0 8 10" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="6,1 2,5 6,9"/></svg>BACK</a>
 <h1>PSTNet Globe</h1><span class="cite">Kriuk et al.</span>
 <div class="hdr-r">
  <div class="badge" id="hM">PSTNet: idle</div>
  <div class="badge" id="hFL">FL300</div>
  <div class="badge" id="hR">No region</div>
 </div>
</div>

<div class="cp">
 <div class="sc">Region Presets</div>
 <div style="font-size:8px;color:#445566;margin-bottom:4px">Click a preset to auto-compute</div>
 <div id="prG"></div>
 <div class="sc">Custom Region</div>
 <button class="sel-b" id="selB">
  <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="6" cy="6" r="4.5"/><circle cx="6" cy="6" r="1" fill="currentColor"/><line x1="6" y1="0" x2="6" y2="3"/><line x1="6" y1="9" x2="6" y2="12"/><line x1="0" y1="6" x2="3" y2="6"/><line x1="9" y1="6" x2="12" y2="6"/></svg>
  SELECT ON GLOBE
 </button>
 <div class="sc">Selected Region</div>
 <div id="rcD" style="font-size:9px;color:#556677">No region selected</div>
 <button class="go-b" id="goB" disabled>
  <svg viewBox="0 0 11 11"><polygon points="1,0 11,5.5 1,11"/></svg>
  COMPUTE TURBULENCE
 </button>
 <div class="sc">Flight Level</div><div class="fl-g" id="flG"></div>
 <div class="sc">Severity Legend</div>
 <div>
  <div class="sev-r"><div class="sev-d" style="background:#042828"></div>NIL 0-5%</div>
  <div class="sev-r"><div class="sev-d" style="background:#005a23"></div>LIGHT 5-20%</div>
  <div class="sev-r"><div class="sev-d" style="background:#50a000"></div>MOD 20-40%</div>
  <div class="sev-r"><div class="sev-d" style="background:#dc7800"></div>SEV 40-60%</div>
  <div class="sev-r"><div class="sev-d" style="background:#dc0030"></div>EXT &gt;60%</div>
 </div>
 <div class="sc">Statistics</div><div id="stP" style="font-size:9px;color:#556677">&mdash;</div>
 <div class="sc">PSTNet Model</div><div id="mdP" style="font-size:9px;color:#556677">&mdash;</div>
</div>

<div class="ftr">
 <span><span class="lb">Pos </span><span class="vl" id="fC">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Turb </span><span class="tv" id="fT">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Sev </span><span class="vl" id="fS">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Wind </span><span class="vl" id="fW">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Terr </span><span class="vl" id="fE">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Regime </span><span class="vl" id="fR">&mdash;</span></span>
</div>

<div id="selBanner" class="sel-banner">STEP 1: Click the first corner on the globe</div>
<div class="inf">Drag to rotate &middot; Scroll to zoom &middot; Click presets or draw region on globe</div>

<div class="ld" id="ldComp">
 <div class="spinner"></div>
 <div class="msg">Computing turbulence...</div>
 <div class="sm">Training PSTNet + 8 flight levels x 576 grid points</div>
</div>

<script>
(function(){
'use strict';
var D=Math.PI/180,W,H,MAP_W=1024,MAP_H=512,GRES=400;
var vLat=25,vLon=20,zoomR=1.0;
var autoRot=true,drag=false,dsx=0,dsy=0,dvlat=0,dvlon=0;
var selMode=false,selPts=[];
var region=null,DATA=null,FL='FL300';
var globeDirty=true,hoverLL=null;
var cv=document.getElementById('cv'),ctx=cv.getContext('2d');
var mapCv=document.createElement('canvas');mapCv.width=MAP_W;mapCv.height=MAP_H;var mapCtx=mapCv.getContext('2d');var mapImg=null;
var turbCv=document.createElement('canvas');turbCv.width=MAP_W;turbCv.height=MAP_H;var turbCtx=turbCv.getContext('2d');var turbImg=null;
var gCv=document.createElement('canvas');gCv.width=GRES;gCv.height=GRES;var gCtx=gCv.getContext('2d');
var gcx,gcy,gsr;
var CTS=[
[[37,-10],[35,12],[33,30],[25,35],[12,44],[2,42],[-5,40],[-12,38],[-22,36],[-28,33],[-35,25],[-35,18],[-28,16],[-12,14],[-5,8],[5,-5],[5,-12],[10,-16],[15,-17],[22,-17],[28,-14],[33,-8]],
[[36,-10],[38,0],[43,-9],[48,-5],[47,2],[51,4],[54,8],[57,10],[60,5],[63,10],[66,14],[70,25],[70,30],[68,28],[62,20],[56,12],[52,6],[48,8],[45,14],[42,20],[40,26],[42,28],[40,30],[38,24],[36,0]],
[[42,28],[45,38],[42,45],[48,50],[52,55],[55,62],[58,68],[62,68],[65,72],[70,70],[72,78],[72,100],[72,120],[70,135],[68,140],[60,140],[52,135],[45,132],[38,128],[35,120],[30,122],[25,105],[22,108],[15,108],[8,100],[4,102],[1,104],[-2,105],[-6,106],[-8,112],[-8,120],[-5,130],[-2,135],[2,128],[6,120],[10,118],[18,108],[22,106],[25,100],[25,88],[22,82],[20,76],[22,72],[25,66],[28,56],[32,48],[35,38]],
[[30,68],[28,76],[24,82],[20,86],[16,82],[12,80],[8,77],[10,74],[15,74],[20,72],[25,68]],
[[30,34],[33,44],[37,44],[35,50],[32,55],[28,57],[24,58],[20,55],[18,50],[15,46],[18,42],[22,38]],
[[18,-88],[15,-84],[14,-88],[20,-105],[25,-110],[30,-118],[35,-120],[40,-124],[48,-125],[52,-130],[55,-133],[58,-138],[60,-145],[62,-155],[64,-165],[68,-165],[72,-160],[72,-125],[72,-95],[72,-80],[68,-65],[60,-58],[50,-55],[47,-62],[44,-66],[40,-70],[35,-78],[30,-82],[25,-80],[22,-84]],
[[12,-70],[12,-62],[10,-55],[5,-52],[0,-50],[-5,-38],[-10,-37],[-15,-40],[-22,-42],[-28,-48],[-32,-52],[-40,-62],[-50,-65],[-55,-68],[-55,-74],[-50,-75],[-35,-72],[-20,-70],[-5,-80],[2,-78],[5,-77],[8,-73]],
[[-15,125],[-20,116],[-25,114],[-30,115],[-35,118],[-37,138],[-37,148],[-30,153],[-22,150],[-15,142],[-12,132]],
[[60,-48],[65,-52],[70,-55],[76,-58],[80,-55],[82,-44],[82,-30],[80,-20],[75,-18],[70,-22],[65,-38]]
];
var STOPS=[[0,.02,30,30],[.05,0,65,30],[.10,0,105,40],[.20,85,165,0],[.30,175,175,0],[.40,225,125,0],[.50,245,65,0],[.60,225,0,48],[.80,185,0,185]];
function tClrA(t,a){if(t<0)return'rgba(25,15,10,'+a+')';t=Math.max(0,Math.min(.8,t));
 for(var i=0;i<STOPS.length-1;i++){if(t<=STOPS[i+1][0]){var f=(t-STOPS[i][0])/(STOPS[i+1][0]-STOPS[i][0]);
  return'rgba('+Math.round(STOPS[i][1]+f*(STOPS[i+1][1]-STOPS[i][1]))+','+Math.round(STOPS[i][2]+f*(STOPS[i+1][2]-STOPS[i][2]))+','+Math.round(STOPS[i][3]+f*(STOPS[i+1][3]-STOPS[i][3]))+','+a+')';}}return'rgba(185,0,185,'+a+')';}
function initMap(){mapCtx.fillStyle='#081422';mapCtx.fillRect(0,0,MAP_W,MAP_H);
 CTS.forEach(function(pts){mapCtx.beginPath();pts.forEach(function(p,i){var x=(p[1]+180)/360*MAP_W,y=(90-p[0])/180*MAP_H;i===0?mapCtx.moveTo(x,y):mapCtx.lineTo(x,y);});mapCtx.closePath();mapCtx.fillStyle='#0c2018';mapCtx.fill();mapCtx.strokeStyle='rgba(40,115,65,0.55)';mapCtx.lineWidth=1.2;mapCtx.stroke();});
 mapCtx.strokeStyle='rgba(25,55,75,0.18)';mapCtx.lineWidth=0.6;
 for(var la=-60;la<=60;la+=30){var y=(90-la)/180*MAP_H;mapCtx.beginPath();mapCtx.moveTo(0,y);mapCtx.lineTo(MAP_W,y);mapCtx.stroke();}
 for(var lo=-150;lo<=180;lo+=30){var x=(lo+180)/360*MAP_W;mapCtx.beginPath();mapCtx.moveTo(x,0);mapCtx.lineTo(x,MAP_H);mapCtx.stroke();}
 mapCtx.strokeStyle='rgba(30,90,110,0.25)';mapCtx.lineWidth=1;var ey=90/180*MAP_H;mapCtx.beginPath();mapCtx.moveTo(0,ey);mapCtx.lineTo(MAP_W,ey);mapCtx.stroke();
 mapImg=mapCtx.getImageData(0,0,MAP_W,MAP_H);}
function buildTurbTex(){turbCtx.clearRect(0,0,MAP_W,MAP_H);turbImg=null;if(!DATA)return;var fl=null;
 for(var k=0;k<DATA.flight_levels.length;k++)if(DATA.flight_levels[k].name===FL){fl=DATA.flight_levels[k];break;}
 if(!fl)return;var N=DATA.grid_n,lats=DATA.lats,lons=DATA.lons,dlat=lats.length>1?lats[1]-lats[0]:1,dlon=lons.length>1?lons[1]-lons[0]:1;
 for(var i=0;i<N;i++){for(var j=0;j<N;j++){var t=fl.turb[i][j],lat0=lats[i]-dlat/2,lat1=lats[i]+dlat/2,lon0=lons[j]-dlon/2,lon1=lons[j]+dlon/2;
  var x0=(lon0+180)/360*MAP_W,x1=(lon1+180)/360*MAP_W,y0=(90-lat1)/180*MAP_H,y1=(90-lat0)/180*MAP_H;
  turbCtx.fillStyle=t<-0.5?'rgba(30,18,8,0.8)':tClrA(t,0.85);turbCtx.fillRect(x0,y0,x1-x0,y1-y0);}}
 turbImg=turbCtx.getImageData(0,0,MAP_W,MAP_H);globeDirty=true;}
function renderGlobe(){var res=GRES,hr=res/2,out=gCtx.createImageData(res,res),od=out.data,md=mapImg.data,td=turbImg?turbImg.data:null;
 var p0=vLat*D,l0=vLon*D,sp0=Math.sin(p0),cp0=Math.cos(p0);
 var lx=0.35,ly=0.5,lz=0.78,ll=Math.sqrt(lx*lx+ly*ly+lz*lz);lx/=ll;ly/=ll;lz/=ll;
 for(var py=0;py<res;py++){for(var px=0;px<res;px++){var nx=(px-hr)/hr,ny=(py-hr)/hr,r2=nx*nx+ny*ny;if(r2>=1)continue;
  var rho=Math.sqrt(r2),c=Math.asin(rho),sc=Math.sin(c),cc=Math.cos(c),lat,lon;
  if(rho<1e-10){lat=vLat;lon=vLon;}else{lat=Math.asin(cc*sp0-ny*sc*cp0/rho)/D;lon=(l0+Math.atan2(nx*sc,rho*cp0*cc+ny*sp0*sc))/D;}
  lon=((lon%360)+540)%360-180;var mx=((lon+180)/360*MAP_W)|0,my=((90-lat)/180*MAP_H)|0;mx=Math.max(0,Math.min(MAP_W-1,mx));my=Math.max(0,Math.min(MAP_H-1,my));var mi=(my*MAP_W+mx)*4;
  var nz=Math.sqrt(Math.max(0,1-r2)),diff=Math.max(0,nx*lx-ny*ly+nz*lz),bright=0.42+0.58*diff;bright*=(0.75+0.25*(1-rho));
  var r0=md[mi]*bright,g0=md[mi+1]*bright,b0=md[mi+2]*bright;
  if(td){var ta=td[mi+3]/255;if(ta>0.01){var oA=ta*0.82;r0=r0*(1-oA)+td[mi]*oA;g0=g0*(1-oA)+td[mi+1]*oA;b0=b0*(1-oA)+td[mi+2]*oA;}}
  var idx=(py*res+px)*4;od[idx]=Math.min(255,r0)|0;od[idx+1]=Math.min(255,g0)|0;od[idx+2]=Math.min(255,b0)|0;od[idx+3]=255;}}
 gCtx.putImageData(out,0,0);globeDirty=false;}
function proj(lat,lon){var p=lat*D,l=lon*D,p0=vLat*D,l0=vLon*D;
 var cc=Math.sin(p0)*Math.sin(p)+Math.cos(p0)*Math.cos(p)*Math.cos(l-l0);if(cc<0)return null;
 return{x:gcx+Math.cos(p)*Math.sin(l-l0)*gsr,y:gcy-(Math.cos(p0)*Math.sin(p)-Math.sin(p0)*Math.cos(p)*Math.cos(l-l0))*gsr};}
function unproj(sx,sy){var nx=(sx-gcx)/gsr,nys=(sy-gcy)/gsr,r2=nx*nx+nys*nys;if(r2>1)return null;
 var rho=Math.sqrt(r2);if(rho<1e-10)return{lat:vLat,lon:vLon};
 var c=Math.asin(rho),sc=Math.sin(c),cc=Math.cos(c),p0=vLat*D,l0=vLon*D,sp=Math.sin(p0),cp=Math.cos(p0);
 var lat=Math.asin(cc*sp-nys*sc*cp/rho)/D,lon=(l0+Math.atan2(nx*sc,rho*cp*cc+nys*sp*sc))/D;
 return{lat:lat,lon:((lon%360)+540)%360-180};}
function drawEdges(la0,lo0,la1,lo1,N,style){
 ctx.strokeStyle=style;ctx.lineWidth=2;ctx.setLineDash([4,3]);
 function dE(aA,oA,aB,oB){var s=false;for(var i=0;i<=N;i++){var la=aA+(aB-aA)*i/N,lo=oA+(oB-oA)*i/N,p=proj(la,lo);
  if(!p){s=false;continue;}if(!s){ctx.beginPath();ctx.moveTo(p.x,p.y);s=true;}else ctx.lineTo(p.x,p.y);}if(s)ctx.stroke();}
 dE(la0,lo0,la0,lo1);dE(la0,lo1,la1,lo1);dE(la1,lo1,la1,lo0);dE(la1,lo0,la0,lo0);ctx.setLineDash([]);}
function compose(){ctx.fillStyle='#000308';ctx.fillRect(0,0,W,H);
 var grd=ctx.createRadialGradient(gcx,gcy,gsr*0.92,gcx,gcy,gsr*1.12);
 grd.addColorStop(0,'rgba(0,40,120,0)');grd.addColorStop(0.5,'rgba(0,50,150,0.12)');grd.addColorStop(1,'rgba(0,20,60,0)');
 ctx.fillStyle=grd;ctx.beginPath();ctx.arc(gcx,gcy,gsr*1.12,0,Math.PI*2);ctx.fill();
 ctx.drawImage(gCv,gcx-gsr,gcy-gsr,gsr*2,gsr*2);
 var rim=ctx.createRadialGradient(gcx-gsr*0.15,gcy-gsr*0.2,gsr*0.3,gcx,gcy,gsr);
 rim.addColorStop(0,'rgba(80,120,180,0.04)');rim.addColorStop(1,'rgba(0,0,0,0)');
 ctx.fillStyle=rim;ctx.beginPath();ctx.arc(gcx,gcy,gsr,0,Math.PI*2);ctx.fill();
 if(region){var t=Date.now()/1000;drawEdges(region.lat_min,region.lon_min,region.lat_max,region.lon_max,40,'rgba(0,221,136,'+(0.5+0.35*Math.sin(t*2.5))+')');}
 if(selMode&&selPts.length===1&&hoverLL){var a=selPts[0],b=hoverLL,t2=Date.now()/1000;
  drawEdges(Math.min(a.lat,b.lat),Math.min(a.lon,b.lon),Math.max(a.lat,b.lat),Math.max(a.lon,b.lon),30,'rgba(255,170,0,'+(0.35+0.3*Math.sin(t2*5))+')');
  ctx.font='10px monospace';ctx.fillStyle='rgba(255,170,0,0.8)';
  ctx.fillText(Math.min(a.lat,b.lat).toFixed(1)+' to '+Math.max(a.lat,b.lat).toFixed(1)+' lat',gcx-gsr-60,gcy+gsr+20);
  ctx.fillText(Math.min(a.lon,b.lon).toFixed(1)+' to '+Math.max(a.lon,b.lon).toFixed(1)+' lon',gcx-gsr-60,gcy+gsr+32);}
 selPts.forEach(function(ll){var p=proj(ll.lat,ll.lon);if(!p)return;
  ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.strokeStyle='#ffaa00';ctx.lineWidth=2;ctx.stroke();
  ctx.beginPath();ctx.arc(p.x,p.y,2,0,Math.PI*2);ctx.fillStyle='#ffaa00';ctx.fill();});}
function onDown(sx,sy){var ll=unproj(sx,sy);if(!ll)return;
 if(selMode){selPts.push(ll);
  if(selPts.length===1)document.getElementById('selBanner').textContent='STEP 2: Click the second corner to define region';
  if(selPts.length>=2){var a=selPts[0],b=selPts[1];
   region={lat_min:Math.min(a.lat,b.lat),lat_max:Math.max(a.lat,b.lat),lon_min:Math.min(a.lon,b.lon),lon_max:Math.max(a.lon,b.lon),name:''};
   if(region.lat_max-region.lat_min<4)region.lat_max=region.lat_min+4;
   if(region.lon_max-region.lon_min<4)region.lon_max=region.lon_min+4;
   DATA=null;turbImg=null;buildTurbTex();updRegion();updStats();toggleSel();setTimeout(doCompute,200);}return;}
 drag=true;autoRot=false;dsx=sx;dsy=sy;dvlat=vLat;dvlon=vLon;cv.classList.add('dragging');}
function onMove(sx,sy){hoverLL=unproj(sx,sy);updFooter();if(!drag)return;
 vLon=dvlon-(sx-dsx)/gsr*50;vLat=Math.max(-85,Math.min(85,dvlat+(sy-dsy)/gsr*50));vLon=((vLon%360)+540)%360-180;globeDirty=true;}
function onUp(){drag=false;cv.classList.remove('dragging');}
cv.addEventListener('mousedown',function(e){e.preventDefault();onDown(e.clientX,e.clientY);});
window.addEventListener('mousemove',function(e){onMove(e.clientX,e.clientY);});
window.addEventListener('mouseup',function(){onUp();});
cv.addEventListener('touchstart',function(e){e.preventDefault();var t=e.touches[0];onDown(t.clientX,t.clientY);},{passive:false});
window.addEventListener('touchmove',function(e){if(drag)e.preventDefault();var t=e.touches[0];onMove(t.clientX,t.clientY);},{passive:false});
window.addEventListener('touchend',function(){onUp();});
cv.addEventListener('wheel',function(e){e.preventDefault();zoomR=Math.max(0.5,Math.min(2.5,zoomR-(e.deltaY>0?0.08:-0.08)));computeGeometry();globeDirty=true;},{passive:false});
function computeGeometry(){W=cv.width=innerWidth;H=cv.height=innerHeight;gcx=W/2+60;gcy=H/2;gsr=Math.min(W-200,H-100)*0.42*zoomR;gsr=Math.max(80,gsr);}
window.addEventListener('resize',function(){computeGeometry();globeDirty=true;});
function buildPresets(list){var g=document.getElementById('prG');g.innerHTML='';
 list.forEach(function(p){var b=document.createElement('button');b.className='pr-b';b.textContent=p.name;
  b.onclick=function(){document.querySelectorAll('.pr-b').forEach(function(x){x.classList.remove('on');});b.classList.add('on');
   region={lat_min:p.lat_min,lat_max:p.lat_max,lon_min:p.lon_min,lon_max:p.lon_max,name:p.name};
   DATA=null;turbImg=null;buildTurbTex();updRegion();updStats();autoRot=false;
   vLat=(p.lat_min+p.lat_max)/2;vLon=(p.lon_min+p.lon_max)/2;globeDirty=true;setTimeout(doCompute,300);};
  g.appendChild(b);});}
function buildFL(){var fls=['FL100','FL180','FL240','FL300','FL340','FL380','FL410','FL450'],fts=[10,18,24,30,34,38,41,45];
 var g=document.getElementById('flG');g.innerHTML='';
 fls.forEach(function(f,idx){var b=document.createElement('button');b.className='fl-b'+(f===FL?' on':'');
  b.innerHTML='<div>'+f+'</div><div class="ft">'+fts[idx]+'k ft</div>';
  b.onclick=function(){FL=f;document.querySelectorAll('.fl-b').forEach(function(x){x.classList.remove('on');});b.classList.add('on');document.getElementById('hFL').textContent=f;buildTurbTex();updStats();};
  g.appendChild(b);});}
function toggleSel(){selMode=!selMode;selPts=[];var b=document.getElementById('selB'),bn=document.getElementById('selBanner');
 if(selMode){b.classList.add('on');b.innerHTML='<svg viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="1" y1="1" x2="9" y2="9"/><line x1="9" y1="1" x2="1" y2="9"/></svg> CANCEL';
  bn.style.display='block';bn.textContent='STEP 1: Click the first corner on the globe';cv.classList.add('selecting');autoRot=false;}
 else{b.classList.remove('on');b.innerHTML='<svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="6" cy="6" r="4.5"/><circle cx="6" cy="6" r="1" fill="currentColor"/><line x1="6" y1="0" x2="6" y2="3"/><line x1="6" y1="9" x2="6" y2="12"/><line x1="0" y1="6" x2="3" y2="6"/><line x1="9" y1="6" x2="12" y2="6"/></svg> SELECT ON GLOBE';
  bn.style.display='none';cv.classList.remove('selecting');}}
document.getElementById('selB').onclick=toggleSel;
function updRegion(){var rc=document.getElementById('rcD'),gb=document.getElementById('goB'),hr=document.getElementById('hR');
 if(!region){rc.innerHTML='No region selected';gb.disabled=true;hr.textContent='No region';hr.classList.remove('ok');return;}
 rc.innerHTML='Lat: '+region.lat_min.toFixed(1)+'\u00b0 to '+region.lat_max.toFixed(1)+'\u00b0<br>Lon: '+region.lon_min.toFixed(1)+'\u00b0 to '+region.lon_max.toFixed(1)+'\u00b0'+(region.name?'<br><span style="color:#00dd88">'+region.name+'</span>':'');
 gb.disabled=false;hr.textContent=region.name||(region.lat_min.toFixed(0)+'\u00b0-'+region.lat_max.toFixed(0)+'\u00b0');hr.classList.add('ok');}
function updStats(){var sp=document.getElementById('stP'),mp=document.getElementById('mdP');
 if(!DATA){sp.innerHTML='&mdash;';mp.innerHTML='&mdash;';return;}
 var fl=null;for(var k=0;k<DATA.flight_levels.length;k++)if(DATA.flight_levels[k].name===FL){fl=DATA.flight_levels[k];break;}
 if(!fl){sp.innerHTML='&mdash;';return;}
 var s=fl.stats,vt=DATA.grid_n*DATA.grid_n-s.blocked;
 sp.innerHTML='<div class="sr"><span class="lb">Mean</span><span class="vl c">'+(s.mean*100).toFixed(1)+'%</span></div>'+
  '<div class="sr"><span class="lb">Peak</span><span class="vl '+(s.max_val>.5?'r':'y')+'">'+(s.max_val*100).toFixed(1)+'%</span></div>'+
  '<div class="sr"><span class="lb">Moderate+</span><span class="vl y">'+s.moderate+'/'+vt+'</span></div>'+
  '<div class="sr"><span class="lb">Severe+</span><span class="vl r">'+s.severe+'</span></div>'+
  '<div class="sr"><span class="lb">Extreme</span><span class="vl r">'+s.extreme+'</span></div>'+
  '<div class="sr"><span class="lb">Blocked</span><span class="vl">'+s.blocked+'</span></div>'+
  '<div class="sr"><span class="lb" style="color:#ff5050">CONV</span><span class="vl" style="color:#ff5050">'+s.regime_conv+'</span></div>'+
  '<div class="sr"><span class="lb" style="color:#50c850">NEUT</span><span class="vl" style="color:#50c850">'+s.regime_neut+'</span></div>'+
  '<div class="sr"><span class="lb" style="color:#5080ff">STAB</span><span class="vl" style="color:#5080ff">'+s.regime_stab+'</span></div>'+
  '<div class="sr"><span class="lb" style="color:#c850ff">STRT</span><span class="vl" style="color:#c850ff">'+s.regime_strt+'</span></div>';
 mp.innerHTML='<div class="sr"><span class="lb">Arch</span><span class="vl">4-regime MoE</span></div>'+
  '<div class="sr"><span class="lb">Params</span><span class="vl c">552</span></div>'+
  '<div class="sr"><span class="lb">Loss</span><span class="vl g">'+(DATA.model_loss?DATA.model_loss.toFixed(6):'?')+'</span></div>'+
  '<div class="sr"><span class="lb">Compute</span><span class="vl">'+DATA.compute_time+'s</span></div>';}
function updFooter(){var fc=document.getElementById('fC'),ft=document.getElementById('fT'),fs=document.getElementById('fS'),fw=document.getElementById('fW'),fe=document.getElementById('fE'),fr=document.getElementById('fR');
 if(!hoverLL){fc.textContent='\u2014';return;}var la=hoverLL.lat,lo=hoverLL.lon;
 fc.textContent=Math.abs(la).toFixed(1)+'\u00b0'+(la>=0?'N':'S')+' '+Math.abs(lo).toFixed(1)+'\u00b0'+(lo>=0?'E':'W');
 if(!DATA||!region){ft.textContent='\u2014';ft.style.background='transparent';ft.style.color='#8899aa';fs.textContent='\u2014';fw.textContent='\u2014';fe.textContent='\u2014';fr.textContent='\u2014';return;}
 if(la<region.lat_min||la>region.lat_max||lo<region.lon_min||lo>region.lon_max){ft.textContent='\u2014';ft.style.background='transparent';ft.style.color='#8899aa';fs.textContent='\u2014';fw.textContent='\u2014';fe.textContent='\u2014';fr.textContent='\u2014';return;}
 var fl=null;for(var k=0;k<DATA.flight_levels.length;k++)if(DATA.flight_levels[k].name===FL){fl=DATA.flight_levels[k];break;}if(!fl)return;
 var N=DATA.grid_n,gi=(la-DATA.lats[0])/(DATA.lats[N-1]-DATA.lats[0])*(N-1),gj=(lo-DATA.lons[0])/(DATA.lons[N-1]-DATA.lons[0])*(N-1);
 gi=Math.max(0,Math.min(N-2,gi));gj=Math.max(0,Math.min(N-2,gj));
 var i0=Math.floor(gi),j0=Math.floor(gj),fi=gi-i0,fj=gj-j0;
 var v=fl.turb[i0][j0]*(1-fi)*(1-fj)+fl.turb[i0][j0+1]*(1-fi)*fj+fl.turb[i0+1][j0]*fi*(1-fj)+fl.turb[i0+1][j0+1]*fi*fj;
 if(v<-0.5){ft.textContent='TERRAIN';ft.style.background='#2a1a0a';ft.style.color='#aa7744';fs.textContent='\u2014';}
 else{ft.textContent=(v*100).toFixed(0)+'%';var sev,sc;
  if(v<.05){sev='NIL';sc='#0a2a22';}else if(v<.20){sev='LIGHT';sc='#1a3a1a';}
  else if(v<.40){sev='MODERATE';sc='#3a3a0a';}else if(v<.60){sev='SEVERE';sc='#3a1a0a';}
  else{sev='EXTREME';sc='#3a0020';}
  ft.style.background=sc;ft.style.color=v<.2?'#00dd88':(v<.4?'#ffaa00':'#ff3344');fs.textContent=sev;}
 var w=fl.wind[i0][j0]*(1-fi)*(1-fj)+fl.wind[i0][j0+1]*(1-fi)*fj+fl.wind[i0+1][j0]*fi*(1-fj)+fl.wind[i0+1][j0+1]*fi*fj;fw.textContent=w.toFixed(0)+' m/s';
 var te=DATA.terrain[i0][j0]*(1-fi)*(1-fj)+DATA.terrain[i0][j0+1]*(1-fi)*fj+DATA.terrain[i0+1][j0]*fi*(1-fj)+DATA.terrain[i0+1][j0+1]*fi*fj;
 fe.textContent=te>0.1?te.toFixed(1)+' km':'flat';
 var RNAMES=['Convective','Neutral','Stable','Stratospheric'],RCLRS=['#ff5050','#50c850','#5080ff','#c850ff'];
 var ri=Math.round(gi),rj=Math.round(gj);ri=Math.max(0,Math.min(N-1,ri));rj=Math.max(0,Math.min(N-1,rj));var rv=fl.regime[ri][rj];
 if(rv>=0&&rv<4)fr.innerHTML='<span style="color:'+RCLRS[rv]+'">'+RNAMES[rv]+'</span>';else fr.textContent='\u2014';}
function doCompute(){if(!region)return;var btn=document.getElementById('goB'),ld=document.getElementById('ldComp');
 btn.disabled=true;btn.classList.add('computing');btn.innerHTML='COMPUTING...';ld.style.display='flex';
 document.getElementById('hM').textContent='PSTNet: computing...';document.getElementById('hM').style.color='#ffaa00';document.getElementById('hM').style.borderColor='#ffaa00';
 fetch('/api/globe/compute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(region)})
 .then(function(r){return r.json();}).then(function(d){DATA=d;buildTurbTex();updStats();ld.style.display='none';
  btn.classList.remove('computing');btn.innerHTML='<svg viewBox="0 0 11 11" width="11" height="11"><polygon points="1,0 11,5.5 1,11" fill="currentColor"/></svg> RECOMPUTE';btn.disabled=false;
  document.getElementById('hM').textContent='PSTNet: ready';document.getElementById('hM').classList.add('ok');
  document.getElementById('hM').style.color='';document.getElementById('hM').style.borderColor='';})
 .catch(function(e){console.error(e);ld.style.display='none';btn.classList.remove('computing');btn.innerHTML='ERROR - RETRY';btn.disabled=false;
  document.getElementById('hM').textContent='PSTNet: error';document.getElementById('hM').style.color='#ff4444';document.getElementById('hM').style.borderColor='#ff4444';});}
document.getElementById('goB').onclick=doCompute;
document.addEventListener('keydown',function(e){
 if(e.key>='1'&&e.key<='8'){var fls=['FL100','FL180','FL240','FL300','FL340','FL380','FL410','FL450'],idx=parseInt(e.key)-1;FL=fls[idx];
  document.querySelectorAll('.fl-b').forEach(function(b,i){b.classList.toggle('on',i===idx);});document.getElementById('hFL').textContent=FL;buildTurbTex();updStats();}
 if(e.key==='Escape'&&selMode)toggleSel();if(e.key==='Enter'&&region&&!document.getElementById('goB').disabled)doCompute();});
var lastT=0;
function frame(t){requestAnimationFrame(frame);if(autoRot&&!drag){vLon+=0.04;if(vLon>180)vLon-=360;globeDirty=true;}
 if(globeDirty&&t-lastT>25){renderGlobe();lastT=t;}compose();}
computeGeometry();initMap();renderGlobe();buildFL();
fetch('/api/globe/presets').then(function(r){return r.json();}).then(function(d){buildPresets(d.presets);});
requestAnimationFrame(frame);
})();
</script></body></html>'''


# ===================================================================
# ROUTES -- DASHBOARD
# ===================================================================
@app.route('/')
def dashboard():
    return render_template_string(DASH_HTML)


# ===================================================================
# ROUTES -- TRAJECTORY
# ===================================================================
@app.route('/trajectory')
def trajectory_page():
    return render_template_string(TRAJ_HTML)

@app.route('/api/weather')
def api_weather():
    w = weather_service.get_weather()
    turb_field.update()
    setup_sim(sim['missile_type'], sim['launch_alt'], sim['target_range'])
    return jsonify(weather=w, turbulence_profile=turb_field.get_altitude_turbulence_profile(), model_info=turb_field.get_model_info())

@app.route('/api/regime')
def api_regime():
    alt = float(request.args.get('alt', 10.0))
    if not turb_field.predictor.trained:
        return jsonify(weights={}, correction={})
    _, layer = turb_field.get_at(alt)
    weights = turb_field.predictor.get_regime_weights(layer['wind_speed'], layer['temperature'], layer['density'], layer['richardson'], alt, layer['pressure'])
    correction = turb_field.predictor.predict(layer['wind_speed'], layer['temperature'], layer['density'], layer['richardson'], alt, layer['pressure'])
    return jsonify(weights=weights, correction=correction)

@app.route('/api/traj/configure', methods=['POST'])
def api_traj_configure():
    d = request.json or {}
    setup_sim(d.get('missile_type', sim['missile_type']), float(d.get('launch_alt', sim['launch_alt'])), float(d.get('target_range', sim['target_range'])))
    return jsonify(success=True)

@app.route('/api/traj/launch', methods=['POST'])
def api_traj_launch():
    sim['corrected'].launch(sim['launch_pos'], sim['target_pos'], sim['launch_alt'])
    sim['uncorrected'].launch(sim['launch_pos'], sim['target_pos'], sim['launch_alt'])
    sim['running'] = True; sim['cr'] = None; sim['ur'] = None; sim['time'] = 0.0
    return jsonify(success=True)

@app.route('/api/traj/step', methods=['POST'])
def api_traj_step():
    if not sim['running']: return jsonify(success=False)
    speed = MISSILES[sim['missile_type']]['speed'] * SPEED_OF_SOUND_SEA
    dt = min(0.5, 0.12 / speed); n_sub = max(3, int(1.5 / dt))
    for _ in range(n_sub):
        if sim['cr'] is None:
            r = sim['corrected'].step(dt)
            if r and r['status'] == 'impact': sim['cr'] = r
        if sim['ur'] is None:
            r = sim['uncorrected'].step(dt)
            if r and r['status'] == 'impact': sim['ur'] = r
        sim['time'] += dt
    if sim['cr'] is not None and sim['ur'] is not None: sim['running'] = False
    return jsonify(success=True)

@app.route('/api/traj/state')
def api_traj_state():
    def mdata(m):
        if m is None: return dict(trajectory=[], cross_track=[], status='idle', position=[0,0,0])
        n = len(m.trajectory); step = max(1, n // 500)
        idx = list(range(0, n, step))
        if n > 0 and (n-1) not in idx: idx.append(n-1)
        return dict(trajectory=[[float(m.trajectory[i][0]),float(m.trajectory[i][1]),float(m.trajectory[i][2])] for i in idx if i < n],
                    cross_track=[float(m.cross_track_log[i]) for i in idx if i < len(m.cross_track_log)],
                    position=[float(m.position[j]) for j in range(3)] if m.position is not None else [0,0,0],
                    status='impact' if m.impact_error is not None else 'active')
    return jsonify(corrected=mdata(sim['corrected']), uncorrected=mdata(sim['uncorrected']),
                   corrected_result=sim['cr'], uncorrected_result=sim['ur'],
                   running=sim['running'], time=sim['time'],
                   config=dict(missile_type=sim['missile_type'], launch_alt=sim['launch_alt'],
                               target_range=sim['target_range'], launch_pos=sim['launch_pos'], target_pos=sim['target_pos']),
                   turbulence_profile=turb_field.get_altitude_turbulence_profile(), model_info=turb_field.get_model_info())

@app.route('/api/traj/reset', methods=['POST'])
def api_traj_reset():
    setup_sim(sim['missile_type'], sim['launch_alt'], sim['target_range'])
    return jsonify(success=True)


# ===================================================================
# ROUTES -- GLOBE
# ===================================================================
@app.route('/globe')
def globe_page():
    return render_template_string(GLOBE_HTML)

@app.route('/api/globe/presets')
def api_globe_presets():
    return jsonify(presets=PRESETS)

@app.route('/api/globe/compute', methods=['POST'])
def api_globe_compute():
    d = request.json or {}
    lat_min = max(-85, min(85, float(d.get('lat_min', 18))))
    lat_max = max(lat_min+2, min(85, float(d.get('lat_max', 46))))
    lon_min = max(-180, min(180, float(d.get('lon_min', 58))))
    lon_max = max(lon_min+2, min(180, float(d.get('lon_max', 102))))
    ws = WeatherService(lat=(lat_min+lat_max)/2, lon=(lon_min+lon_max)/2)
    weather = ws.fetch_nasa_power()
    if not globe_computer.trained: globe_computer.train(weather)
    result = globe_computer.compute_region(lat_min, lat_max, lon_min, lon_max, weather)
    result['weather_source'] = 'NASA POWER'
    return jsonify(result)


# ===================================================================
# MAIN
# ===================================================================
if __name__ == '__main__':
    print('='*64)
    print('  PSTNet - Physically-Structured Turbulence Network')
    print('  Kriuk et al.')
    print('  Unified Application: Trajectory + Globe')
    print('='*64)
    print('  Initializing turbulence field...')
    turb_field.update()
    setup_sim()
    print('='*64)
    print(f'  http://127.0.0.1:80')
    print(f'    /             Dashboard')
    print(f'    /trajectory   3D Missile Trajectory')
    print(f'    /globe        Turbulence Globe')
    print('='*64)
    app.run(host='0.0.0.0', port=80, debug=False, threaded=True)