#!/usr/bin/env python3
# main.py — PSTNet Aviation Turbulence Regional Map
"""
Estimates and visualises Clear Air Turbulence across the Central Asia —
Himalaya Corridor, one of the least-monitored regions for upper-air
turbulence on major aviation routes (Europe ↔ SE Asia).

Uses PSTNet (552-param physics-ML hybrid) with a single NASA POWER
weather observation, terrain physics, and the 4-regime expert gate to
fill data gaps that civil aviation currently cannot cover.

Usage:  python main.py        →  http://127.0.0.1:5890
"""

import math, time
import numpy as np
from flask import Flask, render_template_string, jsonify
from config import ALTITUDE_LAYERS, AIR_DENSITY_SEA
from weather_api import WeatherService, FALLBACK_WEATHER
from turbulence_model import TurbulencePredictor

app = Flask(__name__)

# =====================================================================
#  Configuration
# =====================================================================
REGION = dict(
    name='Central Asia \u2014 Himalaya Corridor',
    lat_min=18.0, lat_max=46.0,
    lon_min=58.0, lon_max=102.0,
)
GRID_N = 24

FLIGHT_LEVELS = [
    dict(name='FL100', alt_km=3.05,  alt_ft=10000),
    dict(name='FL180', alt_km=5.49,  alt_ft=18000),
    dict(name='FL240', alt_km=7.32,  alt_ft=24000),
    dict(name='FL300', alt_km=9.14,  alt_ft=30000),
    dict(name='FL340', alt_km=10.36, alt_ft=34000),
    dict(name='FL380', alt_km=11.58, alt_ft=38000),
    dict(name='FL410', alt_km=12.50, alt_ft=41000),
    dict(name='FL450', alt_km=13.72, alt_ft=45000),
]


# =====================================================================
#  Regional Turbulence Map
# =====================================================================
class RegionalTurbulenceMap:

    def __init__(self):
        self.lats = np.linspace(REGION['lat_min'], REGION['lat_max'], GRID_N)
        self.lons = np.linspace(REGION['lon_min'], REGION['lon_max'], GRID_N)
        self.terrain = np.zeros((GRID_N, GRID_N))
        self.terrain_grad = np.zeros((GRID_N, GRID_N))
        self.data = {}
        self.base_weather = None
        self.predictor = TurbulencePredictor()
        self.computed = False
        self.compute_time = 0.0
        self.weather_source = 'none'

    # ---- terrain elevation (Gaussian-mixture approximation) ----------
    @staticmethod
    def _terrain_km(lat, lon):
        h = 0.0
        h = max(h, 4.8 * math.exp(-((lat - 33) ** 2 / 35 + (lon - 90) ** 2 / 120)))
        h = max(h, 7.0 * math.exp(-((lat - 28.5) ** 2 / 1.2 + (lon - 86) ** 2 / 45)))
        h = max(h, 6.0 * math.exp(-((lat - 36) ** 2 / 2 + (lon - 76.5) ** 2 / 8)))
        h = max(h, 4.5 * math.exp(-((lat - 35.5) ** 2 / 3 + (lon - 70) ** 2 / 15)))
        h = max(h, 5.0 * math.exp(-((lat - 38.5) ** 2 / 2 + (lon - 73) ** 2 / 6)))
        h = max(h, 4.0 * math.exp(-((lat - 42) ** 2 / 3 + (lon - 80) ** 2 / 60)))
        h = max(h, 5.5 * math.exp(-((lat - 36) ** 2 / 1.5 + (lon - 82) ** 2 / 60)))
        return min(h, 8.5)

    def _compute_terrain(self):
        for i in range(GRID_N):
            for j in range(GRID_N):
                self.terrain[i, j] = self._terrain_km(self.lats[i], self.lons[j])
        gy, gx = np.gradient(self.terrain,
                              self.lats[1] - self.lats[0],
                              self.lons[1] - self.lons[0])
        self.terrain_grad = np.sqrt(gx ** 2 + gy ** 2)

    # ---- local atmospheric state from one observation ----------------
    def _atm_state(self, lat, lon, alt_km, terrain_km):
        w = self.base_weather
        T_sl = w['temperature_2m'] - (lat - 32.0) * 0.6
        if alt_km < 11:
            temp = T_sl - 6.5 * alt_km
        elif alt_km < 20:
            temp = T_sl - 6.5 * 11
        else:
            temp = T_sl - 6.5 * 11 + 1.0 * (alt_km - 20)
        temp = max(temp, 180.0)

        pres = w['pressure_surface'] * math.exp(-alt_km / 8.5)
        pres = max(pres, 0.001)
        density = max((pres * 1000) / (287.05 * temp), 1e-5)

        sw = w['wind_speed_10m'] * (1 + 0.4 * abs(lat - 25) / 20)
        if terrain_km > 1 and alt_km < terrain_km + 5:
            sw *= (1 + 0.15 * terrain_km / 5)
        if alt_km < 1:
            wind = sw * (1 + 0.3 * alt_km)
        elif alt_km < 10:
            wind = sw * 1.3 + (alt_km - 1) * 1.2
        elif alt_km < 15:
            wind = sw * 1.3 + 9 * 1.2 + 6 * max(0, 1 - abs(alt_km - 12) / 3)
        else:
            wind = max(sw * 1.3 + 9 * 1.2 - (alt_km - 15) * 0.6, 1.0)
        wind = max(min(wind, 80), 0.5)

        ea = max(alt_km - terrain_km * 0.3, 0.1)
        dT = -6.5 if alt_km < 11 else (0.0 if alt_km < 20 else 1.0)
        ws = max(wind / max(ea, 0.1), 0.1)
        ri = max(min((9.81 / temp) * (dT + 9.8) / (ws ** 2 + 0.01), 10), -5)
        if terrain_km > 1 and alt_km < terrain_km + 5:
            ri *= max(0.3, 1 - 0.4 * terrain_km / 5)

        return dict(wind_speed=wind, temperature=temp, density=density,
                    richardson=ri, pressure=pres)

    # ---- mountain-wave enhancement -----------------------------------
    @staticmethod
    def _mountain_wave(alt_km, wind, ri, terrain_km, terrain_grad):
        if terrain_km < 0.5:
            return 0.0
        above = alt_km - terrain_km
        if above < -2:
            hf = 0.1
        elif above < 0:
            hf = 0.3 + 0.35 * (above + 2) / 2
        elif above < 3:
            hf = 0.65 + 0.35 * (1 - above / 3)
        elif above < 8:
            hf = 0.65 * math.exp(-(above - 3) / 4)
        else:
            hf = 0.08 * math.exp(-(above - 8) / 10)
        wf = min(wind / 15, 1.5)
        gf = min(terrain_grad / 2.0, 1.5)
        tf = min(terrain_km / 4.0, 1.5)
        sf = 1.0 if 0.1 < ri < 2.0 else (0.5 if ri > 2.0 else 0.7)
        return min(0.18 * hf * wf * gf * tf * sf, 0.40)

    # ---- full computation pipeline -----------------------------------
    def compute(self):
        t0 = time.time()
        print(f'\n{"=" * 60}')
        print(f'  PSTNet Aviation Turbulence Map')
        print(f'  Region: {REGION["name"]}')
        print(f'  Grid:   {GRID_N}x{GRID_N} = {GRID_N ** 2} points')
        print(f'  Levels: {len(FLIGHT_LEVELS)}')
        print(f'{"=" * 60}')

        print('\n  [1/4] Terrain ...')
        self._compute_terrain()
        print(f'         peak {self.terrain.max():.1f} km   '
              f'grad max {self.terrain_grad.max():.2f} km/deg')

        print('  [2/4] Weather ...')
        ws = WeatherService(lat=32.0, lon=80.0)
        self.base_weather = ws.fetch_nasa_power()
        is_fb = all(abs(self.base_weather.get(k, 0) - FALLBACK_WEATHER.get(k, -1)) < 0.01
                     for k in FALLBACK_WEATHER)
        self.weather_source = 'Fallback' if is_fb else 'NASA POWER'
        print(f'         source: {self.weather_source}')

        print('  [3/4] Training PSTNet ...')
        samples = [(20, 70), (20, 90), (28, 86), (33, 90),
                   (38, 73), (42, 80), (44, 70), (44, 95)]
        profiles = []
        for slat, slon in samples:
            tkm = self._terrain_km(slat, slon)
            p = []
            for alt in ALTITUDE_LAYERS:
                s = self._atm_state(slat, slon, alt, tkm)
                s['altitude'] = alt
                p.append(s)
            profiles.append(p)
        self.predictor.fit(profiles, epochs=200, lr=0.004)

        print('  [4/4] Turbulence field ...')
        for fl in FLIGHT_LEVELS:
            ak = fl['alt_km']
            tg = np.zeros((GRID_N, GRID_N))
            wg = np.zeros((GRID_N, GRID_N))
            for i in range(GRID_N):
                for j in range(GRID_N):
                    tkm = self.terrain[i, j]
                    if ak < tkm - 0.5:
                        tg[i, j] = -1
                        continue
                    s = self._atm_state(self.lats[i], self.lons[j], ak, tkm)
                    ea = max(ak - tkm * 0.3, 0.1)
                    base = self.predictor.physics_turbulence(
                        ea, s['wind_speed'], s['richardson'], s['density'])
                    mw = self._mountain_wave(
                        ak, s['wind_speed'], s['richardson'],
                        tkm, self.terrain_grad[i, j])
                    jet = 0
                    if 9 < ak < 14:
                        jet = 0.08 * (s['wind_speed'] / 30) * math.exp(
                            -((ak - 11.5) ** 2) / 3)
                    tg[i, j] = min(base + mw + jet, 0.80)
                    wg[i, j] = s['wind_speed']
            v = tg[tg >= 0]
            st = dict(
                mean=round(float(v.mean()), 3) if len(v) else 0,
                max_val=round(float(v.max()), 3) if len(v) else 0,
                moderate=int(np.sum(v > 0.20)),
                severe=int(np.sum(v > 0.40)),
                extreme=int(np.sum(v > 0.60)),
                blocked=int(np.sum(tg < 0)),
            )
            self.data[fl['name']] = dict(turb=tg, wind=wg, stats=st)
            print(f'         {fl["name"]}  mean={st["mean"]:.3f} '
                  f'max={st["max_val"]:.3f}  '
                  f'mod={st["moderate"]} sev={st["severe"]} blk={st["blocked"]}')

        self.compute_time = time.time() - t0
        self.computed = True
        print(f'\n  Done in {self.compute_time:.1f}s\n{"=" * 60}\n')

    # ---- JSON serialisation ------------------------------------------
    def to_json(self):
        fls = []
        for fl in FLIGHT_LEVELS:
            d = self.data[fl['name']]
            fls.append(dict(
                name=fl['name'], alt_km=fl['alt_km'], alt_ft=fl['alt_ft'],
                turb=np.round(d['turb'], 3).tolist(),
                wind=np.round(d['wind'], 1).tolist(),
                stats=d['stats'],
            ))
        return dict(
            region=REGION, grid_n=GRID_N,
            lats=np.round(self.lats, 2).tolist(),
            lons=np.round(self.lons, 2).tolist(),
            terrain=np.round(self.terrain, 2).tolist(),
            flight_levels=fls,
            weather_source=self.weather_source,
            compute_time=round(self.compute_time, 1),
            model_loss=round(self.predictor.loss_history[-1], 6)
            if self.predictor.loss_history else None,
        )


rmap = RegionalTurbulenceMap()


# =====================================================================
#  HTML + CSS + JS
# =====================================================================
HTML = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSTNet — Aviation Turbulence Map</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#06090f;color:#8899aa;font-family:'SF Mono','Menlo','Consolas',monospace;
  overflow:hidden;height:100vh}
.app{display:flex;flex-direction:column;height:100vh}
/* ---- header ---- */
.hdr{height:48px;background:rgba(8,16,28,.95);border-bottom:1px solid #1a2540;
  display:flex;align-items:center;padding:0 16px;flex-shrink:0;gap:12px}
.hdr .icon{color:#00dd88;font-size:18px}
.hdr h1{font-size:12px;color:#00dd88;text-transform:uppercase;letter-spacing:2px}
.hdr .sub{font-size:10px;color:#445566;margin-left:6px}
.hdr-r{margin-left:auto;display:flex;gap:14px;font-size:10px}
.badge{padding:2px 8px;border-radius:3px;border:1px solid #1a2540}
.badge.ok{border-color:#00aa66;color:#00dd88}
/* ---- layout ---- */
.main{flex:1;display:flex;overflow:hidden}
.sb{width:228px;background:rgba(8,16,28,.95);border-right:1px solid #1a2540;
  overflow-y:auto;flex-shrink:0}
.sb::-webkit-scrollbar{width:4px}
.sb::-webkit-scrollbar-thumb{background:#1a2540;border-radius:2px}
.sb-s{padding:10px 12px;border-bottom:1px solid #0d1520}
.sb-t{font-size:9px;color:#00dd88;text-transform:uppercase;letter-spacing:1.5px;
  margin-bottom:8px;font-weight:600}
/* ---- FL buttons ---- */
.fl-g{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.fl-b{padding:6px 2px;border:1px solid #1a2540;background:#0a1018;color:#556677;
  font-family:inherit;font-size:10px;cursor:pointer;border-radius:3px;
  text-align:center;transition:all .15s;line-height:1.4}
.fl-b:hover{border-color:#334466;color:#aabb cc;background:#0e1520}
.fl-b.on{border-color:#00dd88;color:#00dd88;background:#0a1a18}
.fl-b .ft{font-size:7px;color:#445566}.fl-b.on .ft{color:#00aa66}
/* ---- legend ---- */
.lg-bar{height:12px;border-radius:2px;margin:4px 0;overflow:hidden}
.lg-bar canvas{display:block;width:100%;height:12px}
.lg-lbl{display:flex;justify-content:space-between;font-size:8px;color:#556677}
.sev-r{display:flex;align-items:center;gap:6px;padding:2px 0;font-size:9px}
.sev-d{width:10px;height:10px;border-radius:2px;flex-shrink:0}
/* ---- stats ---- */
.sr{display:flex;justify-content:space-between;padding:3px 0;font-size:10px;
  border-bottom:1px solid #0a0f18}
.sr .lb{color:#556677}.sr .vl{font-weight:600}
.sr .g{color:#00dd88}.sr .y{color:#ffaa00}.sr .r{color:#ff3344}.sr .c{color:#00aaff}
/* ---- map ---- */
.mw{flex:1;position:relative;background:#06090f}
.mw canvas{display:block}
/* ---- footer ---- */
.ftr{height:40px;background:rgba(8,16,28,.95);border-top:1px solid #1a2540;
  display:flex;align-items:center;padding:0 16px;font-size:10px;flex-shrink:0;gap:20px}
.ftr .sp{color:#1a2540}.ftr .lb{color:#445566}.ftr .vl{font-weight:600}
.tv{padding:1px 6px;border-radius:2px;font-weight:700;transition:all .2s}
/* ---- loader ---- */
.ld{position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(6,9,15,.95);
  display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:50}
.spinner{width:40px;height:40px;border:3px solid #1a2540;border-top-color:#00dd88;
  border-radius:50%;animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.ld .msg{margin-top:16px;font-size:11px;color:#00dd88}
.ld .sm{margin-top:6px;font-size:9px;color:#556677}
/* ---- responsive ---- */
@media(max-width:800px){.sb{width:170px}.sb-s{padding:8px 8px}
  .hdr .sub{display:none}.ftr{font-size:9px;gap:10px}}
@media(max-width:600px){.sb{display:none}.ftr{font-size:8px;gap:8px;height:34px}}
</style></head><body>
<div class="app">
<div class="hdr">
 <span class="icon">&#9670;</span>
 <h1>PSTNet Turbulence Map</h1>
 <span class="sub">Central Asia &mdash; Himalaya Corridor</span>
 <div class="hdr-r">
  <div class="badge" id="hW">Weather: &hellip;</div>
  <div class="badge" id="hM">Model: &hellip;</div>
  <div class="badge ok" id="hFL">FL300</div>
 </div>
</div>
<div class="main">
<div class="sb">
 <div class="sb-s">
  <div class="sb-t">Flight Level</div>
  <div class="fl-g" id="flG"></div>
 </div>
 <div class="sb-s">
  <div class="sb-t">Turbulence Severity</div>
  <div class="lg-bar" id="lgBar"></div>
  <div class="lg-lbl"><span>0%</span><span>20%</span><span>40%</span><span>60%</span><span>80%</span></div>
  <div style="margin-top:8px">
   <div class="sev-r"><div class="sev-d" style="background:#021e1e"></div>NIL (0&ndash;5%)</div>
   <div class="sev-r"><div class="sev-d" style="background:#005a23"></div>LIGHT (5&ndash;20%)</div>
   <div class="sev-r"><div class="sev-d" style="background:#50a000"></div>MODERATE (20&ndash;40%)</div>
   <div class="sev-r"><div class="sev-d" style="background:#dc7800"></div>SEVERE (40&ndash;60%)</div>
   <div class="sev-r"><div class="sev-d" style="background:#dc0030"></div>EXTREME (&gt;60%)</div>
  </div>
 </div>
 <div class="sb-s">
  <div class="sb-t">Level Statistics</div>
  <div id="stP">&mdash;</div>
 </div>
 <div class="sb-s">
  <div class="sb-t">PSTNet Model</div>
  <div id="mdP">&mdash;</div>
 </div>
 <div class="sb-s">
  <div class="sb-t">About</div>
  <div style="font-size:9px;color:#445566;line-height:1.5">
   Estimates Clear Air Turbulence using PSTNet &mdash; a 552-parameter
   physics-ML hybrid. This region has sparse real-time turbulence data
   at most flight levels. A single weather observation plus terrain
   physics and 4-regime neural experts fill the gap.
  </div>
 </div>
</div>
<div class="mw" id="mwrap">
 <canvas id="cv"></canvas>
 <div class="ld" id="ld">
  <div class="spinner"></div>
  <div class="msg">Loading turbulence data &hellip;</div>
  <div class="sm">24&times;24 grid &middot; 8 flight levels &middot; PSTNet</div>
 </div>
</div>
</div>
<div class="ftr">
 <span><span class="lb">Cursor </span><span class="vl" id="fC">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Turb </span><span class="tv" id="fT">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Severity </span><span class="vl" id="fS">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Wind </span><span class="vl" id="fW">&mdash;</span></span>
 <span class="sp">|</span>
 <span><span class="lb">Terrain </span><span class="vl" id="fE">&mdash;</span></span>
</div>
</div>
<script>
/* ================================================================
   PSTNet Aviation Turbulence Map — Frontend
   ================================================================ */
var D=null,FL='FL300',cv,cx,cw,ch,gN;
var latMin,latMax,lonMin,lonMax;
var hmC={};

/* ---- colour map ---- */
var STOPS=[
 [0.00,2,18,35],[0.05,0,42,21],[0.10,0,90,35],[0.15,10,130,30],
 [0.20,80,160,0],[0.30,170,170,0],[0.40,220,120,0],[0.50,240,60,0],
 [0.60,220,0,30],[0.70,200,0,100],[0.80,180,0,180]];
function tClr(t){
 if(t<0)return[25,18,12];
 t=Math.max(0,Math.min(t,0.80));
 for(var i=0;i<STOPS.length-1;i++){
  if(t<=STOPS[i+1][0]){
   var f=(t-STOPS[i][0])/(STOPS[i+1][0]-STOPS[i][0]);
   return[Math.round(STOPS[i][1]+f*(STOPS[i+1][1]-STOPS[i][1])),
          Math.round(STOPS[i][2]+f*(STOPS[i+1][2]-STOPS[i][2])),
          Math.round(STOPS[i][3]+f*(STOPS[i+1][3]-STOPS[i][3]))];}}
 return[180,0,180];}

/* ---- coordinate helpers ---- */
function c2g(px,py){return[latMax-(py/ch)*(latMax-latMin),lonMin+(px/cw)*(lonMax-lonMin)];}
function g2c(la,lo){return[(lo-lonMin)/(lonMax-lonMin)*cw,(latMax-la)/(latMax-latMin)*ch];}
function gi2c(gi,gj){
 var la=latMin+gi/(gN-1)*(latMax-latMin),lo=lonMin+gj/(gN-1)*(lonMax-lonMin);
 return g2c(la,lo);}

/* ---- bilinear sample ---- */
function smp(grid,px,py){
 var gi=(1-py/ch)*(gN-1),gj=(px/cw)*(gN-1);
 var i0=Math.max(0,Math.min(Math.floor(gi),gN-2));
 var j0=Math.max(0,Math.min(Math.floor(gj),gN-2));
 var i1=i0+1,j1=j0+1;
 var fi=Math.max(0,Math.min(gi-i0,1)),fj=Math.max(0,Math.min(gj-j0,1));
 var v00=grid[i0][j0],v01=grid[i0][j1],v10=grid[i1][j0],v11=grid[i1][j1];
 var nb=(v00<-0.5?1:0)+(v01<-0.5?1:0)+(v10<-0.5?1:0)+(v11<-0.5?1:0);
 if(nb>=2)return -1;
 if(v00<-0.5)v00=0;if(v01<-0.5)v01=0;if(v10<-0.5)v10=0;if(v11<-0.5)v11=0;
 return v00*(1-fi)*(1-fj)+v01*(1-fi)*fj+v10*fi*(1-fj)+v11*fi*fj;}

/* ---- heatmap image ---- */
function mkHM(tGrid){
 var sc=Math.max(1,Math.floor(Math.max(cw,ch)/500));
 var sw=Math.ceil(cw/sc),sh=Math.ceil(ch/sc);
 var oc=document.createElement('canvas');oc.width=sw;oc.height=sh;
 var ox=oc.getContext('2d');var id=ox.createImageData(sw,sh);var dd=id.data;
 for(var sy=0;sy<sh;sy++){for(var sx=0;sx<sw;sx++){
  var px=(sx+.5)*sc,py=(sy+.5)*sc;
  var v=smp(tGrid,px,py);var idx=(sy*sw+sx)*4;
  var rgb=tClr(v);
  dd[idx]=rgb[0];dd[idx+1]=rgb[1];dd[idx+2]=rgb[2];
  dd[idx+3]=v<0?240:195;}}
 ox.putImageData(id,0,0);return oc;}

/* ---- geographic data ---- */
var COAST=[
 [[18,73.2],[19,73],[20,72.8],[21,72],[22.5,69.5],[23.5,68]],
 [[23.5,68],[24.5,66.5],[25.3,63.5],[25.8,61],[26.5,58]],
 [[18,84],[19.5,84.5],[20.5,86.5],[21.5,88],[22.5,89]],
 [[22.5,89],[22.2,90.5],[21.5,91.5],[21,92]],
 [[21,92],[20,93],[19,94],[18,95.5]],
 [[22,59],[23,58.5],[24.5,58]]];
var CITY=[
 {n:'Delhi',la:28.6,lo:77.2},{n:'Mumbai',la:19.1,lo:72.9},
 {n:'Karachi',la:24.9,lo:67},{n:'Kathmandu',la:27.7,lo:85.3},
 {n:'Lhasa',la:29.7,lo:91.1},{n:'Kabul',la:34.5,lo:69.2},
 {n:'Islamabad',la:33.7,lo:73},{n:'Dhaka',la:23.8,lo:90.4},
 {n:'Kolkata',la:22.6,lo:88.4},{n:'Almaty',la:43.2,lo:76.9},
 {n:'Tashkent',la:41.3,lo:69.3},{n:'Urumqi',la:43.8,lo:87.6},
 {n:'Bishkek',la:42.9,lo:74.6},{n:'Dushanbe',la:38.6,lo:68.8},
 {n:'Muscat',la:23.6,lo:58.5},{n:'Chengdu',la:30.6,lo:104},
 {n:'Kunming',la:25,lo:102.7}];
var ROUTE=[
 {n:'EUR\u2192SEA',p:[[46,58],[40,65],[35,75],[30,85],[25,95],[18,102]]},
 {n:'DXB\u2192BKK',p:[[25,58],[26,68],[25,78],[22,90],[18,98]]},
 {n:'DEL\u2192HKG',p:[[28.6,77.2],[28,85],[26,93],[22,102]]}];

/* ---- contour (marching squares) ---- */
function drawCtr(grid,lv,col,lw){
 cx.save();cx.strokeStyle=col;cx.lineWidth=lw;cx.beginPath();
 for(var i=0;i<gN-1;i++){for(var j=0;j<gN-1;j++){
  var tl=grid[i][j],tr=grid[i][j+1],bl=grid[i+1][j],br=grid[i+1][j+1];
  var pts=[];
  if((tl>=lv)!==(tr>=lv)){var t=(lv-tl)/(tr-tl+1e-10);pts.push(gi2c(i,j+t));}
  if((tr>=lv)!==(br>=lv)){var t=(lv-tr)/(br-tr+1e-10);pts.push(gi2c(i+t,j+1));}
  if((bl>=lv)!==(br>=lv)){var t=(lv-bl)/(br-bl+1e-10);pts.push(gi2c(i+1,j+t));}
  if((tl>=lv)!==(bl>=lv)){var t=(lv-tl)/(bl-tl+1e-10);pts.push(gi2c(i+t,j));}
  for(var k=0;k+1<pts.length;k+=2){
   cx.moveTo(pts[k][0],pts[k][1]);cx.lineTo(pts[k+1][0],pts[k+1][1]);}}}
 cx.stroke();cx.restore();}

/* ---- overlay: grid, coasts, cities, routes, contours ---- */
function drawOv(){
 /* grid lines */
 cx.strokeStyle='rgba(30,50,70,0.3)';cx.lineWidth=0.5;
 for(var la=20;la<=45;la+=5){
  var p1=g2c(la,lonMin),p2=g2c(la,lonMax);
  cx.beginPath();cx.moveTo(p1[0],p1[1]);cx.lineTo(p2[0],p2[1]);cx.stroke();
  cx.fillStyle='rgba(80,100,120,0.5)';cx.font='9px monospace';
  cx.fillText(la+'\u00b0N',p1[0]+4,p1[1]-3);}
 for(var lo=60;lo<=100;lo+=10){
  var p1=g2c(latMin,lo),p2=g2c(latMax,lo);
  cx.beginPath();cx.moveTo(p1[0],p1[1]);cx.lineTo(p2[0],p2[1]);cx.stroke();
  cx.fillStyle='rgba(80,100,120,0.5)';
  cx.fillText(lo+'\u00b0E',p1[0]+3,p1[1]-4);}

 /* terrain contours */
 drawCtr(D.terrain,1.0,'rgba(100,80,60,0.25)',0.5);
 drawCtr(D.terrain,3.0,'rgba(140,100,70,0.35)',0.8);
 drawCtr(D.terrain,5.0,'rgba(180,130,80,0.45)',1.0);

 /* coastlines */
 cx.strokeStyle='rgba(90,130,160,0.5)';cx.lineWidth=1;
 COAST.forEach(function(c){cx.beginPath();c.forEach(function(pt,k){
  var p=g2c(pt[0],pt[1]);if(k===0)cx.moveTo(p[0],p[1]);else cx.lineTo(p[0],p[1]);});
  cx.stroke();});

 /* routes */
 cx.save();cx.setLineDash([6,4]);cx.lineWidth=0.8;
 ROUTE.forEach(function(r){cx.strokeStyle='rgba(0,170,255,0.22)';cx.beginPath();
  r.p.forEach(function(pt,k){var p=g2c(pt[0],pt[1]);
   if(k===0)cx.moveTo(p[0],p[1]);else cx.lineTo(p[0],p[1]);});cx.stroke();
  var mi=r.p[Math.floor(r.p.length/2)];var pm=g2c(mi[0],mi[1]);
  cx.fillStyle='rgba(0,150,220,0.3)';cx.font='8px monospace';
  cx.fillText(r.n,pm[0]+4,pm[1]-4);});
 cx.restore();

 /* cities */
 CITY.forEach(function(c){
  if(c.la<latMin||c.la>latMax||c.lo<lonMin||c.lo>lonMax)return;
  var p=g2c(c.la,c.lo);
  cx.beginPath();cx.arc(p[0],p[1],2.5,0,Math.PI*2);
  cx.fillStyle='rgba(220,220,230,0.7)';cx.fill();
  cx.fillStyle='rgba(200,210,220,0.55)';cx.font='9px monospace';
  cx.fillText(c.n,p[0]+5,p[1]+3);});

 /* observation marker */
 var ob=g2c(32,80);
 cx.save();cx.strokeStyle='rgba(0,220,136,0.4)';cx.lineWidth=1;
 cx.setLineDash([3,3]);
 cx.beginPath();cx.arc(ob[0],ob[1],8,0,Math.PI*2);cx.stroke();
 cx.setLineDash([]);cx.beginPath();cx.arc(ob[0],ob[1],2,0,Math.PI*2);
 cx.fillStyle='#00dd88';cx.fill();
 cx.fillStyle='rgba(0,220,136,0.5)';cx.font='8px monospace';
 cx.fillText('OBS',ob[0]+12,ob[1]+3);
 cx.restore();

 /* title overlay */
 var fl=getFL();
 cx.fillStyle='rgba(0,220,136,0.55)';cx.font='bold 11px monospace';
 cx.fillText('TURBULENCE INTENSITY \u2014 '+FL+' ('+
  (fl?fl.alt_ft.toLocaleString():'')+' ft)',10,20);
 cx.fillStyle='rgba(100,130,160,0.35)';cx.font='9px monospace';
 cx.fillText('PSTNet Physics-ML Estimate \u00b7 '+D.weather_source+
  ' \u00b7 18\u201346\u00b0N 58\u2013102\u00b0E',10,34);

 /* severity warning */
 if(fl&&fl.stats.severe>0){
  var txt=fl.stats.severe+' SEVERE'+(fl.stats.extreme>0?
   ' / '+fl.stats.extreme+' EXTREME':'')+ ' cells';
  cx.fillStyle='rgba(50,0,0,0.7)';
  var tw2=cx.measureText(txt).width;
  cx.fillRect(cw-tw2-24,4,tw2+16,20);
  cx.strokeStyle='rgba(255,50,50,0.6)';cx.lineWidth=1;
  cx.strokeRect(cw-tw2-24,4,tw2+16,20);
  cx.fillStyle='#ff4444';cx.font='bold 10px monospace';
  cx.fillText(txt,cw-tw2-16,18);}}

function getFL(){if(!D)return null;
 return D.flight_levels.find(function(f){return f.name===FL;});}

/* ---- main render ---- */
function render(){
 if(!D)return;
 cx.fillStyle='#06090f';cx.fillRect(0,0,cw,ch);
 var fl=getFL();
 if(fl){
  if(!hmC[FL])hmC[FL]=mkHM(fl.turb);
  cx.drawImage(hmC[FL],0,0,cw,ch);}
 drawOv();}

/* ---- stats & model panels ---- */
function updStats(){
 var fl=getFL();if(!fl)return;
 var s=fl.stats,vt=gN*gN-s.blocked;
 document.getElementById('stP').innerHTML=
  '<div class="sr"><span class="lb">Mean intensity</span><span class="vl c">'+
   (s.mean*100).toFixed(1)+'%</span></div>'+
  '<div class="sr"><span class="lb">Peak intensity</span><span class="vl '+
   (s.max_val>0.5?'r':'y')+'">'+(s.max_val*100).toFixed(1)+'%</span></div>'+
  '<div class="sr"><span class="lb">Moderate+ cells</span><span class="vl y">'+
   s.moderate+' / '+vt+'</span></div>'+
  '<div class="sr"><span class="lb">Severe+ cells</span><span class="vl r">'+
   s.severe+'</span></div>'+
  '<div class="sr"><span class="lb">Extreme cells</span><span class="vl r">'+
   s.extreme+'</span></div>'+
  '<div class="sr"><span class="lb">Terrain blocked</span><span class="vl">'+
   s.blocked+'</span></div>'+
  '<div class="sr"><span class="lb">Altitude</span><span class="vl g">'+
   fl.alt_km.toFixed(1)+' km</span></div>';
 document.getElementById('mdP').innerHTML=
  '<div class="sr"><span class="lb">Architecture</span><span class="vl">4-regime MoE</span></div>'+
  '<div class="sr"><span class="lb">Parameters</span><span class="vl c">552</span></div>'+
  '<div class="sr"><span class="lb">Loss</span><span class="vl g">'+
   (D.model_loss?D.model_loss.toFixed(6):'\u2014')+'</span></div>'+
  '<div class="sr"><span class="lb">Compute</span><span class="vl">'+
   D.compute_time+'s</span></div>'+
  '<div class="sr"><span class="lb">Weather</span><span class="vl g">'+
   D.weather_source.split(' ')[0]+'</span></div>';}

/* ---- FL buttons ---- */
function buildFL(){
 var g=document.getElementById('flG');g.innerHTML='';
 D.flight_levels.forEach(function(fl){
  var b=document.createElement('button');
  b.className='fl-b'+(fl.name===FL?' on':'');
  b.innerHTML='<div>'+fl.name+'</div><div class="ft">'+
   Math.round(fl.alt_ft/1000)+',000 ft</div>';
  b.onclick=function(){
   FL=fl.name;
   document.querySelectorAll('.fl-b').forEach(function(x){x.classList.remove('on');});
   b.classList.add('on');
   document.getElementById('hFL').textContent=fl.name;
   updStats();render();};
  g.appendChild(b);});}

/* ---- mouse ---- */
function onM(e){
 var r=cv.getBoundingClientRect();
 var mx=(e.clientX-r.left)*(cw/r.width);
 var my=(e.clientY-r.top)*(ch/r.height);
 var geo=c2g(mx,my);
 document.getElementById('fC').textContent=
  geo[0].toFixed(1)+'\u00b0N, '+geo[1].toFixed(1)+'\u00b0E';
 var ter=smp(D.terrain,mx,my);
 document.getElementById('fE').textContent=
  ter>0.1?ter.toFixed(1)+' km':'< 0.1 km';
 var fl=getFL();if(!fl)return;
 var tb=smp(fl.turb,mx,my);
 var tv=document.getElementById('fT');
 if(tb<0){
  tv.textContent='TERRAIN';tv.style.background='#2a1a0a';tv.style.color='#aa7744';
  document.getElementById('fS').textContent='\u2014';
  document.getElementById('fW').textContent='\u2014';return;}
 tv.textContent=(tb*100).toFixed(0)+'%';
 var sev,sc;
 if(tb<0.05){sev='NIL';sc='#112a22';}
 else if(tb<0.20){sev='LIGHT';sc='#1a3a1a';}
 else if(tb<0.40){sev='MODERATE';sc='#3a3a0a';}
 else if(tb<0.60){sev='SEVERE';sc='#3a1a0a';}
 else{sev='EXTREME';sc='#3a0020';}
 tv.style.background=sc;
 tv.style.color=tb<0.20?'#00dd88':(tb<0.40?'#ffaa00':'#ff3344');
 document.getElementById('fS').textContent=sev;
 var wd=smp(fl.wind,mx,my);
 document.getElementById('fW').textContent=wd.toFixed(0)+' m/s';}

function clrM(){
 ['fC','fT','fS','fW','fE'].forEach(function(id){
  document.getElementById(id).textContent='\u2014';});
 var tv=document.getElementById('fT');
 tv.style.background='transparent';tv.style.color='#8899aa';}

/* ---- legend bar ---- */
function mkLeg(){
 var c=document.createElement('canvas');c.width=200;c.height=12;
 var x=c.getContext('2d');
 for(var i=0;i<200;i++){var t=i/200*0.80;var rgb=tClr(t);
  x.fillStyle='rgb('+rgb[0]+','+rgb[1]+','+rgb[2]+')';x.fillRect(i,0,1,12);}
 document.getElementById('lgBar').appendChild(c);}

/* ---- resize ---- */
function resize(){
 var w=document.getElementById('mwrap');
 cw=w.clientWidth;ch=w.clientHeight;
 cv.width=cw;cv.height=ch;
 hmC={};if(D)render();}

/* ---- keyboard ---- */
function onKey(e){
 if(!D)return;
 var idx=parseInt(e.key)-1;
 if(idx>=0&&idx<D.flight_levels.length){
  FL=D.flight_levels[idx].name;
  document.querySelectorAll('.fl-b').forEach(function(b,i){
   b.classList.toggle('on',i===idx);});
  document.getElementById('hFL').textContent=FL;
  updStats();render();}}

/* ---- init ---- */
function init(){
 cv=document.getElementById('cv');cx=cv.getContext('2d');
 resize();
 window.addEventListener('resize',resize);
 cv.addEventListener('mousemove',onM);
 cv.addEventListener('mouseleave',clrM);
 document.addEventListener('keydown',onKey);
 mkLeg();
 fetch('/api/data').then(function(r){return r.json();}).then(function(d){
  D=d;gN=d.grid_n;
  latMin=d.region.lat_min;latMax=d.region.lat_max;
  lonMin=d.region.lon_min;lonMax=d.region.lon_max;
  document.getElementById('ld').style.display='none';
  document.getElementById('hW').textContent='Weather: '+d.weather_source;
  document.getElementById('hW').classList.add('ok');
  document.getElementById('hM').textContent='PSTNet: '+
   (d.model_loss?d.model_loss.toFixed(5):'OK');
  document.getElementById('hM').classList.add('ok');
  buildFL();updStats();
  /* pre-build all heatmaps */
  d.flight_levels.forEach(function(fl){hmC[fl.name]=mkHM(fl.turb);});
  render();
 }).catch(function(e){
  document.querySelector('.ld .msg').textContent='Error: '+e;
  document.querySelector('.ld .msg').style.color='#ff4444';});}

init();
</script></body></html>'''


# =====================================================================
#  Flask Routes
# =====================================================================
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/data')
def api_data():
    if not rmap.computed:
        return jsonify(error='Not computed yet'), 503
    return jsonify(rmap.to_json())


# =====================================================================
#  Main
# =====================================================================
if __name__ == '__main__':
    print('=' * 60)
    print(' PSTNet — AVIATION TURBULENCE MAP')
    print(' Estimating CAT for data-sparse region')
    print('=' * 60)
    rmap.compute()
    print(f' http://127.0.0.1:5890')
    print('=' * 60)
    app.run(debug=False, port=5890, threaded=True)