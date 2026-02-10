# main.py
"""Flask server + full HTML/JS visual — paired corrected vs uncorrected"""

from flask import Flask, render_template_string, jsonify, request
import numpy as np

from config import SCENARIOS, MISSILES, SPEED_OF_SOUND_SEA
from weather_api import WeatherService
from turbulence_model import TurbulenceField
from trajectory import MissileTrajectory

app = Flask(__name__)

weather_service = WeatherService(lat=35.0, lon=-120.0)
turb_field = TurbulenceField(weather_service)

scenarios = {}
running = False


def init_scenarios():
    global scenarios, running
    scenarios = {}
    running = False
    for sid, cfg in SCENARIOS.items():
        seed = abs(hash(sid)) % (2**31)
        scenarios[sid] = dict(
            config=cfg,
            launch_pos=[30.0, 100.0],
            target_pos=[150.0, 100.0],
            launch_alt=cfg['launch_alt'],
            corrected=MissileTrajectory(cfg['missile_type'], turb_field, True, seed),
            uncorrected=MissileTrajectory(cfg['missile_type'], turb_field, False, seed),
            cr=None, ur=None, time=0.0,
        )


# =====================================================================
HTML = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ML Turbulence Correction</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080808;color:#888;font-family:'Courier New',monospace;overflow:hidden}
.hdr{background:#111;padding:10px 18px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #222}
.hdr h1{color:#0f0;font-size:13px;text-transform:uppercase;letter-spacing:2px}
.hdr-i{display:flex;gap:22px;font-size:10px}
.hdr-i .v{color:#0f0}
.ctl{background:#0a0a0a;padding:8px 18px;display:flex;gap:12px;align-items:center;border-bottom:1px solid #1a1a1a}
.btn{padding:8px 16px;border:1px solid #333;background:#111;color:#0f0;cursor:pointer;font-family:inherit;font-size:10px;text-transform:uppercase}
.btn:hover{background:#1a1a1a;border-color:#0f0}
.btn.p{background:#0a3;border-color:#0f0}
.btn.w{background:#530;border-color:#f80;color:#f80}
.dot{width:8px;height:8px;border-radius:50%;background:#333;display:inline-block;margin-right:5px}
.dot.on{background:#0f0;animation:p 1s infinite}
@keyframes p{50%{opacity:.4}}
.main{display:flex;height:calc(100vh - 82px)}
.grid{flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:2px;background:#111}
.pnl{position:relative;overflow:hidden;background:#0a0a0a}
.pnl canvas{width:100%;height:100%}
.ph{position:absolute;top:0;left:0;right:0;padding:7px 12px;background:linear-gradient(180deg,rgba(0,0,0,.92),transparent);z-index:2;pointer-events:none}
.ph .t{font-size:10px;color:#fff}.ph .d{font-size:8px;color:#555}
.sb{width:218px;background:#0a0a0a;border-left:1px solid #222;padding:12px;overflow-y:auto}
.sb h3{font-size:9px;color:#0f0;text-transform:uppercase;margin:12px 0 6px;padding-bottom:4px;border-bottom:1px solid #222}
.sb h3:first-child{margin-top:0}
.tr{display:flex;align-items:center;margin-bottom:3px;font-size:8px}
.tr .a{width:35px;color:#555}
.tr .bg{flex:1;height:10px;background:#1a1a1a;margin:0 6px;border-radius:2px;overflow:hidden}
.tr .br{height:100%;border-radius:2px}
.tr .vl{width:32px;text-align:right;font-size:8px}
.mi{font-size:9px;padding:8px;background:#111;border:1px solid #222;border-radius:3px;margin-bottom:8px}
.mi .mt{color:#f0f;font-weight:bold;margin-bottom:5px}
.mi .mr{display:flex;justify-content:space-between;padding:2px 0}
.mi .mr .l{color:#555}.mi .mr .v{color:#0ff}
.rr{display:grid;grid-template-columns:auto 1fr 1fr 1fr;gap:2px 8px;font-size:9px;margin-bottom:8px}
.rr .rh{color:#555;font-weight:bold;border-bottom:1px solid #222;padding-bottom:2px;margin-bottom:2px}
.lg{font-size:8px}.lg div{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.lg span.c{display:inline-block;width:18px;height:3px;border-radius:2px}
</style></head><body>
<div class="hdr">
 <h1>ML Turbulence Correction — Paired Comparison</h1>
 <div class="hdr-i">
  <div><span>NASA:</span> <span class="v" id="ws">…</span></div>
  <div><span>ML:</span> <span class="v" id="ms">init</span></div>
  <div><span>Time:</span> <span class="v" id="tm">0 s</span></div>
 </div>
</div>
<div class="ctl">
 <button class="btn" id="bw">Update Weather</button>
 <button class="btn p" id="bl">Launch All</button>
 <button class="btn w" id="br">Reset</button>
 <span style="margin-left:auto;font-size:10px"><span class="dot" id="sd"></span><span id="st">Ready</span></span>
</div>
<div class="main">
 <div class="grid">
  <div class="pnl"><canvas id="c-LOW_ALT"></canvas>
   <div class="ph"><div class="t">LOW ALTITUDE — Subsonic M 0.85</div><div class="d">High turbulence — boundary layer</div></div></div>
  <div class="pnl"><canvas id="c-MID_ALT"></canvas>
   <div class="ph"><div class="t">MID ALTITUDE — Supersonic M 2.8</div><div class="d">Moderate turbulence — troposphere</div></div></div>
  <div class="pnl"><canvas id="c-HIGH_ALT"></canvas>
   <div class="ph"><div class="t">HIGH ALTITUDE — High Supersonic M 4.5</div><div class="d">Low turbulence — tropopause</div></div></div>
  <div class="pnl"><canvas id="c-STRAT"></canvas>
   <div class="ph"><div class="t">STRATOSPHERIC — Hypersonic M 8</div><div class="d">Minimal turbulence — stratosphere</div></div></div>
 </div>
 <div class="sb">
  <h3>Turbulence by Altitude</h3><div id="tc"></div>
  <h3>Results</h3><div id="rt" style="font-size:9px;color:#555">Launch to compare…</div>
  <h3>ML Model</h3>
  <div class="mi"><div class="mt">PSTNet (4-regime MoE, ~552 params)</div>
   <div class="mr"><span class="l">Status</span><span class="v" id="ml">…</span></div>
   <div class="mr"><span class="l">Loss</span><span class="v" id="lo">—</span></div></div>
  <h3>NASA Weather</h3>
  <div style="font-size:9px" id="wi"></div>
  <h3>Legend</h3>
  <div class="lg">
   <div><span class="c" style="background:#0ff"></span> Corrected</div>
   <div><span class="c" style="background:#f60"></span> Uncorrected</div>
   <div><span class="c" style="background:#0f0"></span> Launch</div>
   <div><span class="c" style="background:#f00"></span> Target</div>
   <div><span class="c" style="background:linear-gradient(90deg,#060,#aa0,#a00)"></span> Turb intensity</div>
  </div>
 </div>
</div>
<script>
const SIDS=['LOW_ALT','MID_ALT','HIGH_ALT','STRAT'];
let cvs={},ctx={},S=null,run=false;
SIDS.forEach(s=>{cvs[s]=document.getElementById('c-'+s);ctx[s]=cvs[s].getContext('2d')});
function resize(){SIDS.forEach(s=>{let r=cvs[s].parentElement.getBoundingClientRect();cvs[s].width=r.width;cvs[s].height=r.height});if(S)render(S)}
window.addEventListener('resize',resize);resize();

function tCol(t){
 if(t<.2)return`rgb(0,${100+t*700|0},0)`;
 if(t<.4)return`rgb(${(t-.2)*5*255|0},200,0)`;
 return`rgb(220,${200-(t-.4)*333|0},0)`}

function render(s){
 SIDS.forEach(id=>{if(s.scenarios[id])drawScene(id,s.scenarios[id],s.turbulence_profile)});
 updSide(s)}

function drawScene(sid,D,tp){
 const c=ctx[sid],W=cvs[sid].width,H=cvs[sid].height;
 const sH=H*.54|0, dT=sH+3, dH=H*.34|0, sT=dT+dH+3;
 c.fillStyle='#0a0a0a';c.fillRect(0,0,W,H);

 const mA=D.launch_alt*1.25||1;
 const lx=D.launch_pos[0],ly=D.launch_pos[1],tx=D.target_pos[0],ty=D.target_pos[1];
 const pdx=tx-lx,pdy=ty-ly,pl=Math.sqrt(pdx*pdx+pdy*pdy);
 const ux=pdx/pl,uy=pdy/pl;
 const al=p=>(p[0]-lx)*ux+(p[1]-ly)*uy;
 const mD=pl*1.08;
 const sX=d=>(d/mD)*W, sY=a=>sH-8-((a/mA)*(sH-28));

 // turb bands
 if(tp&&tp.length>1){for(let i=0;i<tp.length-1;i++){
  let a1=tp[i].altitude,a2=tp[i+1].altitude;if(a2>mA)continue;
  let t=(tp[i].turbulence+tp[i+1].turbulence)/2;
  let y1=sY(Math.min(a1,mA)),y2=sY(Math.max(a2,0));
  let al2=Math.min(t*.35,.18);
  c.fillStyle=t<.2?`rgba(0,80,0,${al2})`:t<.4?`rgba(140,140,0,${al2})`:`rgba(160,40,0,${al2})`;
  c.fillRect(0,y1,W,y2-y1)}}

 // grid
 c.strokeStyle='#161616';c.lineWidth=1;
 let ag=Math.max(1,Math.ceil(mA/5));
 for(let a=0;a<=mA;a+=ag){c.beginPath();c.moveTo(0,sY(a));c.lineTo(W,sY(a));c.stroke();
  c.fillStyle='#444';c.font='8px monospace';if(a>0)c.fillText(a+'km',2,sY(a)-2)}
 for(let d=0;d<=mD;d+=20){c.beginPath();c.moveTo(sX(d),0);c.lineTo(sX(d),sH);c.stroke()}
 c.strokeStyle='#444';c.lineWidth=2;c.beginPath();c.moveTo(0,sY(0));c.lineTo(W,sY(0));c.stroke();

 function drawT(tr,col,dash){
  if(!tr||tr.length<2)return;c.beginPath();if(dash)c.setLineDash([5,4]);
  c.strokeStyle=col;c.lineWidth=2;
  for(let i=0;i<tr.length;i++){let x=sX(al(tr[i])),y=sY(tr[i][2]);i?c.lineTo(x,y):c.moveTo(x,y)}
  c.stroke();c.setLineDash([])}
 drawT(D.corrected.trajectory,'#00ccff',false);
 drawT(D.uncorrected.trajectory,'#ee6600',true);

 // markers
 c.fillStyle='#0f0';c.beginPath();c.arc(sX(0),sY(D.launch_alt),5,0,7);c.fill();
 c.fillStyle='#f00';c.beginPath();c.arc(sX(pl),sY(0),5,0,7);c.fill();
 c.strokeStyle='#f00';c.lineWidth=1;c.beginPath();c.arc(sX(pl),sY(0),10,0,7);c.stroke();

 // active dots
 function dot(m,col){if(m.status!=='active'||!m.trajectory.length)return;
  let p=m.trajectory[m.trajectory.length-1];c.fillStyle=col;c.beginPath();c.arc(sX(al(p)),sY(p[2]),4,0,7);c.fill()}
 dot(D.corrected,'#0ff');dot(D.uncorrected,'#f60');

 // === deviation view ===
 c.fillStyle='#070707';c.fillRect(0,dT,W,dH);
 c.strokeStyle='#222';c.lineWidth=1;c.beginPath();c.moveTo(0,dT);c.lineTo(W,dT);c.stroke();
 let ct1=D.corrected.cross_track||[], ct2=D.uncorrected.cross_track||[];
 let mx=10;
 ct1.forEach(v=>{if(Math.abs(v)>mx)mx=Math.abs(v)});
 ct2.forEach(v=>{if(Math.abs(v)>mx)mx=Math.abs(v)});
 mx*=1.15;
 let dC=dT+dH/2, dS=(dH/2-4)/mx;
 c.strokeStyle='#333';c.beginPath();c.moveTo(0,dC);c.lineTo(W,dC);c.stroke();
 c.fillStyle='#444';c.font='7px monospace';
 c.fillText('+'+mx.toFixed(0)+'m',2,dT+9);c.fillText('-'+mx.toFixed(0)+'m',2,dT+dH-3);
 c.fillText('Cross-track deviation',W/2-45,dT+9);

 function drawD(d,col,dash){if(!d||d.length<2)return;c.beginPath();if(dash)c.setLineDash([3,3]);
  c.strokeStyle=col;c.lineWidth=1.5;
  for(let i=0;i<d.length;i++){let x=i/(d.length-1)*W,y=dC-d[i]*dS;i?c.lineTo(x,y):c.moveTo(x,y)}
  c.stroke();c.setLineDash([])}
 drawD(ct1,'#00ccff',false);drawD(ct2,'#ee6600',true);

 // === stats ===
 c.fillStyle='#0a0a0a';c.fillRect(0,sT,W,H-sT);
 let cr=D.corrected_result, ur=D.uncorrected_result;
 let cm=cr?cr.miss_distance_m.toFixed(1)+'m':(D.corrected.status==='active'?'flying…':'—');
 let um=ur?ur.miss_distance_m.toFixed(1)+'m':(D.uncorrected.status==='active'?'flying…':'—');
 let imp='';
 if(cr&&ur&&ur.miss_distance_m>0)imp='\u2193'+((1-cr.miss_distance_m/ur.miss_distance_m)*100).toFixed(0)+'%';
 c.font='10px monospace';
 c.fillStyle='#0cf';c.fillText('Corr: '+cm,8,sT+13);
 c.fillStyle='#e60';c.fillText('Uncorr: '+um,W*.33,sT+13);
 c.fillStyle='#0f0';c.fillText(imp,W*.68,sT+13);
 if(cr){c.fillStyle='#aa0';c.fillText('Turb:'+(cr.avg_turbulence*100).toFixed(0)+'%',W*.83,sT+13)}
}

function updSide(s){
 // turb chart
 let tp=s.turbulence_profile||[],h='';
 tp.forEach(i=>{let p=Math.min(i.turbulence*100,100),cl=tCol(i.turbulence);
  h+='<div class="tr"><span class="a">'+i.altitude+'km</span><div class="bg"><div class="br" style="width:'+p+'%;background:'+cl+'"></div></div><span class="vl" style="color:'+cl+'">'+p.toFixed(0)+'%</span></div>'});
 document.getElementById('tc').innerHTML=h;

 // results
 let rh='',allDone=true;
 SIDS.forEach(id=>{let sc=s.scenarios[id];if(!sc||!sc.complete){allDone=false;return}
  let cm=sc.corrected_result.miss_distance_m,um=sc.uncorrected_result.miss_distance_m;
  let imp=um>0?((1-cm/um)*100).toFixed(0):'—';
  rh+='<span style="color:#aaa">'+sc.config.name.split('(')[0].trim()+'</span><span style="color:#0ff">'+cm.toFixed(1)+'m</span><span style="color:#f60">'+um.toFixed(1)+'m</span><span style="color:#0f0">\u2193'+imp+'%</span>'});
 if(allDone)document.getElementById('rt').innerHTML='<div class="rr"><span class="rh">Scenario</span><span class="rh" style="color:#0ff">Corr</span><span class="rh" style="color:#f60">Uncorr</span><span class="rh" style="color:#0f0">Impr</span>'+rh+'</div>';

 // model
 let mi=s.model_info||{};
 document.getElementById('ml').textContent=mi.trained?'Ready':'Training…';
 if(mi.final_loss!=null)document.getElementById('lo').textContent=mi.final_loss.toFixed(6);
}

// ---- API calls ----
async function fetchW(){
 document.getElementById('ws').textContent='fetching…';
 try{let r=await(await fetch('/api/weather')).json();
  document.getElementById('ws').textContent='OK';
  document.getElementById('ms').textContent='trained';
  let w=r.weather;
  document.getElementById('wi').innerHTML=
   '<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #1a1a1a"><span style="color:#555">Wind 10 m</span><span style="color:#0f0">'+w.wind_speed_10m.toFixed(1)+' m/s</span></div>'+
   '<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #1a1a1a"><span style="color:#555">Temp</span><span style="color:#0f0">'+(w.temperature_2m-273.15).toFixed(1)+' °C</span></div>'+
   '<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #1a1a1a"><span style="color:#555">Pressure</span><span style="color:#0f0">'+w.pressure_surface.toFixed(1)+' kPa</span></div>'+
   '<div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:#555">Humidity</span><span style="color:#0f0">'+w.humidity.toFixed(0)+'%</span></div>';
  await refresh();
 }catch(e){document.getElementById('ws').textContent='error';console.error(e)}}

async function refresh(){
 try{S=await(await fetch('/api/state')).json();
  let mt=0;SIDS.forEach(id=>{if(S.scenarios[id])mt=Math.max(mt,S.scenarios[id].time)});
  document.getElementById('tm').textContent=mt.toFixed(1)+'s';
  render(S);
  let sd=document.getElementById('sd'),st=document.getElementById('st');
  if(S.running){sd.classList.add('on');st.textContent='Simulating…'}
  else{sd.classList.remove('on');st.textContent=SIDS.every(id=>S.scenarios[id]&&S.scenarios[id].complete)?'Complete':'Ready'}
 }catch(e){console.error(e)}}

document.getElementById('bw').onclick=fetchW;
document.getElementById('bl').onclick=async()=>{await fetch('/api/launch',{method:'POST'});run=true;loop()};
document.getElementById('br').onclick=async()=>{run=false;await fetch('/api/reset',{method:'POST'});await refresh()};

async function loop(){
 if(!run)return;
 await fetch('/api/step',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
 await refresh();
 if(!S.running){run=false;return}
 setTimeout(loop,35)}

fetchW();
</script></body></html>'''


# =====================================================================
# Routes
# =====================================================================
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/weather')
def api_weather():
    w = weather_service.get_weather()
    turb_field.update()
    init_scenarios()
    return jsonify(weather=w,
                   turbulence_profile=turb_field.get_altitude_turbulence_profile(),
                   model_info=turb_field.get_model_info())


@app.route('/api/state')
def api_state():
    def mdata(m):
        n = len(m.trajectory)
        step = max(1, n // 300)
        idx = list(range(0, n, step))
        if n > 0 and (n - 1) not in idx:
            idx.append(n - 1)
        pick = lambda arr: [float(arr[i]) for i in idx if i < len(arr)]
        traj = [[float(m.trajectory[i][0]), float(m.trajectory[i][1]),
                  float(m.trajectory[i][2])] for i in idx if i < n]
        return dict(trajectory=traj, cross_track=pick(m.cross_track_log),
                    altitude=pick(m.altitude_log), turbulence=pick(m.turbulence_log),
                    correction=pick(m.correction_log),
                    position=([float(m.position[0]), float(m.position[1]),
                               float(m.position[2])] if m.position is not None else [0, 0, 0]),
                    status='impact' if m.impact_error is not None else 'active')

    out = dict(scenarios={}, running=running,
               turbulence_profile=turb_field.get_altitude_turbulence_profile(),
               model_info=turb_field.get_model_info())
    for sid, sc in scenarios.items():
        out['scenarios'][sid] = dict(
            config=sc['config'], name=sc['config']['name'],
            launch_pos=sc['launch_pos'], target_pos=sc['target_pos'],
            launch_alt=sc['launch_alt'], time=sc['time'],
            corrected=mdata(sc['corrected']),
            uncorrected=mdata(sc['uncorrected']),
            corrected_result=sc['cr'], uncorrected_result=sc['ur'],
            complete=sc['cr'] is not None and sc['ur'] is not None)
    return jsonify(out)


@app.route('/api/launch', methods=['POST'])
def api_launch():
    global running
    for sc in scenarios.values():
        sc['corrected'].launch(sc['launch_pos'], sc['target_pos'], sc['launch_alt'])
        sc['uncorrected'].launch(sc['launch_pos'], sc['target_pos'], sc['launch_alt'])
    running = True
    return jsonify(success=True)


@app.route('/api/step', methods=['POST'])
def api_step():
    global running
    SIM_ADVANCE = 1.5                             # seconds of sim time per call
    for sid, sc in scenarios.items():
        speed = MISSILES[sc['config']['missile_type']]['speed'] * SPEED_OF_SOUND_SEA
        dt = min(0.5, 0.12 / speed)               # ~120 m per step regardless of Mach
        n_sub = max(3, int(SIM_ADVANCE / dt))
        for _ in range(n_sub):
            if sc['cr'] is None:
                r = sc['corrected'].step(dt)
                if r and r['status'] == 'impact':
                    sc['cr'] = r
            if sc['ur'] is None:
                r = sc['uncorrected'].step(dt)
                if r and r['status'] == 'impact':
                    sc['ur'] = r
            sc['time'] += dt
    if all(sc['cr'] is not None and sc['ur'] is not None for sc in scenarios.values()):
        running = False
    return jsonify(success=True)


@app.route('/api/reset', methods=['POST'])
def api_reset():
    init_scenarios()
    return jsonify(success=True)


if __name__ == '__main__':
    print('=' * 60)
    print(' ML TURBULENCE CORRECTION — PAIRED COMPARISON')
    print('=' * 60)
    turb_field.update()
    init_scenarios()
    print(f' http://127.0.0.1:5890')
    print('=' * 60)
    app.run(debug=False, port=5890, threaded=True)