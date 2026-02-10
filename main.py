# main.py
"""Flask server + 3D Three.js visual — paired corrected vs uncorrected"""

from flask import Flask, render_template_string, jsonify, request
import numpy as np

from config import MISSILES, SPEED_OF_SOUND_SEA
from weather_api import WeatherService
from turbulence_model import TurbulenceField
from trajectory import MissileTrajectory

app = Flask(__name__)

weather_service = WeatherService(lat=35.0, lon=-120.0)
turb_field = TurbulenceField(weather_service)

# ── Simulation state ────────────────────────────────────────────────
sim = dict(
    missile_type='SUPERSONIC', launch_alt=10.0, target_range=120.0,
    launch_pos=[30.0, 100.0], target_pos=[150.0, 100.0],
    corrected=None, uncorrected=None, cr=None, ur=None,
    time=0.0, running=False,
)


def setup(mt='SUPERSONIC', alt=10.0, rng=120.0):
    seed = np.random.randint(0, 2**31)
    sim.update(
        missile_type=mt, launch_alt=alt, target_range=rng,
        launch_pos=[30.0, 100.0], target_pos=[30.0 + rng, 100.0],
        corrected=MissileTrajectory(mt, turb_field, True, seed),
        uncorrected=MissileTrajectory(mt, turb_field, False, seed),
        cr=None, ur=None, time=0.0, running=False,
    )


# =====================================================================
# HTML + Three.js 3D Visualisation
# =====================================================================
HTML = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSTNet 3D Trajectory</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;overflow:hidden;font-family:'Courier New',monospace;color:#888}
#c3d{position:fixed;top:0;left:0;width:100vw;height:100vh}
.hdr{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(0,0,0,.88);
  border-bottom:1px solid #1a1a1a;padding:7px 16px;display:flex;
  justify-content:space-between;align-items:center}
.hdr h1{color:#0f0;font-size:11px;text-transform:uppercase;letter-spacing:2px}
.hdr-r{display:flex;gap:18px;font-size:10px}
.hdr-r .v{color:#0f0}
.dot{width:7px;height:7px;border-radius:50%;background:#333;
  display:inline-block;margin-right:4px}
.dot.on{background:#0f0;animation:bk 1s infinite}
@keyframes bk{50%{opacity:.3}}
.cp{position:fixed;top:42px;left:10px;z-index:100;
  background:rgba(4,8,14,.93);border:1px solid #162016;border-radius:4px;
  padding:12px;width:196px;max-height:calc(100vh - 55px);overflow-y:auto}
.cp::-webkit-scrollbar{width:3px}
.cp::-webkit-scrollbar-thumb{background:#1a2a1a}
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
  cursor:pointer;letter-spacing:1px;transition:all .15s}
.btn:hover{background:#1a1a1a;border-color:#0f0}
.btn:disabled{opacity:.4;cursor:default}
.btn.go{background:#0a2a0a;border-color:#0a0}
.btn.rst{background:#2a1a0a;border-color:#f80;color:#f80}
.ta{font-size:11px;color:#aa0;padding:3px 0}
.tr{display:flex;align-items:center;margin-bottom:2px;font-size:7px}
.tr .a{width:30px;color:#555}
.tr .bg{flex:1;height:7px;background:#111;margin:0 4px;border-radius:2px;overflow:hidden}
.tr .fl{height:100%;border-radius:2px;transition:width .3s}
.tr .vl{width:26px;text-align:right}
.sp{position:fixed;top:42px;right:10px;z-index:100;
  background:rgba(4,8,14,.93);border:1px solid #162016;border-radius:4px;
  padding:14px;width:230px;display:none}
.sp.show{display:block}
.sp h3{font-size:9px;color:#0f0;text-transform:uppercase;margin-bottom:8px;
  padding-bottom:3px;border-bottom:1px solid #162016}
.rw{display:flex;justify-content:space-between;padding:3px 0;font-size:10px;
  border-bottom:1px solid #0a0a0a}
.rw .lb{color:#555}
.rw .vv{font-weight:bold}
.big{text-align:center;padding:10px;margin-top:8px;border:1px solid #162016;
  border-radius:3px;background:rgba(0,0,0,.3)}
.big .n{font-size:20px;font-weight:bold}
.big .l{font-size:8px;color:#555;margin-top:2px}
.lg{position:fixed;bottom:10px;left:10px;z-index:100;
  background:rgba(4,8,14,.85);border:1px solid #162016;border-radius:4px;
  padding:6px 12px;font-size:9px;display:flex;gap:14px}
.lg .li{display:flex;align-items:center;gap:4px}
.lg .lc{width:16px;height:3px;border-radius:2px}
.inf{position:fixed;bottom:10px;right:10px;z-index:100;
  background:rgba(4,8,14,.8);border:1px solid #162016;border-radius:4px;
  padding:5px 10px;font-size:8px;color:#444}
</style></head><body>

<canvas id="c3d"></canvas>

<div class="hdr">
 <h1>PSTNet &mdash; 3D Trajectory</h1>
 <div class="hdr-r">
  <div>Weather: <span class="v" id="hW">&hellip;</span></div>
  <div>ML: <span class="v" id="hM">init</span></div>
  <div>Time: <span class="v" id="hT">0.0 s</span></div>
  <div><span class="dot" id="hD"></span><span id="hS">Ready</span></div>
 </div>
</div>

<div class="cp">
 <div class="sc">Missile Type</div>
 <div class="rg" id="tG">
  <label><input type="radio" name="mt" value="SUBSONIC"> Subsonic M 0.85</label>
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
 <button class="btn go" id="bL">&#9654; Launch</button>
 <button class="btn rst" id="bR">&#8635; Reset</button>
 <button class="btn" id="bW" style="margin-top:3px;font-size:8px">Update Weather</button>
 <div class="sc">Turbulence Profile</div>
 <div id="tP"></div>
 <div class="sc">Model Info</div>
 <div style="font-size:9px">
  <div style="color:#f0f;margin-bottom:3px">PSTNet (4-regime MoE)</div>
  <div style="display:flex;justify-content:space-between"><span style="color:#555">Status</span><span class="v" id="mS">&mdash;</span></div>
  <div style="display:flex;justify-content:space-between"><span style="color:#555">Loss</span><span style="color:#0ff" id="mL">&mdash;</span></div>
  <div style="display:flex;justify-content:space-between"><span style="color:#555">Params</span><span style="color:#0ff">552</span></div>
 </div>
</div>

<div class="sp" id="sP">
 <h3>Impact Results</h3>
 <div id="sR"></div>
</div>

<div class="lg">
 <div class="li"><div class="lc" style="background:#0cf"></div> Corrected (ML)</div>
 <div class="li"><div class="lc" style="background:#f60"></div> Uncorrected</div>
 <div class="li"><div class="lc" style="background:#0f0"></div> Launch</div>
 <div class="li"><div class="lc" style="background:#f00"></div> Target / Ship</div>
</div>

<div class="inf">Alt 3x exaggerated &middot; Scroll zoom &middot; Drag orbit</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
/* =============================================================
   PSTNet — 3D Trajectory Visualisation
   ============================================================= */
var VS=3, CX=90, CZ=100;
var MDEF={
 SUBSONIC:{aMin:0.5,aMax:8,aDef:3,rDef:120},
 SUPERSONIC:{aMin:2,aMax:15,aDef:10,rDef:120},
 HIGH_SUPERSONIC:{aMin:8,aMax:25,aDef:18,rDef:120},
 HYPERSONIC:{aMin:15,aMax:35,aDef:25,rDef:150}
};

var scene,cam,ren,ctl,clk;
var oG,oM,sGrp,lMk,tMk;
var mC,mU,lC,lU,aC,aU;
var expl=[],shkI=0,simT0=0;
var running=false,cD=false,uD=false;
var tProf=[];
var cfg={missile_type:'SUPERSONIC',launch_alt:10,target_range:120};

function tw(x,y,a){return new THREE.Vector3(x-CX,a*VS,-(y-CZ))}

/* ---- INIT ---- */
function init(){
 clk=new THREE.Clock();
 scene=new THREE.Scene();
 scene.background=new THREE.Color(0x000811);
 scene.fog=new THREE.FogExp2(0x000811,0.0012);

 cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.1,3000);
 cam.position.set(0,40,80);

 ren=new THREE.WebGLRenderer({canvas:document.getElementById('c3d'),antialias:true});
 ren.setSize(innerWidth,innerHeight);
 ren.setPixelRatio(Math.min(devicePixelRatio,2));

 ctl=new THREE.OrbitControls(cam,ren.domElement);
 ctl.enableDamping=true;ctl.dampingFactor=0.06;
 ctl.maxPolarAngle=Math.PI/2-0.02;
 ctl.target.set(0,10,0);

 scene.add(new THREE.AmbientLight(0x112244,0.5));
 var sun=new THREE.DirectionalLight(0xffeedd,0.6);
 sun.position.set(60,100,40);scene.add(sun);
 scene.add(new THREE.HemisphereLight(0x001144,0x000000,0.25));

 mkOcean();mkGrid();mkStars();mkMoon();mkShip();
 mC=mkMsl(0x00ccff);mU=mkMsl(0xee6600);
 mC.visible=mU.visible=false;scene.add(mC);scene.add(mU);

 lC=mkLn(0x00ccff,0.85);lU=mkLn(0xee6600,0.55);
 scene.add(lC);scene.add(lU);
 aC=mkLn(0x00ccff,0.15);aU=mkLn(0xee6600,0.15);
 aC.visible=aU.visible=false;scene.add(aC);scene.add(aU);

 mkMarkers();

 window.addEventListener('resize',function(){
  cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();
  ren.setSize(innerWidth,innerHeight);
 });
}

function mkOcean(){
 oG=new THREE.PlaneGeometry(3000,3000,60,60);
 oG.rotateX(-Math.PI/2);
 oM=new THREE.Mesh(oG,new THREE.MeshPhongMaterial({
  color:0x001828,specular:0x002233,shininess:25,
  transparent:true,opacity:0.92}));
 oM.position.y=-0.15;scene.add(oM);
}
function mkGrid(){
 var g=new THREE.GridHelper(400,40,0x002800,0x001000);
 g.position.y=0.01;scene.add(g);
}
function mkStars(){
 var n=2000,p=new Float32Array(n*3);
 for(var i=0;i<n;i++){p[i*3]=(Math.random()-0.5)*2000;
  p[i*3+1]=50+Math.random()*400;p[i*3+2]=(Math.random()-0.5)*2000;}
 var g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(p,3));
 scene.add(new THREE.Points(g,new THREE.PointsMaterial({
  color:0xffffff,size:0.3,transparent:true,opacity:0.45})));
}
function mkMoon(){
 var m=new THREE.Mesh(new THREE.SphereGeometry(4,16,16),
  new THREE.MeshBasicMaterial({color:0xddeeff}));
 m.position.set(250,180,-400);scene.add(m);
}

/* ---- SHIP ---- */
function mkShip(){
 sGrp=new THREE.Group();
 var m1=new THREE.MeshPhongMaterial({color:0x3a4a5a});
 var m2=new THREE.MeshPhongMaterial({color:0x4a5a6a});
 var m3=new THREE.MeshPhongMaterial({color:0x222a32});
 function bx(w,h,d,mt,px,py,pz){
  var me=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mt);
  me.position.set(px||0,py||0,pz||0);return me;}
 function cy(rt,rb,h,mt,px,py,pz){
  var me=new THREE.Mesh(new THREE.CylinderGeometry(rt,rb,h,8),mt);
  me.position.set(px||0,py||0,pz||0);return me;}

 sGrp.add(bx(5,.8,1.3,m1,0,.4,0));
 var bow=new THREE.Mesh(new THREE.ConeGeometry(.65,2,4),m1);
 bow.position.set(3.5,.4,0);bow.rotation.z=-Math.PI/2;sGrp.add(bow);
 sGrp.add(bx(2.2,1.2,.95,m2,-.3,1.4,0));
 sGrp.add(bx(1.2,.7,.7,m2,-.3,2.3,0));
 sGrp.add(cy(.15,.22,.65,m3,.4,1.9,0));
 sGrp.add(cy(.03,.03,2.2,m2,-.3,3.6,0));
 sGrp.add(bx(.5,.25,.04,m2,-.3,4.5,0));
 var rl=new THREE.PointLight(0xff0000,.4,5);
 rl.position.set(-.3,4.8,0);sGrp.add(rl);

 /* small navigation lights */
 var gl=new THREE.Mesh(new THREE.SphereGeometry(.06,6,6),
  new THREE.MeshBasicMaterial({color:0x00ff00}));
 gl.position.set(-2.3,.9,.65);sGrp.add(gl);
 var rl2=new THREE.Mesh(new THREE.SphereGeometry(.06,6,6),
  new THREE.MeshBasicMaterial({color:0xff0000}));
 rl2.position.set(-2.3,.9,-.65);sGrp.add(rl2);

 updShip();scene.add(sGrp);
}
function updShip(){
 var p=tw(30+cfg.target_range,100,0);
 sGrp.position.set(p.x,0,p.z);
 sGrp.rotation.y=Math.PI*.7;
}

/* ---- MISSILE MESH ---- */
function mkMsl(color){
 var g=new THREE.Group();
 var mt=new THREE.MeshPhongMaterial({color:color,emissive:color,emissiveIntensity:.3});
 var bd=new THREE.Mesh(new THREE.CylinderGeometry(.12,.12,1.8,8),mt);
 bd.rotation.x=Math.PI/2;g.add(bd);
 var ns=new THREE.Mesh(new THREE.ConeGeometry(.12,.5,8),mt);
 ns.rotation.x=-Math.PI/2;ns.position.z=-1.15;g.add(ns);
 var fm=new THREE.MeshPhongMaterial({color:0x444444});
 for(var i=0;i<4;i++){
  var f=new THREE.Mesh(new THREE.BoxGeometry(.35,.02,.25),fm);
  f.position.z=.7;var w=new THREE.Group();w.add(f);
  w.rotation.z=(Math.PI/2)*i;g.add(w);}
 var gl=new THREE.Mesh(new THREE.SphereGeometry(.16,8,8),
  new THREE.MeshBasicMaterial({color:0xff4400,transparent:true,opacity:.7}));
 gl.position.z=1.05;g.add(gl);
 var pl=new THREE.PointLight(color,.8,6);pl.position.z=1;g.add(pl);
 return g;
}

/* ---- LINE HELPERS ---- */
function mkLn(color,op){
 return new THREE.Line(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({color:color,transparent:true,opacity:op||.8}));
}
function setLn(ln,pts){
 if(!pts||pts.length<2)return;
 var a=new Float32Array(pts.length*3);
 for(var i=0;i<pts.length;i++){
  var w=tw(pts[i][0],pts[i][1],pts[i][2]);
  a[i*3]=w.x;a[i*3+1]=w.y;a[i*3+2]=w.z;}
 var ng=new THREE.BufferGeometry();
 ng.setAttribute('position',new THREE.BufferAttribute(a,3));
 var og=ln.geometry;ln.geometry=ng;og.dispose();
}
function setAl(ln,wp){
 if(!wp)return;
 var a=new Float32Array([wp.x,wp.y,wp.z,wp.x,0,wp.z]);
 var ng=new THREE.BufferGeometry();
 ng.setAttribute('position',new THREE.BufferAttribute(a,3));
 var og=ln.geometry;ln.geometry=ng;og.dispose();
}
function oriM(m,tr){
 var n=tr.length;if(n<2)return;
 var a=tw(tr[n-2][0],tr[n-2][1],tr[n-2][2]);
 var b=tw(tr[n-1][0],tr[n-1][1],tr[n-1][2]);
 var d=b.clone().sub(a).normalize();
 m.lookAt(m.position.clone().add(d));
}

/* ---- MARKERS ---- */
function mkMarkers(){
 lMk=new THREE.Group();
 var lr=new THREE.Mesh(
  new THREE.RingGeometry(1.5,2,32).rotateX(-Math.PI/2),
  new THREE.MeshBasicMaterial({color:0x00ff00,side:THREE.DoubleSide,
   transparent:true,opacity:.35}));
 lMk.add(lr);
 var lv=new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([
   new THREE.Vector3(0,0,0),new THREE.Vector3(0,4,0)]),
  new THREE.LineBasicMaterial({color:0x00ff00,transparent:true,opacity:.2}));
 lMk.add(lv);
 var lp=tw(30,100,0);lMk.position.set(lp.x,.05,lp.z);
 scene.add(lMk);

 tMk=new THREE.Group();
 var tr=new THREE.Mesh(
  new THREE.RingGeometry(1.5,2,32).rotateX(-Math.PI/2),
  new THREE.MeshBasicMaterial({color:0xff0000,side:THREE.DoubleSide,
   transparent:true,opacity:.3}));
 tMk.add(tr);
 var t1=new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([
   new THREE.Vector3(-2,0,0),new THREE.Vector3(2,0,0)]),
  new THREE.LineBasicMaterial({color:0xff0000,transparent:true,opacity:.2}));
 var t2=new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([
   new THREE.Vector3(0,0,-2),new THREE.Vector3(0,0,2)]),
  new THREE.LineBasicMaterial({color:0xff0000,transparent:true,opacity:.2}));
 tMk.add(t1);tMk.add(t2);
 var tp=tw(30+cfg.target_range,100,0);tMk.position.set(tp.x,.05,tp.z);
 scene.add(tMk);
}

/* ---- EXPLOSION ---- */
function boom(pos,color){
 var N=300,P=new Float32Array(N*3),C=new Float32Array(N*3),V=[];
 for(var i=0;i<N;i++){
  P[i*3]=pos.x;P[i*3+1]=pos.y;P[i*3+2]=pos.z;
  var th=Math.random()*6.283,ph=Math.acos(2*Math.random()-1);
  var sp=1+Math.random()*5;
  V.push(Math.sin(ph)*Math.cos(th)*sp,
         Math.sin(ph)*Math.sin(th)*sp*0.5+2.5,
         Math.cos(ph)*sp);
  var mx=Math.random();
  C[i*3]=1;C[i*3+1]=.3+mx*.7;C[i*3+2]=mx*.2;
 }
 var g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(P,3));
 g.setAttribute('color',new THREE.BufferAttribute(C,3));
 var pts=new THREE.Points(g,new THREE.PointsMaterial({
  size:.45,vertexColors:true,transparent:true,opacity:1,
  blending:THREE.AdditiveBlending,depthWrite:false}));
 scene.add(pts);
 var fl=new THREE.PointLight(0xffaa44,15,80);
 fl.position.copy(pos);scene.add(fl);
 var sp2=new THREE.Mesh(new THREE.SphereGeometry(.8,12,12),
  new THREE.MeshBasicMaterial({color:0xffcc00,transparent:true,opacity:.85}));
 sp2.position.copy(pos);scene.add(sp2);

 /* debris ring on water */
 var rg=new THREE.Mesh(
  new THREE.RingGeometry(.5,1.5,24).rotateX(-Math.PI/2),
  new THREE.MeshBasicMaterial({color:color,transparent:true,
   opacity:.6,side:THREE.DoubleSide}));
 rg.position.set(pos.x,.1,pos.z);scene.add(rg);

 expl.push({pts:pts,V:V,age:0,mx:3.5,fl:fl,sp:sp2,rg:rg});
 shkI=1.5;
}
function updExpl(dt){
 for(var i=expl.length-1;i>=0;i--){
  var e=expl[i];e.age+=dt;
  if(e.age>e.mx){
   scene.remove(e.pts);scene.remove(e.fl);scene.remove(e.sp);scene.remove(e.rg);
   e.pts.geometry.dispose();e.pts.material.dispose();
   e.sp.geometry.dispose();e.sp.material.dispose();
   e.rg.geometry.dispose();e.rg.material.dispose();
   expl.splice(i,1);continue;}
  var pa=e.pts.geometry.attributes.position;
  for(var j=0;j<pa.count;j++){
   pa.array[j*3]+=e.V[j*3]*dt;
   pa.array[j*3+1]+=e.V[j*3+1]*dt;
   pa.array[j*3+2]+=e.V[j*3+2]*dt;
   e.V[j*3+1]-=5*dt;}
  pa.needsUpdate=true;
  e.pts.material.opacity=Math.max(0,1-e.age/e.mx);
  e.fl.intensity=Math.max(0,15*(1-e.age/.3));
  var s=1+e.age*4;e.sp.scale.set(s,s,s);
  e.sp.material.opacity=Math.max(0,.85*(1-e.age/.5));
  var rs=1+e.age*6;e.rg.scale.set(rs,1,rs);
  e.rg.material.opacity=Math.max(0,.6*(1-e.age/2));
 }
}

/* ---- RENDER LOOP ---- */
function rLoop(){
 requestAnimationFrame(rLoop);
 var dt=clk.getDelta(),t=clk.getElapsedTime();
 ctl.update();

 var op=oG.attributes.position;
 for(var i=0;i<op.count;i++){
  op.setY(i,Math.sin(op.getX(i)*.015+t*.4)*.1+
             Math.cos(op.getZ(i)*.02+t*.3)*.07);}
 op.needsUpdate=true;

 if(sGrp){sGrp.position.y=Math.sin(t*.5)*.06;
  sGrp.rotation.x=Math.sin(t*.3)*.006;}

 updExpl(dt);

 if(shkI>.01){
  scene.position.set((Math.random()-.5)*shkI,
   (Math.random()-.5)*shkI*.5,(Math.random()-.5)*shkI);
  shkI*=.92;
 }else{scene.position.set(0,0,0);shkI=0;}

 ren.render(scene,cam);
}

/* =============== API / SIMULATION =============== */
function fetchW(){
 document.getElementById('hW').textContent='...';
 fetch('/api/weather').then(function(r){return r.json()}).then(function(r){
  document.getElementById('hW').textContent='OK';
  document.getElementById('hM').textContent='ready';
  tProf=r.turbulence_profile||[];
  updTurbUI();updTurbAt();
  if(r.model_info){
   document.getElementById('mS').textContent=r.model_info.trained?'Trained':'...';
   if(r.model_info.final_loss!=null)
    document.getElementById('mL').textContent=r.model_info.final_loss.toFixed(6);}
 }).catch(function(e){document.getElementById('hW').textContent='err';console.error(e);});
}

function updTurbUI(){
 var h='';
 tProf.forEach(function(p){
  var v=Math.min(p.turbulence*100,100);
  var c=v<20?'#0a0':v<40?'#aa0':'#c40';
  h+='<div class="tr"><span class="a">'+p.altitude+'</span>'+
   '<div class="bg"><div class="fl" style="width:'+v+'%;background:'+c+'"></div></div>'+
   '<span class="vl" style="color:'+c+'">'+v.toFixed(0)+'%</span></div>';});
 document.getElementById('tP').innerHTML=h;
}

function updTurbAt(){
 var alt=parseFloat(document.getElementById('aS').value);
 var turb=0;
 if(tProf.length){
  var best=tProf[0];
  for(var i=0;i<tProf.length;i++)
   if(Math.abs(tProf[i].altitude-alt)<Math.abs(best.altitude-alt))best=tProf[i];
  turb=best.turbulence;}
 var v=(turb*100).toFixed(0);
 var c=turb<.2?'#0a0':turb<.4?'#aa0':'#c40';
 document.getElementById('tA').innerHTML=
  '<span style="color:'+c+';font-weight:bold">'+v+'%</span>'+
  ' <span style="color:#555">at '+alt.toFixed(1)+' km</span>';
}

function cfgSim(){
 cfg.missile_type=document.querySelector('input[name="mt"]:checked').value;
 cfg.launch_alt=parseFloat(document.getElementById('aS').value);
 cfg.target_range=parseFloat(document.getElementById('rS').value);
 return fetch('/api/configure',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify(cfg)});
}

function doLaunch(){
 document.getElementById('bL').disabled=true;
 cfgSim().then(function(){
  cD=uD=false;
  mC.visible=mU.visible=true;
  aC.visible=aU.visible=true;

  [lC,lU,aC,aU].forEach(function(l){
   var g=l.geometry;l.geometry=new THREE.BufferGeometry();g.dispose();});
  expl.forEach(function(e){scene.remove(e.pts);scene.remove(e.fl);
   scene.remove(e.sp);scene.remove(e.rg);});
  expl=[];
  document.getElementById('sP').classList.remove('show');

  updShip();
  var tp=tw(30+cfg.target_range,100,0);tMk.position.set(tp.x,.05,tp.z);

  var r2=cfg.target_range,a2=cfg.launch_alt;
  cam.position.set(0,Math.max(25,a2*VS*1.1),r2*.55);
  ctl.target.set(0,a2*VS*.25,0);ctl.update();

  return fetch('/api/launch',{method:'POST'});
 }).then(function(){
  running=true;simT0=Date.now();
  document.getElementById('hS').textContent='Simulating...';
  document.getElementById('hD').classList.add('on');
  simLoop();
 }).catch(function(e){
  console.error(e);document.getElementById('bL').disabled=false;});
}

function simLoop(){
 if(!running)return;
 if(Date.now()-simT0>90000){
  running=false;
  document.getElementById('hS').textContent='Timeout';
  document.getElementById('hD').classList.remove('on');
  document.getElementById('bL').disabled=false;return;}
 fetch('/api/step',{method:'POST'})
 .then(function(){return fetch('/api/state')})
 .then(function(r){return r.json()})
 .then(function(s){
  if(s.corrected.trajectory.length>1)setLn(lC,s.corrected.trajectory);
  if(s.uncorrected.trajectory.length>1)setLn(lU,s.uncorrected.trajectory);

  if(s.corrected.status==='active'&&s.corrected.trajectory.length>0){
   var l=s.corrected.trajectory[s.corrected.trajectory.length-1];
   var w=tw(l[0],l[1],l[2]);mC.position.copy(w);
   oriM(mC,s.corrected.trajectory);setAl(aC,w);aC.visible=true;}
  if(s.uncorrected.status==='active'&&s.uncorrected.trajectory.length>0){
   var l2=s.uncorrected.trajectory[s.uncorrected.trajectory.length-1];
   var w2=tw(l2[0],l2[1],l2[2]);mU.position.copy(w2);
   oriM(mU,s.uncorrected.trajectory);setAl(aU,w2);aU.visible=true;}

  if(s.corrected.status==='impact'&&!cD){cD=true;
   var lx=s.corrected.trajectory[s.corrected.trajectory.length-1];
   boom(tw(lx[0],lx[1],lx[2]),0x00ccff);mC.visible=false;aC.visible=false;}
  if(s.uncorrected.status==='impact'&&!uD){uD=true;
   var lx2=s.uncorrected.trajectory[s.uncorrected.trajectory.length-1];
   boom(tw(lx2[0],lx2[1],lx2[2]),0xff6600);mU.visible=false;aU.visible=false;}

  document.getElementById('hT').textContent=s.time.toFixed(1)+' s';

  if(!s.running){running=false;
   document.getElementById('hS').textContent='Complete';
   document.getElementById('hD').classList.remove('on');
   document.getElementById('bL').disabled=false;
   showRes(s);return;}
  setTimeout(simLoop,30);
 }).catch(function(e){console.error(e);running=false;
  document.getElementById('bL').disabled=false;});
}

function showRes(s){
 var cr=s.corrected_result,ur=s.uncorrected_result;
 if(!cr||!ur)return;
 var cm=cr.miss_distance_m,um=ur.miss_distance_m;
 var imp=um>0?((1-cm/um)*100).toFixed(1):'0';
 var iv=parseFloat(imp);
 var ic=iv>0?'#0f0':'#f44';
 var wn=cm<um?'ML Corrected':'Uncorrected';
 var wc=cm<um?'#0cf':'#f60';
 document.getElementById('sR').innerHTML=
  '<div class="rw"><span class="lb">Corrected (ML)</span>'+
   '<span class="vv" style="color:#0cf">'+cm.toFixed(1)+' m</span></div>'+
  '<div class="rw"><span class="lb">Uncorrected</span>'+
   '<span class="vv" style="color:#f60">'+um.toFixed(1)+' m</span></div>'+
  '<div class="rw"><span class="lb">Improvement</span>'+
   '<span class="vv" style="color:'+ic+'">'+(iv>0?'+':'')+imp+'%</span></div>'+
  '<div class="rw"><span class="lb">Avg Turbulence</span>'+
   '<span class="vv" style="color:#aa0">'+(cr.avg_turbulence*100).toFixed(0)+'%</span></div>'+
  '<div class="big"><div class="n" style="color:'+wc+'">'+wn+'</div>'+
   '<div class="l">closer to target</div></div>';
 document.getElementById('sP').classList.add('show');
}

function doReset(){
 running=false;
 fetch('/api/reset',{method:'POST'}).then(function(){
  mC.visible=mU.visible=false;
  aC.visible=aU.visible=false;
  [lC,lU,aC,aU].forEach(function(l){
   var g=l.geometry;l.geometry=new THREE.BufferGeometry();g.dispose();});
  expl.forEach(function(e){scene.remove(e.pts);scene.remove(e.fl);
   scene.remove(e.sp);scene.remove(e.rg);});
  expl=[];shkI=0;scene.position.set(0,0,0);
  document.getElementById('sP').classList.remove('show');
  document.getElementById('hS').textContent='Ready';
  document.getElementById('hD').classList.remove('on');
  document.getElementById('hT').textContent='0.0 s';
  document.getElementById('bL').disabled=false;
 });
}

/* ---- EVENTS ---- */
document.getElementById('bL').onclick=doLaunch;
document.getElementById('bR').onclick=doReset;
document.getElementById('bW').onclick=fetchW;
document.getElementById('aS').oninput=function(){
 document.getElementById('aV').textContent=this.value+' km';updTurbAt();};
document.getElementById('rS').oninput=function(){
 document.getElementById('rV').textContent=this.value+' km';};
document.querySelectorAll('input[name="mt"]').forEach(function(r){
 r.onchange=function(){
  var d=MDEF[this.value],a=document.getElementById('aS');
  a.min=d.aMin;a.max=d.aMax;a.value=d.aDef;
  document.getElementById('aV').textContent=d.aDef+' km';
  var rs=document.getElementById('rS');rs.value=d.rDef;
  document.getElementById('rV').textContent=d.rDef+' km';
  updTurbAt();};
});

/* ---- START ---- */
init();fetchW();rLoop();
</script></body></html>'''


# =====================================================================
# Flask Routes
# =====================================================================
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/weather')
def api_weather():
    w = weather_service.get_weather()
    turb_field.update()
    setup(sim['missile_type'], sim['launch_alt'], sim['target_range'])
    return jsonify(
        weather=w,
        turbulence_profile=turb_field.get_altitude_turbulence_profile(),
        model_info=turb_field.get_model_info(),
    )


@app.route('/api/configure', methods=['POST'])
def api_configure():
    d = request.json or {}
    mt = d.get('missile_type', sim['missile_type'])
    alt = float(d.get('launch_alt', sim['launch_alt']))
    rng = float(d.get('target_range', sim['target_range']))
    setup(mt, alt, rng)
    return jsonify(success=True)


@app.route('/api/launch', methods=['POST'])
def api_launch():
    sim['corrected'].launch(sim['launch_pos'], sim['target_pos'], sim['launch_alt'])
    sim['uncorrected'].launch(sim['launch_pos'], sim['target_pos'], sim['launch_alt'])
    sim['running'] = True
    sim['cr'] = None
    sim['ur'] = None
    sim['time'] = 0.0
    return jsonify(success=True)


@app.route('/api/step', methods=['POST'])
def api_step():
    if not sim['running']:
        return jsonify(success=False)
    speed = MISSILES[sim['missile_type']]['speed'] * SPEED_OF_SOUND_SEA
    dt = min(0.5, 0.12 / speed)
    n_sub = max(3, int(1.5 / dt))
    for _ in range(n_sub):
        if sim['cr'] is None:
            r = sim['corrected'].step(dt)
            if r and r['status'] == 'impact':
                sim['cr'] = r
        if sim['ur'] is None:
            r = sim['uncorrected'].step(dt)
            if r and r['status'] == 'impact':
                sim['ur'] = r
        sim['time'] += dt
    if sim['cr'] is not None and sim['ur'] is not None:
        sim['running'] = False
    return jsonify(success=True)


@app.route('/api/state')
def api_state():
    def mdata(m):
        if m is None:
            return dict(trajectory=[], cross_track=[], status='idle',
                        position=[0, 0, 0])
        n = len(m.trajectory)
        step = max(1, n // 500)
        idx = list(range(0, n, step))
        if n > 0 and (n - 1) not in idx:
            idx.append(n - 1)
        traj = [[float(m.trajectory[i][0]), float(m.trajectory[i][1]),
                  float(m.trajectory[i][2])] for i in idx if i < n]
        ct = [float(m.cross_track_log[i]) for i in idx
              if i < len(m.cross_track_log)]
        pos = ([float(m.position[j]) for j in range(3)]
               if m.position is not None else [0, 0, 0])
        return dict(trajectory=traj, cross_track=ct, position=pos,
                    status='impact' if m.impact_error is not None else 'active')

    return jsonify(
        corrected=mdata(sim['corrected']),
        uncorrected=mdata(sim['uncorrected']),
        corrected_result=sim['cr'],
        uncorrected_result=sim['ur'],
        running=sim['running'],
        time=sim['time'],
        config=dict(
            missile_type=sim['missile_type'],
            launch_alt=sim['launch_alt'],
            target_range=sim['target_range'],
            launch_pos=sim['launch_pos'],
            target_pos=sim['target_pos'],
        ),
        turbulence_profile=turb_field.get_altitude_turbulence_profile(),
        model_info=turb_field.get_model_info(),
    )


@app.route('/api/reset', methods=['POST'])
def api_reset():
    setup(sim['missile_type'], sim['launch_alt'], sim['target_range'])
    return jsonify(success=True)


if __name__ == '__main__':
    print('=' * 60)
    print(' PSTNet — 3D TRAJECTORY VISUALIZATION')
    print('=' * 60)
    turb_field.update()
    setup()
    print(f' http://127.0.0.1:5890')
    print('=' * 60)
    app.run(debug=False, port=5890, threaded=True)