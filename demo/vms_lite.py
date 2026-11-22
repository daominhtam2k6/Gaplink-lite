"""GapLink-VMS (lean): 3 demo clips, baseline (left) vs GapLink-Lite (right).

  .venv\\Scripts\\python.exe -m demo.vms_lite                 # dashboard at http://127.0.0.1:8000
  .venv\\Scripts\\python.exe -m demo.vms_lite --check         # print the metrics of the 3 clips (no UI)
  .venv\\Scripts\\python.exe -m demo.vms_lite --render        # render the 3 clips to demo/out/vms_lite/*.mp4

GapLink runs on CPU with 1 thread by default (fastest on the RTX 3050 laptop: kernel-launch overhead dominates on the GPU);
override with GAPLINK_GL_DEVICE / GAPLINK_GL_THREADS.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from demo.vms_lite_core import ROOT, ClipRunner, find_clips, save_zones_to_meta

BASELINE = "C-3"
SPEEDS = (1.0, 0.5, 0.25)


def make_assocs():
    import torch
    from demo.vms_profile import load_profile, make_baseline, make_gaplink
    prof = load_profile(ROOT)
    dev = os.environ.get("GAPLINK_GL_DEVICE", "cpu")
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("GAPLINK_GL_THREADS", "1")))
    gl = make_gaplink(prof, dev)
    bl_label = f"{BASELINE} Kalman + Mahalanobis"
    gl_label = f"GapLink-Lite {prof.version}"
    return (lambda: make_baseline(BASELINE, prof)), gl, bl_label, gl_label


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.bl_factory, self.gl, self.bl_label, self.gl_label = make_assocs()
        self.clips = find_clips()
        self.clip = None
        self.runner: ClipRunner | None = None
        self.cap = None
        self.frame_idx = 0
        self.frame = None
        self.paused = True
        self.speed = 1.0
        self.show_ids = True
        self.zone_edit: list[dict] | None = None
        self.ema_ms = None
        self.hist_ver = 0
        if self.clips:
            self.load(self.clips[0]["slot"])

    # ------------------------------------------------------------ clip control
    def load(self, slot: int, zones_override: list[dict] | None = None):
        c = next((c for c in self.clips if c["slot"] == slot), None)
        if c is None:
            return
        self.clip = c
        self.zone_edit = zones_override
        self._rebuild()

    def _rebuild(self):
        if self.cap is not None:
            self.cap.release()
        self.runner = ClipRunner(self.clip["path"], self.clip["scenario"], self.bl_factory(), self.gl,
                                 self.bl_label, self.gl_label, zones_override=self.zone_edit)
        self.cap = cv2.VideoCapture(str(self.clip["path"]))
        self.frame_idx, self.frame = 0, None
        self.hist_ver += 1
        self._advance()          # show first frame

    def _advance(self) -> bool:
        ok, frm = self.cap.read()
        if not ok or frm is None:
            self.paused = True
            return False
        self.frame_idx += 1
        self.frame = frm
        nh = len(self.runner.left.history) + len(self.runner.right.history)
        self.runner.step(self.frame_idx, frm)
        if len(self.runner.left.history) + len(self.runner.right.history) != nh:
            self.hist_ver += 1
        return True

    def seek(self, target: int):
        target = int(np.clip(target, 1, max(1, self.runner.n)))
        if target <= self.frame_idx:
            self._rebuild()
        while self.frame_idx < target and self._advance():
            pass

    # ------------------------------------------------------------ per-tick
    def tick(self) -> dict | None:
        with self.lock:
            if self.runner is None:
                return None
            t0 = time.perf_counter()
            if not self.paused:
                self._advance()
            img = self.runner.draw(self.frame, self.frame_idx, self.show_ids)
            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            ms = (time.perf_counter() - t0) * 1000.0
            if not self.paused:
                self.ema_ms = ms if self.ema_ms is None else 0.85 * self.ema_ms + 0.15 * ms
            return {"img": base64.b64encode(jpg.tobytes()).decode("ascii"), "frame": self.frame_idx,
                    "total": self.runner.n, "paused": self.paused, "speed": self.speed, "ids": self.show_ids,
                    "fps": round(1000.0 / self.ema_ms, 1) if self.ema_ms else None,
                    "slot": self.clip["slot"], "scenario": self.runner.mode, "W": self.runner.W, "H": self.runner.H,
                    "metrics": self.runner.metrics(), "hist_ver": self.hist_ver,
                    "labels": {"bl": self.bl_label, "gl": self.gl_label},
                    "zones": [{"zone_id": z.zone_id, "points": [list(p) for p in z.points]} for z in self.runner.left.zones]}


from pydantic import BaseModel  # noqa: E402


class Act(BaseModel):          # module level: FastAPI must resolve it from the string annotation
    action: str
    slot: Optional[int] = None
    value: Optional[float] = None
    flag: Optional[bool] = None
    x: Optional[float] = None
    y: Optional[float] = None
    zone_id: Optional[str] = None
    points: Optional[List[List[float]]] = None


SESSION: Session | None = None


def sess() -> Session:
    global SESSION
    if SESSION is None:
        SESSION = Session()
    return SESSION


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
    app = FastAPI(title="GapLink-VMS lite")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/clips")
    def clips():
        return [{"slot": c["slot"], "scenario": c["scenario"], "id": c["id"]} for c in sess().clips]

    @app.get("/api/history")
    def history():
        s = sess()
        with s.lock:
            return {"ver": s.hist_ver, "items": s.runner.history() if s.runner else []}

    @app.post("/api/act")
    def act(a: Act):
        s = sess()
        with s.lock:
            if a.action == "load" and a.slot:
                s.load(a.slot)
            elif a.action == "play":
                s.paused = not s.paused if a.flag is None else not a.flag
            elif a.action == "restart":
                s._rebuild()
                s.paused = False
            elif a.action == "step" and a.value is not None:
                s.paused = True
                s.seek(s.frame_idx + int(a.value))
            elif a.action == "seek" and a.value is not None:
                s.seek(int(a.value))
            elif a.action == "speed" and a.value in SPEEDS:
                s.speed = float(a.value)
            elif a.action == "ids" and a.flag is not None:
                s.show_ids = bool(a.flag)
            elif a.action == "lock" and a.x is not None and a.y is not None:
                return {"ok": True, **s.runner.lock_click(a.x * s.runner.W, a.y * s.runner.H, s.frame_idx, s.frame)}
            elif a.action == "zone" and a.zone_id in ("Z_A", "Z_B") and a.points and len(a.points) >= 3:
                cur = [{"zone_id": z.zone_id, "points": [list(p) for p in z.points]} for z in s.runner.left.zones]
                cur = [z for z in cur if z["zone_id"] != a.zone_id] + [{"zone_id": a.zone_id, "points": a.points}]
                s.zone_edit = cur
                s._rebuild()
            elif a.action == "reset_zones":
                s.zone_edit = None
                s._rebuild()
            elif a.action == "save_zones" and s.zone_edit:
                save_zones_to_meta(s.clip["path"], s.zone_edit)
            return {"ok": True}

    @app.get("/api/stream")
    async def stream():
        s = sess()

        async def gen():
            while True:
                t = time.perf_counter()
                pkt = await asyncio.to_thread(s.tick)
                if pkt is not None:
                    yield f"data: {json.dumps(pkt)}\n\n"
                period = 1.0 / (s.runner.fps if s.runner else 25.0) / max(0.05, s.speed)
                await asyncio.sleep(max(0.004, (period if not s.paused else 0.12) - (time.perf_counter() - t)))

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


# ---------------------------------------------------------------------------------------------- offline
def run_check():
    bl_f, gl, bl_label, gl_label = make_assocs()
    out = {}
    for c in find_clips():
        r = ClipRunner(c["path"], c["scenario"], bl_f(), gl, bl_label, gl_label)
        cap = cv2.VideoCapture(str(c["path"]))
        f = 0
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            f += 1
            r.step(f, frm)
        m = r.metrics()
        out[c["id"]] = {"scenario": c["scenario"], "baseline": bl_label, "gaplink": gl_label, "bl": m["bl"], "gl": m["gl"],
                        "history_items": len(r.history())}
        print(f"[{c['slot']}] {c['id']:28s} {c['scenario']:6s}  {bl_label}: {m['bl']}  |  {gl_label}: {m['gl']}  "
              f"| history items {len(r.history())}", flush=True)
    od = ROOT / "runs" / "vms_lite_check"
    od.mkdir(parents=True, exist_ok=True)
    (od / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", od / "metrics.json")


def run_render(slow: float = 0.25):
    import imageio.v2 as imageio
    from demo.clip_meta import slow_windows
    bl_f, gl, bl_label, gl_label = make_assocs()
    out_dir = ROOT / "demo" / "out" / "vms_lite"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in find_clips():
        r = ClipRunner(c["path"], c["scenario"], bl_f(), gl, bl_label, gl_label)
        wins = slow_windows(r.meta, pre=20, post=30)
        snap_at = set()
        for k in r.meta.get("key_moments", []):
            snap_at |= {int(k["frame"]), int(k["frame"]) - 25}
        snap_dir = ROOT / "runs" / "vms_lite_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(c["path"]))
        tmp = out_dir / f"{c['path'].stem}_tmp.mp4"
        w = imageio.get_writer(str(tmp), fps=25, codec="libx264", quality=None, macro_block_size=None,
                               pixelformat="yuv420p", ffmpeg_params=["-crf", "18", "-preset", "veryfast"])
        f = 0
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            f += 1
            r.step(f, frm)
            img = r.draw(frm, f, True)
            if f in snap_at:
                cv2.imwrite(str(snap_dir / f"{c['path'].stem}_f{f:04d}.png"), img)
            canvas = np.full((1080, 1920, 3), (18, 14, 10), np.uint8)
            pane = cv2.resize(img, (1920, int(round(1920 * img.shape[0] / img.shape[1]))), interpolation=cv2.INTER_AREA)
            y0 = (1080 - pane.shape[0]) // 2
            canvas[y0:y0 + pane.shape[0]] = pane
            in_win = any(a <= f <= b for a, b, _, _ in wins)
            if in_win:
                cv2.putText(canvas, f"SLOW x{slow}", (1700, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (40, 170, 255), 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            for _ in range(int(round(1 / slow)) if in_win else 1):
                w.append_data(rgb)
        w.close()
        dst = out_dir / f"{c['path'].stem}.mp4"
        tmp.replace(dst)
        print(f"rendered {dst}  metrics {r.metrics()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.check:
        return run_check()
    if a.render:
        return run_render()
    import uvicorn
    app = build_app()
    sess()
    print(f"GapLink-VMS lite: http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


HTML = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"/><title>GapLink-VMS</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b0f17;color:#e5e7eb;font:14px system-ui,Segoe UI,Arial}
header{display:flex;align-items:center;gap:10px;padding:10px 16px;background:#111827;border-bottom:1px solid #1f2937;flex-wrap:wrap}
h1{font-size:18px;margin:0 14px 0 0}button{background:#1f2937;color:#e5e7eb;border:1px solid #374151;border-radius:7px;padding:7px 12px;cursor:pointer;font-size:13px}
button:hover{background:#273449}button.on{background:#2563eb;border-color:#3b82f6}.sep{width:1px;height:26px;background:#374151}
main{display:grid;grid-template-columns:1fr 380px;gap:12px;padding:12px}
#view{position:relative}#cv{width:100%;border-radius:8px;background:#000;cursor:crosshair;display:block}
.bar{display:flex;align-items:center;gap:8px;margin-top:8px}#sl{flex:1}
.card{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:10px 12px;margin-bottom:12px}
.card h3{margin:0 0 8px;font-size:13px;color:#9ca3af;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse}td,th{padding:6px 4px;border-bottom:1px solid #1f2937;text-align:center}td:first-child,th:first-child{text-align:left}
th.bl{color:#8ea2ff}th.gl{color:#7ee787}.big{font-size:18px;font-weight:700}
.hint{color:#9ca3af;font-size:12px}#hist{max-height:52vh;overflow:auto}
.hi{display:flex;gap:8px;align-items:center;padding:6px;border-radius:8px;background:#0f172a;margin-bottom:6px;border-left:4px solid #555}
.hi img{width:84px;height:60px;object-fit:contain;background:#000;border-radius:4px}.hi .t{font-size:12px;line-height:1.35}
.tool{display:none}.tool.show{display:block}
</style></head><body>
<header><h1>GapLink-VMS</h1>
<div id="clipBtns"></div><div class="sep"></div>
<button id="bPlay" onclick="act({action:'play'})">▶ Chạy</button>
<button onclick="act({action:'restart'})">⏮ Đầu</button>
<button onclick="act({action:'step',value:-1})">◀ 1f</button>
<button onclick="act({action:'step',value:1})">1f ▶</button><div class="sep"></div>
<button id="sp1" onclick="spd(1)">1×</button><button id="sp05" onclick="spd(0.5)">0.5×</button><button id="sp025" onclick="spd(0.25)">0.25×</button><div class="sep"></div>
<button id="bIds" class="on" onclick="toggleIds()">ID: bật</button>
<span class="hint" id="fps"></span>
</header>
<main>
<div id="view"><canvas id="cv"></canvas>
<div class="bar"><span id="fr" class="hint">0 / 0</span><input id="sl" type="range" min="1" max="1" value="1"/></div>
<div class="hint" id="tip" style="margin-top:6px"></div></div>
<div>
<div class="card"><h3 id="mTitle">Kết quả</h3><table id="mTab"></table></div>
<div class="card tool" id="toolRoute"><h3>Vùng lộ trình cấm (Z_A → Z_B)</h3>
<button id="dA" onclick="draw('Z_A')">✏️ Vẽ Z_A</button> <button id="dB" onclick="draw('Z_B')">✏️ Vẽ Z_B</button>
<button onclick="finish()">✔ Xong vùng</button><br/><div style="height:6px"></div>
<button onclick="act({action:'reset_zones'})">↺ Vùng mặc định</button> <button onclick="act({action:'save_zones'})">💾 Lưu vào clip</button>
<div class="hint" style="margin-top:6px">Dừng video, bấm Vẽ, click các đỉnh trên hình (bên nào cũng được), rồi Xong vùng. Video chạy lại từ đầu.</div></div>
<div class="card tool" id="toolLock"><h3>Truy vết</h3><div class="hint">Dừng video, click vào xe cần theo dõi (bên nào cũng được). Cả hai phương pháp khóa cùng một xe.</div></div>
<div class="card"><h3>Lịch sử sự cố (trước → sau)</h3><div id="hist" class="hint">Chưa có sự cố.</div></div>
</div></main>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),img=new Image();let S={},drawZone=null,pts=[],histVer=-1,drag=false;
const NAMES={1:'1 · Khôi phục ID',2:'2 · Lộ trình cấm',3:'3 · Truy vết'};
async function act(b){const r=await fetch('/api/act',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json();}
function spd(v){act({action:'speed',value:v});}
function toggleIds(){act({action:'ids',flag:!S.ids});}
function draw(z){if(!S.paused)act({action:'play',flag:false});drawZone=z;pts=[];document.getElementById('dA').classList.toggle('on',z==='Z_A');document.getElementById('dB').classList.toggle('on',z==='Z_B');}
function finish(){if(drawZone&&pts.length>=3)act({action:'zone',zone_id:drawZone,points:pts});drawZone=null;pts=[];['dA','dB'].forEach(i=>document.getElementById(i).classList.remove('on'));}
(async()=>{const cl=await (await fetch('/api/clips')).json();const box=document.getElementById('clipBtns');
 cl.forEach(c=>{const b=document.createElement('button');b.id='clip'+c.slot;b.textContent=NAMES[c.slot]||c.id;b.title=c.id;b.onclick=()=>act({action:'load',slot:c.slot});box.appendChild(b);box.append(' ');});})();
cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect();let x=(e.clientX-r.left)/r.width*2,y=(e.clientY-r.top)/r.height;if(x>=1)x-=1;
 if(drawZone){pts.push([+x.toFixed(4),+y.toFixed(4)]);return;} if(S.scenario==='lock')act({action:'lock',x:x,y:y});});
const sl=document.getElementById('sl');sl.oninput=()=>{drag=true;};sl.onchange=()=>{act({action:'seek',value:+sl.value});drag=false;};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();act({action:'play'});}else if(k==='arrowleft')act({action:'step',value:-1});else if(k==='arrowright')act({action:'step',value:1});
 else if(k==='i')toggleIds();else if(['1','2','3'].includes(k))act({action:'load',slot:+k});else if(k==='enter'&&drawZone)finish();else if(k==='escape'){drawZone=null;pts=[];}});
function overlay(){ if(!drawZone||!pts.length)return; ctx.strokeStyle=drawZone==='Z_A'?'#ffc800':'#ff4040';ctx.lineWidth=3;
 for(const off of [0,cv.width/2]){ctx.beginPath();pts.forEach((p,i)=>{const X=off+p[0]*cv.width/2,Y=p[1]*cv.height;i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();
 pts.forEach(p=>{ctx.fillStyle='#fff';ctx.fillRect(off+p[0]*cv.width/2-3,p[1]*cv.height-3,6,6);});}}
function rows(sc,m){const b=m.bl,g=m.gl;
 if(sc==='relink')return [['Khôi phục đúng ID / tổng xe bị che',`${b.ok} / ${b.total}`,`${g.ok} / ${g.total}`],['Ghép nhầm sang xe khác',b.wrong,g.wrong],['Bị gán ID mới',b.missed,g.missed]];
 if(sc==='route')return [['Vi phạm, đúng ID',b.ok,g.ok],['Vi phạm, sai ID',b.wrong_id,g.wrong_id],['Vi phạm giả (không vi phạm)',b.false,g.false]];
 const st=x=>({IDLE:'chưa khóa',LOCKED:'đang theo',REACQUIRED:'đã khôi phục',OCCLUDED_SEARCHING:`bị che ${x.gap}f`,AMBIGUOUS_HOLD:`bị che ${x.gap}f`,HIJACKED_WRONG:(x.reason==='new_id'?'đổi ID':'nhầm xe'),LOST_EXPIRED:'mất dấu'}[x.state]||x.state);
 return [['Trạng thái mục tiêu',st(b),st(g)],['Khôi phục đúng xe',b.reacquired,g.reacquired],['Mất mục tiêu (đổi ID / nhầm xe)',b.wrong,g.wrong]];}
const TITLE={relink:'Khôi phục danh tính sau che khuất',route:'Lộ trình cấm Z_A → Z_B',lock:'Truy vết xe bị che khuất'};
const TIP={relink:'Xanh: khôi phục đúng ID · Đỏ: ghép nhầm sang xe khác · Tím: xe bị gán ID mới. Chấm điểm theo nhãn GT của UA-DETRAC (khoảng che ≥ 3 frame).',
 route:'Đỏ: vi phạm đúng ID · Cam: vi phạm nhưng sai ID · Tím: vi phạm giả. Vẽ lại vùng nếu cần (khung bên phải).',
 lock:'Click chọn xe khi video dừng. Nét đứt xanh lơ: đường đi dự đoán khi xe bị che.'};
const HK={wrong:['Ghép nhầm','#eb2828'],missed:['Gán ID mới','#c83cc8'],route_ok:['Vi phạm, đúng ID','#e61e1e'],route_wrong_id:['Vi phạm, sai ID','#ffbe00'],route_false:['Vi phạm giả','#c83cc8'],route_unk:['Vi phạm (không có GT)','#888'],lock_wrong:['Nhầm xe mục tiêu','#eb2828'],lock_newid:['Mục tiêu bị đổi ID','#c83cc8'],lock_lost:['Mất dấu mục tiêu','#888']};
async function loadHist(){const d=await (await fetch('/api/history')).json();histVer=d.ver;const el=document.getElementById('hist');
 if(!d.items.length){el.innerHTML='Chưa có sự cố.';return;}
 el.innerHTML=d.items.map(h=>{const k=HK[h.kind]||[h.kind,'#888'];const who=h.side==='gl'?'<b style="color:#7ee787">GapLink</b>':'<b style="color:#8ea2ff">'+(S.labels?S.labels.bl.split(' ')[0]:'Baseline')+'</b>';
  const id=(h.kind==='missed'||h.kind==='lock_newid')?`#${h.was} → #${h.tid}`:`#${h.tid}`;const t=(h.frame/25).toFixed(1);
  return `<div class="hi" style="border-left-color:${k[1]}"><img src="${h.before||''}"/>→<img src="${h.after||''}"/><div class="t">${who}<br/><b style="color:${k[1]}">${k[0]}</b> ${id}<br/>frame ${h.frame} (${t}s)</div></div>`;}).join('');}
const es=new EventSource('/api/stream');
es.onmessage=ev=>{const d=JSON.parse(ev.data);S=d;img.onload=()=>{if(cv.width!==img.width){cv.width=img.width;cv.height=img.height;}ctx.drawImage(img,0,0);overlay();};img.src='data:image/jpeg;base64,'+d.img;
 document.getElementById('fr').textContent=`${d.frame} / ${d.total}`;if(!drag){sl.max=d.total;sl.value=d.frame;}
 document.getElementById('fps').textContent=d.fps?`${d.fps} FPS xử lý`:'';document.getElementById('bPlay').textContent=d.paused?'▶ Chạy':'⏸ Dừng';
 document.getElementById('bIds').textContent='ID: '+(d.ids?'bật':'tắt');document.getElementById('bIds').classList.toggle('on',d.ids);
 [['sp1',1],['sp05',0.5],['sp025',0.25]].forEach(([i,v])=>document.getElementById(i).classList.toggle('on',d.speed===v));
 [1,2,3].forEach(n=>{const b=document.getElementById('clip'+n);if(b)b.classList.toggle('on',d.slot===n);});
 document.getElementById('mTitle').textContent=TITLE[d.scenario]||'';document.getElementById('tip').textContent=TIP[d.scenario]||'';
 document.getElementById('toolRoute').classList.toggle('show',d.scenario==='route');document.getElementById('toolLock').classList.toggle('show',d.scenario==='lock');
 const L=d.labels||{bl:'Baseline',gl:'GapLink'};
 document.getElementById('mTab').innerHTML=`<tr><th></th><th class="bl">${L.bl.split(' ')[0]}</th><th class="gl">GapLink</th></tr>`+rows(d.scenario,d.metrics).map(r=>`<tr><td>${r[0]}</td><td class="big">${r[1]}</td><td class="big">${r[2]}</td></tr>`).join('');
 if(d.hist_ver!==histVer)loadHist();};
</script></body></html>"""


if __name__ == "__main__":
    main()
