from __future__ import annotations
import base64, json, threading, time, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import torch

_STREAMERS = {}


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

def _enc(a):
    a=np.ascontiguousarray(np.asarray(a,dtype=np.float32))
    return {"shape":list(a.shape),"dtype":str(a.dtype),"encoding":"base64+zlib","data":base64.b64encode(zlib.compress(a.tobytes(),3)).decode("ascii")}
def _empty(): return np.zeros((0,3),np.float32)
def _finite_xyz_mask(xyz):
    if xyz.ndim != 2 or xyz.shape[-1] < 3:
        return np.zeros((0,), dtype=bool)
    return np.all(np.isfinite(xyz[:, :3]), axis=1)

class Slot4DJsonRpcStreamer:
    def __new__(cls, host="127.0.0.1", port=8771, sample_points=4096):
        key = (str(host), int(port))
        existing = _STREAMERS.get(key)
        if existing is not None:
            return existing
        obj = super().__new__(cls)
        _STREAMERS[key] = obj
        return obj

    def __init__(self, host="127.0.0.1", port=8771, sample_points=4096):
        if getattr(self, "_initialized", False):
            return
        self.host=str(host); self.port=int(port); self.sample_points=max(16,int(sample_points))
        self.lock=threading.RLock(); self.frames={}; self.last_update_unix=0.0; self.started=False; self.prev_phase={}; self.prev_deformed_xyz={}; self.live_delta_eps=1.0e-6
        self._initialized=True
    def start(self):
        if self.started: return True
        streamer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): return
            def do_GET(self): self._send(streamer.reply({"id":1,"method":"slot_viewer.get_status","params":{}}))
            def do_POST(self):
                try:
                    n=int(self.headers.get("Content-Length","0") or 0)
                    self._send(streamer.reply(json.loads(self.rfile.read(n).decode("utf-8") or "{}")))
                except Exception as e:
                    self._send({"jsonrpc":"2.0","id":None,"error":{"code":-32000,"message":str(e)}})
            def _send(self,obj):
                data=json.dumps(obj,ensure_ascii=False).encode("utf-8")
                self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
        try:
            self.server=_ReusableThreadingHTTPServer((self.host,self.port),Handler)
            self.thread=threading.Thread(target=self.server.serve_forever,daemon=True,name="Slot4DJsonRpcStreamer")
            self.thread.start(); self.started=True; return True
        except Exception as e:
            self.started=False
            print("[slot_4d_jsonrpc] start failed:",e); return False
    def shutdown(self, timeout=1.0):
        server=getattr(self,"server",None)
        thread=getattr(self,"thread",None)
        if server is not None:
            try: server.shutdown()
            except Exception: pass
            try: server.server_close()
            except Exception: pass
        if thread is not None and thread.is_alive():
            try: thread.join(timeout=float(timeout))
            except Exception: pass
        self.started=False
    def reply(self,req):
        rid=req.get("id"); method=str(req.get("method","")); params=req.get("params",{}) or {}
        if method=="slot_viewer.ping": result={"ok":True,"time":time.time()}
        elif method=="slot_viewer.get_status": result=self.status()
        elif method=="slot_viewer.get_slot_frame": result=self.slot(int(params.get("slot_id",0)),str(params.get("mode","deformed")))
        elif method=="slot_viewer.get_both_slots":
            mode=str(params.get("mode","deformed")); result={"mode":mode,"last_update_unix":self.last_update_unix,"slots":[self.slot(0,mode),self.slot(1,mode)]}
        else: return {"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"unknown method: "+method}}
        return {"jsonrpc":"2.0","id":rid,"result":result}
    def _np(self,x):
        if x is None or not torch.is_tensor(x): return None
        return x.detach().float().cpu().numpy().astype(np.float32)
    def _arrays(self,state):
        xyz=self._np(getattr(state,"xyz",None))
        if xyz is None or xyz.ndim!=2 or xyz.shape[-1]<3: return _empty(),_empty(),np.zeros((0,),np.float32)
        xyz=xyz[:,:3]
        color=None; c=getattr(state,"color_logit",None)
        if torch.is_tensor(c):
            color=torch.sigmoid(c.detach().float()).cpu().numpy().astype(np.float32)
            color=color[:,:3] if color.ndim==2 and color.shape[-1]>=3 else None
        if color is None or len(color)!=len(xyz): color=np.ones((len(xyz),3),np.float32)*0.7
        alpha=None; o=getattr(state,"opacity_logit",None)
        if torch.is_tensor(o): alpha=torch.sigmoid(o.detach().float()).cpu().numpy().astype(np.float32).reshape(-1)
        if alpha is None or len(alpha)!=len(xyz): alpha=np.ones((len(xyz),),np.float32)
        color=np.nan_to_num(color, nan=0.7, posinf=1.0, neginf=0.0)
        alpha=np.nan_to_num(alpha, nan=1.0, posinf=1.0, neginf=0.0)
        mask=_finite_xyz_mask(xyz) & np.all(np.isfinite(color[:, :3]), axis=1) & np.isfinite(alpha)
        xyz,color,alpha=xyz[mask],color[mask],alpha[mask]
        if len(xyz)>self.sample_points:
            idx=np.linspace(0,len(xyz)-1,self.sample_points).astype(np.int64); xyz,color,alpha=xyz[idx],color[idx],alpha[idx]
        return xyz.astype(np.float32),np.clip(color,0,1),np.clip(alpha,0,1)
    def _deform(self,sid,state,raw,trainer,phase):
        if trainer is None: return raw.copy(),False,0.0
        model=(getattr(trainer,"models",{}) or {}).get(int(sid)); x=getattr(state,"xyz",None)
        if model is None or not torch.is_tensor(x): return raw.copy(),False,0.0
        try:
            with torch.no_grad():
                try:
                    device=next(model.parameters()).device
                except Exception:
                    device=x.device
                xx=torch.as_tensor(raw, device=device, dtype=x.dtype)
                delta=model(xx[:,:3],torch.tensor(float(phase),device=xx.device,dtype=xx.dtype))
                out=(xx[:,:3]+delta).detach().float().cpu().numpy().astype(np.float32)
                if out.shape != raw.shape or not np.all(np.isfinite(out)):
                    return raw.copy(),False,0.0
                return out,True,float(torch.mean(torch.linalg.norm(delta,dim=-1)).detach().cpu().item())
        except Exception: return raw.copy(),False,0.0
    def publish(self,gaussian_reconstructor,deformation_trainer=None,playback_renderer=None,live_step=None):
        if not self.started: self.start()
        states=getattr(gaussian_reconstructor,"states",{}) or {}; gm=getattr(gaussian_reconstructor,"last_metrics",{}) or {}; pm=getattr(playback_renderer,"last_metrics",{}) or {}; frames={}
        for sid in (0,1):
            st=states.get(sid); target=str(getattr(gm.get(sid),"target_name",getattr(st,"target_name",f"slot_{sid}")) if st is not None else f"slot_{sid}")
            raw,col,alpha=self._arrays(st) if st is not None else (_empty(),_empty(),np.zeros((0,),np.float32))
            phase=float(getattr(pm.get(sid),"playback_phase",0.0) or 0.0) if pm.get(sid) is not None else 0.0
            if live_step is not None and playback_renderer is not None and hasattr(playback_renderer, "phase_from_step"):
                try:
                    phase=float(playback_renderer.phase_from_step(int(live_step)))
                except Exception:
                    pass
            deformed,used,dnorm=self._deform(sid,st,raw,deformation_trainer,phase) if st is not None and len(raw) else (raw.copy(),False,0.0)
            prev_phase=self.prev_phase.get(sid, None)
            prev_xyz=self.prev_deformed_xyz.get(sid, None)
            phase_delta=0.0 if prev_phase is None else abs(float(phase)-float(prev_phase))
            xyz_delta=0.0
            if prev_xyz is not None and getattr(prev_xyz, "shape", None)==getattr(deformed, "shape", None) and len(deformed):
                try: xyz_delta=float(np.mean(np.linalg.norm(deformed-prev_xyz, axis=-1)))
                except Exception: xyz_delta=0.0
            live_phase_changed=bool(phase_delta>self.live_delta_eps)
            live_xyz_changed=bool(xyz_delta>self.live_delta_eps)
            self.prev_phase[sid]=float(phase)
            self.prev_deformed_xyz[sid]=deformed.copy()
            frames[sid]={"slot_id":sid,"target_name":target,"raw_xyz":raw,"deformed_xyz":deformed,"color":col,"alpha":alpha,"playback_phase":phase,"deformation_used":used,"pred_delta_norm":dnorm,"raw_points":int(len(raw)),"deformed_points":int(len(deformed)),"phase_delta":phase_delta,"xyz_delta":xyz_delta,"live_phase_changed":live_phase_changed,"live_xyz_changed":live_xyz_changed}
        with self.lock: self.frames=frames; self.last_update_unix=time.time()
        return {"enabled":True,"host":self.host,"port":self.port,"started":self.started,"updated":True,"reason":"ok","slot_0_points":frames[0]["raw_points"],"slot_1_points":frames[1]["raw_points"],"slot_0_deformed_points":frames[0]["deformed_points"],"slot_1_deformed_points":frames[1]["deformed_points"],"slot_0_phase_delta":frames[0].get("phase_delta",0.0),"slot_1_phase_delta":frames[1].get("phase_delta",0.0),"slot_0_xyz_delta":frames[0].get("xyz_delta",0.0),"slot_1_xyz_delta":frames[1].get("xyz_delta",0.0),"slot_0_live_phase_changed":frames[0].get("live_phase_changed",False),"slot_1_live_phase_changed":frames[1].get("live_phase_changed",False),"slot_0_live_xyz_changed":frames[0].get("live_xyz_changed",False),"slot_1_live_xyz_changed":frames[1].get("live_xyz_changed",False)}
    def status(self):
        with self.lock:
            return {"ok":True,"host":self.host,"port":self.port,"started":self.started,"last_update_unix":self.last_update_unix,"slots":{str(k):{"target_name":v["target_name"],"raw_points":v["raw_points"],"deformed_points":v["deformed_points"],"playback_phase":v["playback_phase"],"deformation_used":v["deformation_used"],"phase_delta":v.get("phase_delta",0.0),"xyz_delta":v.get("xyz_delta",0.0),"live_phase_changed":v.get("live_phase_changed",False),"live_xyz_changed":v.get("live_xyz_changed",False)} for k,v in self.frames.items()}}
    def slot(self,sid,mode="deformed"):
        mode="raw" if mode=="raw" else "deformed"
        with self.lock:
            f=self.frames.get(int(sid),{"slot_id":sid,"target_name":f"slot_{sid}","raw_xyz":_empty(),"deformed_xyz":_empty(),"color":_empty(),"alpha":np.zeros((0,),np.float32),"playback_phase":0.0,"deformation_used":False,"pred_delta_norm":0.0})
            xyz=f["raw_xyz"] if mode=="raw" else f["deformed_xyz"]
            return {"slot_id":int(sid),"target_name":str(f["target_name"]),"mode":mode,"xyz":_enc(xyz),"color":_enc(f["color"]),"alpha":_enc(f["alpha"]),"point_count":int(len(xyz)),"playback_phase":float(f["playback_phase"]),"deformation_used":bool(f["deformation_used"]),"pred_delta_norm":float(f["pred_delta_norm"]),"phase_delta":float(f.get("phase_delta",0.0)),"xyz_delta":float(f.get("xyz_delta",0.0)),"live_phase_changed":bool(f.get("live_phase_changed",False)),"live_xyz_changed":bool(f.get("live_xyz_changed",False)),"last_update_unix":float(self.last_update_unix)}
