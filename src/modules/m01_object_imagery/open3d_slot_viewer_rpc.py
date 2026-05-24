from __future__ import annotations
import argparse,base64,json,time,urllib.request,zlib
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from src.shared.console_colors import install_colored_errors
    install_colored_errors()
except Exception:
    pass
def o3d_import():
    try:
        import open3d as o3d; return o3d
    except Exception as e: raise SystemExit(f"Open3D import failed: {e}")
def rpc(url,method,params=None):
    data=json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params or {}}).encode()
    try:
        req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req,timeout=1.0) as r: return json.loads(r.read().decode()).get("result")
    except Exception as e: print("RPC unavailable:",e); return None
def dec(spec):
    if not isinstance(spec,dict): return np.zeros((0,3),np.float32)
    try:
        return np.frombuffer(zlib.decompress(base64.b64decode(spec["data"].encode("ascii"))),dtype=np.dtype(spec["dtype"])).reshape(tuple(spec["shape"])).astype(np.float32)
    except Exception: return np.zeros((0,3),np.float32)
def frame(res,sid):
    for f in (res or {}).get("slots",[]):
        if int(f.get("slot_id",-1))==int(sid): return f
    return {}
def pts(f):
    a=dec(f.get("xyz"))
    if a.ndim!=2 or a.shape[-1]<3:
        return np.zeros((0,3),np.float32)
    a=a[:,:3]
    return a[np.all(np.isfinite(a),axis=1)].astype(np.float32)
def cols(f,n,tint):
    c=dec(f.get("color"))
    if c.ndim!=2 or c.shape[-1]<3 or c.shape[0]!=n: c=np.ones((n,3),np.float32)*0.7
    c=np.nan_to_num(c[:,:3],nan=0.7,posinf=1.0,neginf=0.0)
    return np.clip(0.65*c+0.35*np.asarray(tint,np.float32).reshape(1,3),0,1)
def main():
    p=argparse.ArgumentParser(description="Open3D Slot Viewer over JSON-RPC")
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8771); p.add_argument("--slot",default="both",choices=["0","1","both"]); p.add_argument("--mode",default="deformed",choices=["raw","deformed"]); p.add_argument("--refresh",type=float,default=0.08); p.add_argument("--point-size",type=float,default=5.0); p.add_argument("--offset",type=float,default=1.25)
    a=p.parse_args(); o3d=o3d_import(); url=f"http://{a.host}:{a.port}/"
    vis=o3d.visualization.VisualizerWithKeyCallback(); vis.create_window(window_name=f"Inner Object Open3D JSON-RPC {a.host}:{a.port}",width=1280,height=760)
    opt=vis.get_render_option(); opt.point_size=a.point_size; opt.background_color=np.asarray([0.02,0.025,0.035])
    p0,p1=o3d.geometry.PointCloud(),o3d.geometry.PointCloud(); vis.add_geometry(p0); vis.add_geometry(p1)
    st={"slot":a.slot,"mode":a.mode,"last":""}
    vis.register_key_callback(ord("R"),lambda _v: st.update(mode="raw") or False); vis.register_key_callback(ord("D"),lambda _v: st.update(mode="deformed") or False); vis.register_key_callback(ord("0"),lambda _v: st.update(slot="0") or False); vis.register_key_callback(ord("1"),lambda _v: st.update(slot="1") or False); vis.register_key_callback(ord("B"),lambda _v: st.update(slot="both") or False)
    print("[Inner Object Open3D JSON-RPC]",url); print("keys: 0 slot0 | 1 slot1 | B both | R raw | D deformed")
    view_reset_pending=True
    while True:
        res=rpc(url,"slot_viewer.get_both_slots",{"mode":st["mode"]})
        if res:
            f0,f1=frame(res,0),frame(res,1)
            for sid,f,pc,visible,xoff,tint in [(0,f0,p0,st["slot"] in ("0","both"),-a.offset/2 if st["slot"]=="both" else 0,(0.2,0.9,1.0)),(1,f1,p1,st["slot"] in ("1","both"),a.offset/2 if st["slot"]=="both" else 0,(1.0,0.8,0.2))]:
                x=pts(f)
                if not visible: x=np.zeros((0,3),np.float32)
                if len(x): x=x.copy(); x[:,0]+=xoff
                pc.points=o3d.utility.Vector3dVector(x.astype(np.float64)); pc.colors=o3d.utility.Vector3dVector(cols(f,len(x),tint).astype(np.float64)); vis.update_geometry(pc)
            title=f"mode={st['mode']} slot={st['slot']} | s0={f0.get('target_name','?')} n={f0.get('point_count',0)} | s1={f1.get('target_name','?')} n={f1.get('point_count',0)}"
            if title!=st["last"]: print(title); st["last"]=title
            if view_reset_pending and (int(f0.get("point_count",0) or 0)>0 or int(f1.get("point_count",0) or 0)>0):
                vis.reset_view_point(True); view_reset_pending=False
        if not vis.poll_events(): break
        vis.update_renderer(); time.sleep(max(0.01,a.refresh))
    vis.destroy_window()
if __name__=="__main__": main()
