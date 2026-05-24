
from __future__ import annotations
import argparse, time
from pathlib import Path
import sys
from typing import Any
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from src.shared.console_colors import install_colored_errors
    install_colored_errors()
except Exception:
    pass

def _load_open3d():
    try:
        import open3d as o3d
        return o3d
    except Exception as e:
        raise SystemExit(f'Open3D import failed: {e}\nInstall/open display first, then rerun this viewer.')

def _read_npz(path:Path)->dict[str,Any]|None:
    if not path.exists(): return None
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k:z[k] for k in z.files}
    except Exception:
        return None

def _points(data, sid:int, mode:str):
    key=f'slot_{sid}_{"deformed_xyz" if mode=="deformed" else "raw_xyz"}'
    arr=np.asarray(data.get(key, np.zeros((0,3),np.float32)), dtype=np.float32)
    return arr[:, :3] if arr.ndim==2 and arr.shape[-1]>=3 else np.zeros((0,3),np.float32)

def _colors(data, sid:int, n:int, tint=None):
    arr=np.asarray(data.get(f'slot_{sid}_color', np.ones((n,3),np.float32)*0.7), dtype=np.float32)
    if arr.ndim!=2 or arr.shape[-1]<3 or arr.shape[0]!=n:
        arr=np.ones((n,3),np.float32)*0.7
    arr=np.clip(arr[:, :3],0,1)
    if tint is not None:
        arr=np.clip(0.65*arr+0.35*np.asarray(tint,np.float32).reshape(1,3),0,1)
    return arr

def _target(data, sid:int):
    try: return str(data.get(f'slot_{sid}_target', np.array([f'slot_{sid}'], dtype=object)).reshape(-1)[0])
    except Exception: return f'slot_{sid}'

def _phase(data, sid:int):
    try: return float(np.asarray(data.get(f'slot_{sid}_playback_phase',[0.0])).reshape(-1)[0])
    except Exception: return 0.0

def main():
    ap=argparse.ArgumentParser(description='Open3D Slot Viewer for Conscious World Model slots')
    ap.add_argument('--path', default='./checkpoint/slot_viewer/slot_4d_open3d_latest.npz')
    ap.add_argument('--slot', default='both', choices=['0','1','both'])
    ap.add_argument('--mode', default='deformed', choices=['raw','deformed'])
    ap.add_argument('--refresh', type=float, default=0.15)
    ap.add_argument('--point-size', type=float, default=5.0)
    ap.add_argument('--offset', type=float, default=1.25)
    args=ap.parse_args(); o3d=_load_open3d(); path=Path(args.path)
    vis=o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Open3D Slot Viewer: slot-0 tetrahedron / slot-1 cube', width=1280, height=760)
    opt=vis.get_render_option(); opt.point_size=float(args.point_size); opt.background_color=np.asarray([0.02,0.025,0.035], dtype=np.float32)
    pcd0=o3d.geometry.PointCloud(); pcd1=o3d.geometry.PointCloud(); vis.add_geometry(pcd0); vis.add_geometry(pcd1)
    state={'mode':args.mode, 'slot':args.slot, 'last_title':''}
    vis.register_key_callback(ord('R'), lambda v: state.update(mode='raw') or False)
    vis.register_key_callback(ord('D'), lambda v: state.update(mode='deformed') or False)
    vis.register_key_callback(ord('0'), lambda v: state.update(slot='0') or False)
    vis.register_key_callback(ord('1'), lambda v: state.update(slot='1') or False)
    vis.register_key_callback(ord('B'), lambda v: state.update(slot='both') or False)
    print('[Open3D Slot Viewer]')
    print(f'reading: {path}')
    print('keys: 0 slot0 | 1 slot1 | B both | R raw | D deformed')
    first=True
    while True:
        data=_read_npz(path)
        if data is not None:
            slot=state['slot']; mode=state['mode']
            for sid,pcd,visible,xoff,tint in [(0,pcd0,slot in ('0','both'),-args.offset*0.5 if slot=='both' else 0.0,(0.2,0.9,1.0)),(1,pcd1,slot in ('1','both'),args.offset*0.5 if slot=='both' else 0.0,(1.0,0.8,0.2))]:
                pts=_points(data,sid,mode)
                if not visible: pts=np.zeros((0,3),np.float32)
                if pts.shape[0]>0:
                    pts=pts.copy(); pts[:,0]+=float(xoff)
                cols=_colors(data,sid,pts.shape[0],tint)
                pcd.points=o3d.utility.Vector3dVector(pts.astype(np.float64)); pcd.colors=o3d.utility.Vector3dVector(cols.astype(np.float64)); vis.update_geometry(pcd)
            title=f"mode={mode} slot={slot} | slot0={_target(data,0)} phase={_phase(data,0):.2f} n={_points(data,0,mode).shape[0]} | slot1={_target(data,1)} phase={_phase(data,1):.2f} n={_points(data,1,mode).shape[0]}"
            if title!=state['last_title']: print(title); state['last_title']=title
            if first: vis.reset_view_point(True); first=False
        if not vis.poll_events(): break
        vis.update_renderer(); time.sleep(max(0.01,float(args.refresh)))
    vis.destroy_window()
if __name__=='__main__': main()
