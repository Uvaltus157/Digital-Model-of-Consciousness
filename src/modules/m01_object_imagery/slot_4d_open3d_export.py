
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import numpy as np
import torch

@dataclass
class SlotOpen3DExportMetrics:
    enabled: bool
    export_path: str
    slot_id: int
    target_name: str
    raw_points: int
    deformed_points: int
    deformation_used: bool
    written: bool
    reason: str

class Slot4DOpen3DExporter:
    def __init__(self, *, export_path: str | Path = './checkpoint/slot_viewer/slot_4d_open3d_latest.npz', sample_points: int = 4096, min_interval_sec: float = 0.05) -> None:
        self.export_path = Path(export_path)
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_points = max(16, int(sample_points))
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.last_write_time = 0.0
        self.last_metrics: dict[int, SlotOpen3DExportMetrics] = {}

    @staticmethod
    def _tensor_to_np(x: Any) -> np.ndarray | None:
        if x is None or not torch.is_tensor(x):
            return None
        return x.detach().float().cpu().numpy().astype(np.float32)

    def _state_arrays(self, state: Any) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        xyz = self._tensor_to_np(getattr(state, 'xyz', None))
        if xyz is None or xyz.ndim != 2 or xyz.shape[-1] < 3:
            return None, None, None
        xyz = xyz[:, :3].astype(np.float32)
        color = None
        color_logit = getattr(state, 'color_logit', None)
        if torch.is_tensor(color_logit):
            color = torch.sigmoid(color_logit.detach().float()).cpu().numpy().astype(np.float32)
            color = color[:, :3] if color.ndim == 2 and color.shape[-1] >= 3 else None
        if color is None or color.shape[0] != xyz.shape[0]:
            color = np.ones((xyz.shape[0], 3), dtype=np.float32) * 0.65
        alpha = None
        opacity_logit = getattr(state, 'opacity_logit', None)
        if torch.is_tensor(opacity_logit):
            alpha = torch.sigmoid(opacity_logit.detach().float()).cpu().numpy().astype(np.float32).reshape(-1)
        if alpha is None or alpha.shape[0] != xyz.shape[0]:
            alpha = np.ones((xyz.shape[0],), dtype=np.float32)
        if xyz.shape[0] > self.sample_points:
            idx = np.linspace(0, xyz.shape[0] - 1, self.sample_points).astype(np.int64)
            xyz, color, alpha = xyz[idx], color[idx], alpha[idx]
        return xyz, np.clip(color,0,1), np.clip(alpha,0,1)

    def _deformed_xyz(self, *, slot_id:int, xyz:np.ndarray, gaussian_state:Any, deformation_trainer:Any, playback_phase:float) -> tuple[np.ndarray,bool,float]:
        model = (getattr(deformation_trainer, 'models', {}) or {}).get(int(slot_id))
        base = getattr(gaussian_state, 'xyz', None)
        if model is None or not torch.is_tensor(base):
            return xyz.copy(), False, 0.0
        try:
            with torch.no_grad():
                x = base.detach()
                if x.shape[0] > self.sample_points:
                    idx = torch.linspace(0, x.shape[0]-1, self.sample_points, device=x.device).long()
                    x = x.index_select(0, idx)
                phase = torch.tensor(float(playback_phase), device=x.device, dtype=x.dtype)
                delta = model(x[:, :3], phase)
                out = x[:, :3] + delta
                norm = float(torch.mean(torch.linalg.norm(delta, dim=-1)).detach().cpu().item())
                return out.detach().float().cpu().numpy().astype(np.float32), True, norm
        except Exception:
            return xyz.copy(), False, 0.0

    def export(self, *, gaussian_reconstructor:Any, deformation_trainer:Any|None=None, playback_renderer:Any|None=None, force:bool=False) -> dict[str,Any]:
        now=time.time()
        if not force and self.min_interval_sec>0 and (now-self.last_write_time)<self.min_interval_sec:
            return {'enabled':True,'export_path':str(self.export_path),'written':False,'reason':'throttled'}
        states=getattr(gaussian_reconstructor,'states',{}) or {}
        last_metrics=getattr(gaussian_reconstructor,'last_metrics',{}) or {}
        playback_metrics=getattr(playback_renderer,'last_metrics',{}) or {}
        payload={'created_unix':np.array([now],dtype=np.float64),'slot_ids':np.array([0,1],dtype=np.int32)}
        any_points=False; written_metrics={}
        for sid in (0,1):
            state=states.get(sid); gm=last_metrics.get(sid)
            target=str(getattr(gm,'target_name',getattr(state,'target_name',f'slot_{sid}')) if state is not None else f'slot_{sid}')
            raw,color,alpha=self._state_arrays(state) if state is not None else (None,None,None)
            if raw is None:
                raw=np.zeros((0,3),np.float32); color=np.zeros((0,3),np.float32); alpha=np.zeros((0,),np.float32)
            phase=0.0; pm=playback_metrics.get(sid)
            if pm is not None: phase=float(getattr(pm,'playback_phase',0.0) or 0.0)
            if deformation_trainer is not None and raw.shape[0]>0 and state is not None:
                deformed, used, delta_norm = self._deformed_xyz(slot_id=sid, xyz=raw, gaussian_state=state, deformation_trainer=deformation_trainer, playback_phase=phase)
            else:
                deformed, used, delta_norm = raw.copy(), False, 0.0
            any_points = any_points or raw.shape[0]>0
            payload[f'slot_{sid}_target']=np.array([target], dtype=object)
            payload[f'slot_{sid}_raw_xyz']=raw.astype(np.float32)
            payload[f'slot_{sid}_deformed_xyz']=deformed.astype(np.float32)
            payload[f'slot_{sid}_color']=color.astype(np.float32)
            payload[f'slot_{sid}_alpha']=alpha.astype(np.float32)
            payload[f'slot_{sid}_playback_phase']=np.array([phase],np.float32)
            payload[f'slot_{sid}_deformation_used']=np.array([1 if used else 0],np.int32)
            payload[f'slot_{sid}_pred_delta_norm']=np.array([delta_norm],np.float32)
            payload[f'slot_{sid}_raw_points']=np.array([raw.shape[0]],np.int32)
            payload[f'slot_{sid}_deformed_points']=np.array([deformed.shape[0]],np.int32)
            written_metrics[sid]=SlotOpen3DExportMetrics(True,str(self.export_path),sid,target,int(raw.shape[0]),int(deformed.shape[0]),bool(used),False,'pending')
        if not any_points:
            self.last_metrics=written_metrics
            return {'enabled':True,'export_path':str(self.export_path),'written':False,'reason':'no_points'}
        tmp=self.export_path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **payload)
        tmp.replace(self.export_path)
        self.last_write_time=now
        for m in written_metrics.values():
            m.written=True; m.reason='ok'
        self.last_metrics=written_metrics
        return {'enabled':True,'export_path':str(self.export_path),'written':True,'reason':'ok',
                'slot_0_raw_points':int(payload['slot_0_raw_points'][0]), 'slot_1_raw_points':int(payload['slot_1_raw_points'][0]),
                'slot_0_deformed_points':int(payload['slot_0_deformed_points'][0]), 'slot_1_deformed_points':int(payload['slot_1_deformed_points'][0])}
