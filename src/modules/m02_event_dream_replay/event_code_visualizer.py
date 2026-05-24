from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from src.platform.gui.opencv_gui_thread import submit_cv2_frame, close_cv2_window, get_cv2_last_key


@dataclass
class EventCodeVisualizerV2Config:
    enabled: bool = True
    window_name: str = "event code / slot vocabulary"
    width: int = 1500
    height: int = 980
    show_every_steps: int = 1
    delay_ms: int = 1
    max_slots: int = 10
    max_events: int = 14


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().reshape(-1)[0].cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _safe_text(x: Any, max_len: int = 160) -> str:
    try:
        s = str(x or "")
    except Exception:
        s = ""
    s = s.replace("\n", " ").replace("\r", " ")
    return s[:max_len]


class EventCodeVisualizerV2:
    """
    Second-level visualizer.

    Level 1 window:
        decoded image / depth / mask / slot latent heatmap

    Level 2 window:
        SlotVocabulary + event-code sentences:
            SLOT_1 -> OBJ_001 -> latent signature -> event stream

    It is a debug viewer only; it does not train anything.
    """

    def __init__(self, cfg: Optional[EventCodeVisualizerV2Config] = None, **kwargs):
        self.cfg = cfg or EventCodeVisualizerV2Config(**kwargs)
        self.window_name = self.cfg.window_name
        self.width = int(self.cfg.width)
        self.height = int(self.cfg.height)
        self.created = False
        self._display_size = (max(self.width, 1200), max(self.height, 760))
        self._freeratio_applied = False

    def close(self):
        close_cv2_window(self.window_name)
        self.created = False

    def _ensure(self):
        # Window creation/resize happens in the dedicated OpenCV GUI thread.
        self.created = True

    def _fit_to_window(self, frame: np.ndarray) -> np.ndarray:
        target_w = max(int(getattr(self, "width", 1200)), 1200)
        target_h = max(int(getattr(self, "height", 760)), 760)

        if not hasattr(self, "_display_size"):
            self._display_size = (target_w, target_h)

        target_w, target_h = self._display_size
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            interp = cv2.INTER_AREA if (frame.shape[1] > target_w or frame.shape[0] > target_h) else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)
        return frame

    def _panel(self, w: int, h: int, title: str) -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (8, 12, 20)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (65, 85, 110), 1)
        cv2.putText(img, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (240, 245, 255), 1, cv2.LINE_AA)
        return img

    def _get_entries(self, event_memory: Any) -> Dict[int, Dict[str, Any]]:
        try:
            vocab = getattr(event_memory, "slot_vocabulary", None)
            entries = getattr(vocab, "entries", {}) if vocab is not None else {}
            if isinstance(entries, dict):
                return {int(k): v for k, v in entries.items()}
        except Exception:
            pass
        return {}

    def _get_events(self, event_memory: Any) -> List[Dict[str, Any]]:
        try:
            events = getattr(event_memory, "events", [])
            return list(events)
        except Exception:
            return []

    def _slot_vocabulary_panel(self, event_memory: Any, obj: Optional[Dict[str, Any]], w: int, h: int) -> np.ndarray:
        img = self._panel(w, h, "Level 2: SlotVocabulary — internal words for latent slots")
        entries = self._get_entries(event_memory)
        active_slot = int(_as_float((obj or {}).get("active_slot_index"), 0.0)) if isinstance(obj, dict) else 0

        if not entries:
            cv2.putText(img, "SlotVocabulary empty: wait until an event changes a slot.", (18, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 170, 200), 1, cv2.LINE_AA)
            return img

        n = min(int(self.cfg.max_slots), max(entries.keys()) + 1 if entries else 0)
        top = 54
        row_h = max(48, int((h - top - 12) / max(1, n)))
        for i in range(n):
            e = entries.get(i, {})
            y0 = top + i * row_h
            y1 = min(y0 + row_h - 6, h - 12)
            token = _safe_text(e.get("token", f"OBJ_{i:03d}"), 32)
            conf = float(e.get("confidence_ema", 0.0) or 0.0)
            z_norm = float(e.get("z_norm_ema", 0.0) or 0.0)
            src = _safe_text(e.get("dominant_source", "unknown"), 32)
            shape = _safe_text(e.get("shape_name", "unknown"), 32)
            updates = int(e.get("updates", 0) or 0)
            top_dims = e.get("top_dims", [])
            if not isinstance(top_dims, list):
                top_dims = []
            top_txt = ",".join(str(int(x)) for x in top_dims[:6])
            active = (i == active_slot)

            bg = (18, 34, 28) if active else (12, 18, 28)
            border = (80, 240, 130) if active else (55, 75, 100)
            cv2.rectangle(img, (12, y0), (w - 12, y1), bg, -1)
            cv2.rectangle(img, (12, y0), (w - 12, y1), border, 2 if active else 1)

            cv2.putText(img, f"SLOT_{i} -> {token}", (24, y0 + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, border, 1, cv2.LINE_AA)
            cv2.putText(img, f"conf={conf:.2f} z={z_norm:.2f} src={src} shape={shape} updates={updates}",
                        (24, y0 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 230, 245), 1, cv2.LINE_AA)
            cv2.putText(img, f"latent signature top dims: [{top_txt}]",
                        (w - 350, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 205, 245), 1, cv2.LINE_AA)
        return img


    def _get_sentence_memory_view(self, event_memory: Any) -> Dict[str, Any]:
        try:
            sm = getattr(event_memory, "sentence_memory", None)
            if sm is not None and hasattr(sm, "get_state_view"):
                return sm.get_state_view(max_sentences=10, max_episodes=6)
        except Exception:
            pass
        return {}

    def _event_stream_panel(self, event_memory: Any, obj: Optional[Dict[str, Any]], w: int, h: int) -> np.ndarray:
        img = self._panel(w, h, "Level 2: event code stream — DNA-like sentences")
        events = self._get_events(event_memory)
        sentence_view = self._get_sentence_memory_view(event_memory)
        latest_sentence = ""
        try:
            latest_sentence = str(
                (obj or {}).get("semantic_sentence", "")
                or (obj or {}).get("semantic_code_sentence", "")
                or (obj or {}).get("event_code_sentence", "")
                or ""
            )
        except Exception:
            latest_sentence = ""

        y = 58
        ep_summary = str(sentence_view.get("latest_episode_summary", "") or "")
        sent_count = int(sentence_view.get("sentence_count", 0) or 0)
        ep_count = int(sentence_view.get("episode_count", 0) or 0)
        cv2.putText(img, f"events={len(events)} sentences={sent_count} episodes={ep_count} | latest:", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 245), 1, cv2.LINE_AA)
        y += 24
        if latest_sentence:
            for start in range(0, len(latest_sentence), 125):
                cv2.putText(img, latest_sentence[start:start + 125], (26, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 230, 150), 1, cv2.LINE_AA)
                y += 19
        else:
            cv2.putText(img, "no active event sentence yet", (26, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (145, 160, 185), 1, cv2.LINE_AA)
            y += 24

        if ep_summary:
            cv2.putText(img, "latest episode:", (16, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 205, 245), 1, cv2.LINE_AA)
            y += 22
            for start in range(0, len(ep_summary), 125):
                cv2.putText(img, ep_summary[start:start + 125], (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 210, 245), 1, cv2.LINE_AA)
                y += 17
                if y > h - 34:
                    break

        cv2.line(img, (16, y + 8), (w - 16, y + 8), (45, 60, 80), 1)
        y += 32

        recent = events[-int(self.cfg.max_events):]
        if not recent:
            cv2.putText(img, "event memory empty", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (145, 160, 185), 1, cv2.LINE_AA)
            return img

        for ev in reversed(recent):
            sent = _safe_text(ev.get("semantic_sentence", "") or ev.get("semantic_code_sentence", "") or ev.get("sentence", ""), 180)
            token = _safe_text(ev.get("slot_token", ""), 32)
            kind = _safe_text(ev.get("kind", ""), 32)
            step = ev.get("step", "?")
            slot = ev.get("slot", "?")
            dz = float(ev.get("delta_norm", 0.0) or 0.0)
            contact = float(ev.get("contact_norm", 0.0) or 0.0)
            color = (210, 230, 245)
            if "contact" in kind:
                color = (120, 230, 255)
            elif "dream" in kind:
                color = (230, 160, 255)
            cv2.putText(img, f"t={step} SLOT_{slot} {token} {kind} dz={dz:.3f} touch={contact:.3f}",
                        (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
            y += 17
            cv2.putText(img, sent[:150], (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 195, 220), 1, cv2.LINE_AA)
            y += 22
            if y > h - 18:
                break
        return img

    def _slot_passport_panel(self, event_memory: Any, obj: Optional[Dict[str, Any]], w: int, h: int) -> np.ndarray:
        img = self._panel(w, h, "Selected/active slot passport")
        entries = self._get_entries(event_memory)
        active_slot = int(_as_float((obj or {}).get("active_slot_index"), 0.0)) if isinstance(obj, dict) else 0
        e = entries.get(active_slot, {})

        y = 60
        if not e:
            cv2.putText(img, f"SLOT_{active_slot}: no vocabulary entry yet", (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 170, 200), 1, cv2.LINE_AA)
            return img

        lines = [
            f"address: SLOT_{active_slot}",
            f"internal word: {e.get('token', f'OBJ_{active_slot:03d}')}",
            f"dominant source: {e.get('dominant_source', 'unknown')}",
            f"formed_by: {', '.join(e.get('formed_by', [])) if isinstance(e.get('formed_by', []), list) else e.get('formed_by', '')}",
            f"confidence_ema: {float(e.get('confidence_ema', 0.0) or 0.0):.3f}",
            f"z_norm_ema: {float(e.get('z_norm_ema', 0.0) or 0.0):.3f}",
            f"shape: {e.get('shape_name', 'unknown')}  size={float(e.get('size', 0.0) or 0.0):.2f}  stability={float(e.get('stability', 0.0) or 0.0):.2f}",
            f"color_rgb: {np.round(np.asarray(e.get('color_rgb', [0,0,0]), dtype=float), 2).tolist()}",
            f"top latent dims: {e.get('top_dims', [])[:10]}",
            f"event_counts: {e.get('event_counts', {})}",
            f"last_event: {_safe_text(e.get('last_event_sentence', ''), 120)}",
        ]

        for line in lines:
            cv2.putText(img, line[:145], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (220, 230, 245), 1, cv2.LINE_AA)
            y += 25
            if y > h - 15:
                break
        return img

    def draw(self, obj: Optional[Dict[str, Any]], event_memory: Any = None, global_step: int = 0) -> None:
        if not bool(getattr(self.cfg, "enabled", True)):
            return
        self._ensure()

        w = max(self.width, 1200)
        h = max(self.height, 760)
        header_h = 58
        left_w = int(w * 0.52)
        right_w = w - left_w
        mid_h = int((h - header_h) * 0.58)
        bot_h = h - header_h - mid_h

        header = np.zeros((header_h, w, 3), dtype=np.uint8)
        header[:] = (6, 10, 16)
        cv2.putText(header, "Event Code Visualizer V2 — Level 2 semantic code layer", (14, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 245, 255), 1, cv2.LINE_AA)
        scenario_active = bool(_as_float((obj or {}).get("scenario_active"), 0.0) > 0.5) if isinstance(obj, dict) else False
        scenario_cursor = _as_float((obj or {}).get("scenario_cursor"), 0.0) if isinstance(obj, dict) else 0.0
        scenario_len = _as_float((obj or {}).get("scenario_sequence_len"), 0.0) if isinstance(obj, dict) else 0.0
        scenario_sentence = str((obj or {}).get("scenario_sentence", "") or "") if isinstance(obj, dict) else ""
        cv2.putText(header, f"step={int(global_step)} | SLOT_N = memory address | OBJ_NNN = internal word | z_obj[128] = actual meaning",
                    (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 195, 230), 1, cv2.LINE_AA)
        if scenario_active:
            cv2.putText(header, f"SCENARIO DECODER active cursor={scenario_cursor:.0f}/{scenario_len:.0f} | {scenario_sentence[:95]}",
                        (720, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 190, 120), 1, cv2.LINE_AA)
        neural_active = bool(_as_float((obj or {}).get("neural_event_decoder_active"), 0.0) > 0.5) if isinstance(obj, dict) else False
        if neural_active:
            cv2.putText(header, "NEURAL EVENT DECODER: predicted z_after from sentence/code", (720, 51),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 255, 170), 1, cv2.LINE_AA)
        mind_active = bool(_as_float((obj or {}).get("inner_mind_active"), 0.0) > 0.5) if isinstance(obj, dict) else False
        if mind_active:
            mind_sentence = str((obj or {}).get("inner_mind_selected_sentence", "") or "") if isinstance(obj, dict) else ""
            mind_score = _as_float((obj or {}).get("inner_mind_selected_score"), 0.0) if isinstance(obj, dict) else 0.0
            cv2.putText(header, f"INNER SCENARIO MIND score={mind_score:.3f} | {mind_sentence[:85]}", (720, 71),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
        action_active = bool(_as_float((obj or {}).get("inner_action_active"), 0.0) > 0.5) if isinstance(obj, dict) else False
        if action_active:
            action_conf = _as_float((obj or {}).get("inner_action_confidence"), 0.0) if isinstance(obj, dict) else 0.0
            slot_token = str((obj or {}).get("inner_action_slot_token", "") or "") if isinstance(obj, dict) else ""
            cv2.putText(header, f"INNER ACTION INTENT conf={action_conf:.3f} token={slot_token}", (720, 91),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 245, 170), 1, cv2.LINE_AA)
        outcome_active = bool(_as_float((obj or {}).get("inner_outcome_active"), 0.0) > 0.5) if isinstance(obj, dict) else False
        if outcome_active:
            err = _as_float((obj or {}).get("inner_outcome_error"), 0.0) if isinstance(obj, dict) else 0.0
            succ = _as_float((obj or {}).get("inner_outcome_success_ema"), 0.0) if isinstance(obj, dict) else 0.0
            cv2.putText(header, f"INNER OUTCOME err={err:.3f} success_ema={succ:.3f}", (720, 111),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 160), 1, cv2.LINE_AA)
        trust_active = bool(_as_float((obj or {}).get("inner_trust_active"), 0.0) > 0.5) if isinstance(obj, dict) else False
        if trust_active:
            trust = _as_float((obj or {}).get("inner_trust_value"), 0.0) if isinstance(obj, dict) else 0.0
            alpha = _as_float((obj or {}).get("inner_trust_alpha"), 0.0) if isinstance(obj, dict) else 0.0
            reason = str((obj or {}).get("inner_trust_reason", "") or "") if isinstance(obj, dict) else ""
            cv2.putText(header, f"INNER TRUST trust={trust:.3f} alpha={alpha:.3f} {reason[:45]}", (720, 131),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 255, 210), 1, cv2.LINE_AA)
        trace_path = str((obj or {}).get("inner_real_action_trace_path", "") or "") if isinstance(obj, dict) else ""
        if trace_path:
            cv2.putText(header, f"TRACE path: {trace_path[-70:]}", (720, 151),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 210, 255), 1, cv2.LINE_AA)
        passport_active = bool(_as_float((obj or {}).get("passport_active"), 0.0) > 0.5) if isinstance(obj, dict) else False
        if passport_active:
            ptoken = str((obj or {}).get("passport_token", "") or "") if isinstance(obj, dict) else ""
            pcount = _as_float((obj or {}).get("passport_count"), 0.0) if isinstance(obj, dict) else 0.0
            pscore = _as_float((obj or {}).get("passport_dynamic_score"), 0.0) if isinstance(obj, dict) else 0.0
            cv2.putText(header, f"DYNAMIC PASSPORT {ptoken} count={pcount:.0f} dyn={pscore:.3f}", (720, 171),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 210, 255), 1, cv2.LINE_AA)

        vocab_panel = self._slot_vocabulary_panel(event_memory, obj, left_w, mid_h)
        event_panel = self._event_stream_panel(event_memory, obj, right_w, mid_h)
        pass_panel = self._slot_passport_panel(event_memory, obj, w, bot_h)

        top = np.concatenate([vocab_panel, event_panel], axis=1)
        frame = np.concatenate([header, top, pass_panel], axis=0)

        frame = self._fit_to_window(frame)
        submit_cv2_frame(self.window_name, frame, max(int(getattr(self, "width", 1200)), 1200), max(int(getattr(self, "height", 760)), 760))
        return
