# scripts/scorers_drills.py
"""
Luật chấm cho các bài BỘ PHÁP / PHÒNG THỦ trong thế thủ đối kháng:

  footwork     : nhảy tiến – lùi trong thế thủ
  duck         : thủ thế thụt người (hạ người né đòn rồi bật về)
  knee_chamber : thủ co gối (nâng gối chặn / chuẩn bị đá)

Giữ khung 10 = Chính xác 4 (1.6 / 1.2 / 0.8 / 0.4) + Biểu hiện 6 (2.0 / 1.6 / 1.4 / 1.0) của tác giả.
Tọa độ 3D MediaPipe cho góc khớp, tay thủ, độ nghiêng thân; 2D cho dịch chuyển của cả người (tọa độ 3D
của MediaPipe lấy gốc ở hông nên không thấy người di chuyển). Ngưỡng trong DrillRanges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from action_scorers import ALIASES, ScoreResult, register_scorer
from punch_scoring import _add_item, _band01, _clamp, _component, _interp_nan, _map01, _smooth, _vec_angle

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)


@dataclass
class DrillRanges:
    guard: Tuple[float, float] = (-0.90, -0.35)      # cổ tay cao hơn vai (× thân): ở hông ≈ −1, ngang cằm ≈ −0.2
    lean: Tuple[float, float] = (25.0, 10.0)
    sway: Tuple[float, float] = (0.20, 0.08)
    recovery: Tuple[float, float] = (0.30, 0.80)
    smooth: Tuple[float, float] = (0.35, 0.80)
    # bộ pháp
    fw_width: Tuple[float, float, float, float] = (0.6, 1.0, 2.6, 3.2)   # 2 chân / rộng vai (nhỏ nhất trong rep)
    fw_knee: Tuple[float, float] = (178.0, 160.0)                        # gối hơi chùng
    fw_reset: Tuple[float, float] = (0.50, 0.15)                         # độ rộng tấn cuối / đầu rep lệch
    fw_disp: Tuple[float, float] = (0.10, 0.45)                          # dịch chuyển hông (× thân, 2D)
    fw_speed: Tuple[float, float] = (0.8, 3.0)                           # tốc độ hông (× thân / s, 2D)
    fw_time: Tuple[float, float] = (0.90, 0.35)
    # thụt người
    dk_depth: Tuple[float, float] = (0.15, 0.50)                         # đầu hạ xuống (× thân, 2D)
    dk_knee: Tuple[float, float] = (170.0, 125.0)
    dk_lean: Tuple[float, float] = (55.0, 25.0)
    dk_speed: Tuple[float, float] = (0.8, 2.5)
    dk_up: Tuple[float, float] = (0.6, 2.0)
    dk_time: Tuple[float, float] = (0.80, 0.30)
    # co gối
    kc_height: Tuple[float, float] = (-0.70, -0.05)                      # gối so với hông (× thân, 3D; 0 = ngang hông)
    kc_fold: Tuple[float, float] = (120.0, 70.0)
    kc_speed: Tuple[float, float] = (1.5, 4.5)
    kc_rise: Tuple[float, float] = (0.40, 0.90)
    kc_time: Tuple[float, float] = (0.70, 0.25)


DR = DrillRanges()


def _prep(pose: Dict):
    w = _smooth(_interp_nan(pose["world"]), 5)
    k = _smooth(_interp_nan(pose["kpts"]), 5)
    Lt = float(np.median(np.linalg.norm(0.5 * (w[:, LS] + w[:, RS]) - 0.5 * (w[:, LH] + w[:, RH]), axis=1)))
    L2 = float(np.median(np.linalg.norm(0.5 * (k[:, LS] + k[:, RS]) - 0.5 * (k[:, LH] + k[:, RH]), axis=1)))
    return w, k, max(Lt, 1e-3), max(L2, 1e-3)


def _guard_h(w: np.ndarray, Lt: float) -> np.ndarray:
    """Tay thủ: độ cao trung bình 2 cổ tay so với vai (× thân, + = cao hơn). Không dùng khoảng cách tới cằm vì
    độ sâu 3D của MediaPipe (tay thủ đưa ra trước) sai số lớn."""
    return 0.5 * ((w[:, LS, 1] - w[:, LW, 1]) + (w[:, RS, 1] - w[:, RW, 1])) / Lt


def _lean(w: np.ndarray) -> np.ndarray:
    tr = 0.5 * (w[:, LS] + w[:, RS]) - 0.5 * (w[:, LH] + w[:, RH])
    return np.degrees(np.arccos(np.clip(-tr[:, 1] / (np.linalg.norm(tr, axis=1) + 1e-9), -1, 1)))


def _peak(sig: np.ndarray, sp: np.ndarray, fps: float, tol: float):
    """Như punch_scoring: lần tăng tín hiệu lớn nhất trong 1 s -> điểm đến; lùi từ đỉnh tốc độ -> bắt đầu."""
    T = len(sig)
    win = int(round(fps))
    rise_t = np.array([sig[t] - np.min(sig[max(0, t - win):t + 1]) for t in range(T)])
    t_max = int(np.argmax(rise_t))
    while t_max + 1 < T and sig[t_max + 1] >= sig[t_max]:
        t_max += 1
    t_imp = t_max
    while t_imp > 0 and sig[t_imp - 1] >= sig[t_max] - tol:
        t_imp -= 1
    lo = max(0, t_imp - win)
    t_v = lo + int(np.argmax(sp[lo:t_imp + 1]))
    t0, seen = 0, False
    for t in range(t_v, -1, -1):
        if sp[t] < 0.2 * sp[t_v]:
            t0, seen = t, True
            break
    base = float(np.median(sig[max(0, t0 - int(0.3 * fps)):t0 + 1])) if seen else float(np.min(sig[lo:t_imp + 1]))
    amp = float(sig[t_imp] - base)
    post = sig[t_imp:min(T, t_imp + int(round(1.2 * fps)) + 1)]
    rec = _clamp(float(sig[t_imp] - np.min(post)) / (amp + 1e-9), 0, 1)
    rec_ok = (T - 1 - t_imp) >= int(round(0.4 * fps)) or np.min(post) <= base + 0.3 * amp
    t_rec = next((t_imp + i for i, s in enumerate(post) if s <= base + 0.3 * amp), None)
    spd = sp[t0:t_imp + 1]
    tv = float(np.sum(np.abs(np.diff(spd)))) + abs(float(spd[0])) + abs(float(spd[-1]))
    return {"t0": t0, "t_imp": t_imp, "t_rec": t_rec, "seen": seen, "amp": amp, "rise": float(rise_t.max()),
            "rec": rec if rec_ok else None, "time": (t_imp - t0) / fps if seen else None,
            "smooth": _clamp(2.0 * float(sp[t_v]) / (tv + 1e-9), 0, 1), "vpeak": float(sp[t_v])}


def _result(action, acc, expr, phases, series, status, raw, hl, labels, names, warnings=()) -> ScoreResult:
    acc4, expr6 = _component(acc, 4.0), _component(expr, 6.0)
    top = sorted([d for d in acc + expr if d["assessable"] and d["deduct"] > 1e-6], key=lambda d: -d["deduct"])[:6]
    return ScoreResult(action=action, valid=True, total=acc4 + expr6, acc=acc4, expr=expr6, acc_items=acc,
                       expr_items=expr, top_issues=top, warnings=list(warnings), phases=phases, series=series,
                       status=status, raw=raw, highlight=hl, phase_labels=labels, phase_names=names)


def _rec_item(acc, code, label, p, tip):
    _add_item(acc, "accuracy", code, label, 0.8, None if p["rec"] is None else _map01(p["rec"], *DR.recovery) * 0.8,
              "video kết thúc trước khi trở về — không đánh giá" if p["rec"] is None
              else f"trở về≈{p['rec'] * 100:.0f}%", tip, key=True)


def _common_expr(expr, code, p, speed_rng, speed_txt, e2, time_rng):
    _add_item(expr, "expression", f"{code}_E1", "Tốc độ", 2.0, _map01(p["vpeak"], *speed_rng) * 2.0,
              speed_txt, "Di chuyển nhanh, bộc phát rồi dừng gọn.", key=True)
    expr.append(e2)
    _add_item(expr, "expression", f"{code}_E3", "Dứt khoát (thời gian thực hiện)", 1.4,
              None if p["time"] is None else _map01(p["time"], *time_rng) * 1.4,
              "video bắt đầu giữa động tác — không đánh giá" if p["time"] is None else f"≈{p['time']:.2f}s",
              "Thực hiện gọn, không chần chừ.")
    _add_item(expr, "expression", f"{code}_E4", "Độ mượt", 1.0, _map01(p["smooth"], *DR.smooth) * 1.0,
              f"độ mượt≈{p['smooth']:.2f}", "Một nhịp liền mạch, không khựng giữa chừng.")


# ========== nhảy tiến – lùi ==========
def score_footwork(pose: Dict, action: str, **_) -> ScoreResult:
    if pose.get("world") is None or pose["kpts"].shape[0] < 8:
        return ScoreResult(action=action, valid=False, reason="Video quá ngắn hoặc thiếu tọa độ 3D")
    w, k, Lt, L2 = _prep(pose)
    fps = float(pose["fps"])
    Ws = float(np.median(np.linalg.norm(w[:, LS] - w[:, RS], axis=1)))
    width = np.linalg.norm((w[:, LA] - w[:, RA])[:, [0, 2]], axis=1) / Ws
    knee = np.minimum(_vec_angle(w[:, LH] - w[:, LK], w[:, LA] - w[:, LK]),
                      _vec_angle(w[:, RH] - w[:, RK], w[:, RA] - w[:, RK]))
    hip2 = 0.5 * (k[:, LH] + k[:, RH]) / L2
    disp = np.linalg.norm(hip2 - hip2[0], axis=1)
    sp = np.r_[0.0, np.linalg.norm(np.diff(hip2, axis=0), axis=1) * fps]
    p = _peak(disp, sp, fps, tol=0.02)
    if p["rise"] < 0.05:
        return ScoreResult(action=action, valid=False, reason="Không thấy bước di chuyển.")
    guard = float(np.median(_guard_h(w, Lt)))
    lean = float(np.median(_lean(w)))
    n = max(1, int(0.15 * fps))
    reset = abs(float(np.median(width[-n:])) - float(np.median(width[:n]))) / (float(np.median(width[:n])) + 1e-9)
    acc: List[Dict] = []
    expr: List[Dict] = []
    a1 = 0.5 * _band01(float(np.min(width)), *DR.fw_width) + 0.5 * _map01(float(np.median(knee)), *DR.fw_knee)
    _add_item(acc, "accuracy", "FW_A1", "Giữ tấn đối kháng khi di chuyển", 1.6, a1 * 1.6,
              f"hai chân hẹp nhất≈{np.min(width):.1f}× vai, gối≈{np.median(knee):.0f}°",
              "Hai chân luôn cách nhau ~1,5 vai, không chụm hay bắt chéo; gối hơi chùng, nhún trên mũi chân.",
              key=True)
    _add_item(acc, "accuracy", "FW_A2", "Tay thủ", 1.2, _map01(guard, *DR.guard) * 1.2,
              f"cổ tay so với vai≈{guard:+.2f} thân", "Hai tay giữ trước cằm suốt lúc di chuyển.", key=True)
    _add_item(acc, "accuracy", "FW_A3", "Thân thẳng", 0.8, _map01(lean, *DR.lean) * 0.8, f"thân nghiêng≈{lean:.0f}°",
              "Thân thẳng, không chúi đầu theo bước chân.", key=True)
    _add_item(acc, "accuracy", "FW_A4", "Về lại thế thủ sau bước", 0.4, _map01(reset, *DR.fw_reset) * 0.4,
              f"độ rộng tấn cuối/đầu lệch≈{reset * 100:.0f}%", "Sau mỗi bước trở lại đúng khoảng cách hai chân.")
    e2: List[Dict] = []
    _add_item(e2, "expression", "FX_E2", "Biên độ bước", 1.6, _map01(p["amp"], *DR.fw_disp) * 1.6,
              f"dịch chuyển≈{p['amp']:.2f} thân", "Bước đủ xa để vào/ra khoảng cách đòn.", key=True)
    _common_expr(expr, "FX", p, DR.fw_speed, f"tốc độ hông≈{p['vpeak']:.1f} thân/giây", e2[0], DR.fw_time)
    return _result(action, acc, expr, {"t0": p["t0"], "t_peak": p["t_imp"], "t_end": p["t_rec"]},
                   {"sig": disp, "sig_name": "Dịch chuyển hông (× thân)", "speed": sp,
                    "speed_name": "Tốc độ hông (thân/s)"},
                   f"dịch {p['amp']:.2f} thân, tay thủ {guard:.2f}",
                   dict(p, guard=guard, lean=lean, width_min=float(np.min(width)), reset=reset), (LA, RA),
                   ("Bắt đầu", "Xa nhất", "Ổn định"), ("Thủ", "Di chuyển", "XA NHẤT", "Ổn định", "Thủ"))


# ========== thủ thế thụt người ==========
def score_duck(pose: Dict, action: str, **_) -> ScoreResult:
    if pose.get("world") is None or pose["kpts"].shape[0] < 8:
        return ScoreResult(action=action, valid=False, reason="Video quá ngắn hoặc thiếu tọa độ 3D")
    w, k, Lt, L2 = _prep(pose)
    fps = float(pose["fps"])
    sig = k[:, HEAD, 1] / L2                                   # 2D: y tăng = đầu hạ xuống
    sp = np.r_[0.0, np.abs(np.diff(sig)) * fps]
    p = _peak(sig, sp, fps, tol=0.02)
    if p["rise"] < 0.08:
        return ScoreResult(action=action, valid=False, reason="Không thấy động tác hạ người.")
    t = p["t_imp"]
    knee = float(min(_vec_angle(w[t, LH] - w[t, LK], w[t, LA] - w[t, LK]),
                     _vec_angle(w[t, RH] - w[t, RK], w[t, RA] - w[t, RK])))
    lean = float(_lean(w)[t])
    guard = float(_guard_h(w, Lt)[t])
    post = sp[t:min(len(sp), t + int(1.2 * fps))]
    up = float(np.max(post)) if post.size else 0.0
    acc: List[Dict] = []
    expr: List[Dict] = []
    a1 = 0.5 * _map01(p["amp"], *DR.dk_depth) + 0.5 * _map01(knee, *DR.dk_knee)
    _add_item(acc, "accuracy", "DK_A1", "Hạ thấp bằng gối", 1.6, a1 * 1.6,
              f"đầu hạ≈{p['amp']:.2f} thân, gối≈{knee:.0f}°",
              "Khuỵu gối để hạ đầu xuống dưới tầm đòn, không chỉ cúi lưng.", key=True)
    _add_item(acc, "accuracy", "DK_A2", "Tay thủ che mặt", 1.2, _map01(guard, *DR.guard) * 1.2,
              f"cổ tay so với vai≈{guard:+.2f} thân", "Hai tay vẫn giữ trước cằm khi hạ người.", key=True)
    _rec_item(acc, "DK_A3", "Bật về thế thủ", p, "Né xong bật ngay về thế thủ, sẵn sàng phản đòn.")
    _add_item(acc, "accuracy", "DK_A4", "Lưng không gập quá", 0.4, _map01(lean, *DR.dk_lean) * 0.4,
              f"thân nghiêng≈{lean:.0f}°", "Giữ lưng tương đối thẳng, mắt vẫn nhìn đối phương.")
    e2: List[Dict] = []
    _add_item(e2, "expression", "DX_E2", "Bật trở lên", 1.6, _map01(up, *DR.dk_up) * 1.6,
              f"tốc độ bật lên≈{up:.1f} thân/giây", "Đẩy chân bật lên nhanh sau khi né.", key=True)
    _common_expr(expr, "DX", p, DR.dk_speed, f"tốc độ hạ người≈{p['vpeak']:.1f} thân/giây", e2[0], DR.dk_time)
    return _result(action, acc, expr, {"t0": p["t0"], "t_peak": t, "t_end": p["t_rec"]},
                   {"sig": sig - np.min(sig), "sig_name": "Đầu hạ xuống (× thân)", "speed": sp,
                    "speed_name": "Tốc độ đầu (thân/s)"},
                   f"hạ {p['amp']:.2f} thân, gối {knee:.0f}°",
                   dict(p, knee=knee, lean=lean, guard=guard, up=up), (HEAD, LK, RK),
                   ("Bắt đầu", "Thấp nhất", "Về thủ"), ("Thủ", "Hạ người", "THẤP NHẤT", "Bật lên", "Thủ"))


# ========== thủ co gối ==========
def score_knee_chamber(pose: Dict, action: str, side: Optional[str] = None, **_) -> ScoreResult:
    if pose.get("world") is None or pose["kpts"].shape[0] < 8:
        return ScoreResult(action=action, valid=False, reason="Video quá ngắn hoặc thiếu tọa độ 3D")
    w, k, Lt, L2 = _prep(pose)
    fps = float(pose["fps"])
    hip_y = 0.5 * (w[:, LH, 1] + w[:, RH, 1])
    hL, hR = (hip_y - w[:, LK, 1]) / Lt, (hip_y - w[:, RK, 1]) / Lt
    if side not in ("L", "R"):
        side = "L" if np.ptp(hL) >= np.ptp(hR) else "R"
    H, K, A, sig = (LH, LK, LA, hL) if side == "L" else (RH, RK, RA, hR)
    sp = np.r_[0.0, np.linalg.norm(np.diff(w[:, K], axis=0), axis=1) * fps / Lt]
    p = _peak(sig, sp, fps, tol=0.03)
    if p["rise"] < 0.2:
        return ScoreResult(action=action, valid=False, reason="Không thấy động tác nâng gối.")
    t = p["t_imp"]
    fold = float(_vec_angle(w[t, H] - w[t, K], w[t, A] - w[t, K]))
    guard = float(_guard_h(w, Lt)[t])
    lean = float(_lean(w)[t])
    hip2 = 0.5 * (k[:, LH] + k[:, RH]) / L2
    sway = float(np.max(np.std(hip2, axis=0)))
    acc: List[Dict] = []
    expr: List[Dict] = []
    a1 = 0.6 * _map01(float(sig[t]), *DR.kc_height) + 0.4 * _map01(fold, *DR.kc_fold)
    _add_item(acc, "accuracy", "KC_A1", "Gối nâng cao, co chặt", 1.6, a1 * 1.6,
              f"gối so với hông≈{sig[t]:+.2f} thân, góc gối≈{fold:.0f}°",
              "Nâng gối lên ngang hông trở lên, gót chân sát đùi sau.", key=True)
    _add_item(acc, "accuracy", "KC_A2", "Tay thủ", 1.2, _map01(guard, *DR.guard) * 1.2,
              f"cổ tay so với vai≈{guard:+.2f} thân", "Hai tay giữ trước cằm, không vung tay lấy thăng bằng.",
              key=True)
    _rec_item(acc, "KC_A3", "Hạ chân về thế thủ", p, "Hạ chân về đúng vị trí thế thủ, không đổ người.")
    _add_item(acc, "accuracy", "KC_A4", "Thăng bằng chân trụ", 0.4,
              (0.5 * _map01(lean, *DR.lean) + 0.5 * _map01(sway, *DR.sway)) * 0.4,
              f"thân nghiêng≈{lean:.0f}°, lắc hông≈{sway:.2f}", "Chân trụ vững, thân thẳng.")
    e2: List[Dict] = []
    _add_item(e2, "expression", "KX_E2", "Biên độ nâng gối", 1.6, _map01(p["amp"], *DR.kc_rise) * 1.6,
              f"gối nâng≈{p['amp']:.2f} thân", "Nâng gối dứt khoát từ thế thủ lên cao.", key=True)
    _common_expr(expr, "KX", p, DR.kc_speed, f"tốc độ gối≈{p['vpeak']:.1f} thân/giây", e2[0], DR.kc_time)
    return _result(action, acc, expr, {"t0": p["t0"], "t_peak": t, "t_end": p["t_rec"]},
                   {"sig": sig, "sig_name": "Gối so với hông (× thân)", "speed": sp,
                    "speed_name": "Tốc độ gối (thân/s)"},
                   f"gối {sig[t]:+.2f} thân, {fold:.0f}°",
                   dict(p, side=side, fold=fold, guard=guard, lean=lean, sway=sway), (H, K, A),
                   ("Bắt đầu", "Gối cao nhất", "Về thủ"), ("Thủ", "Nâng gối", "CAO NHẤT", "Hạ chân", "Thủ"))


def _duck_signal(pose: Dict, action: str, **_) -> np.ndarray:
    k = _interp_nan(pose["kpts"])
    L2 = float(np.median(np.linalg.norm(0.5 * (k[:, LS] + k[:, RS]) - 0.5 * (k[:, LH] + k[:, RH]), axis=1)))
    return _smooth(k[:, HEAD, 1:2] / max(L2, 1e-3), 5)[:, 0]


def _knee_signal(pose: Dict, action: str, side: Optional[str] = None, **_) -> np.ndarray:
    w = _smooth(_interp_nan(pose["world"]), 5)
    Lt = float(np.median(np.linalg.norm(0.5 * (w[:, LS] + w[:, RS]) - 0.5 * (w[:, LH] + w[:, RH]), axis=1)))
    hip_y = 0.5 * (w[:, LH, 1] + w[:, RH, 1])
    hL, hR = (hip_y - w[:, LK, 1]) / Lt, (hip_y - w[:, RK, 1]) / Lt
    return hL if side == "L" else hR if side == "R" else np.maximum(hL, hR)


score_footwork.rep_cfg = {"use": ("legs",), "min_gap_s": 0.6, "pre_s": 0.6, "post_s": 0.8}
score_duck.rep_signal_fn = _duck_signal
score_duck.rep_cfg = {"min_gap_s": 0.8}
score_knee_chamber.rep_signal_fn = _knee_signal
score_knee_chamber.rep_cfg = {"min_gap_s": 0.8}
register_scorer("footwork", "Bộ pháp: nhảy tiến – lùi")(score_footwork)
register_scorer("duck", "Thủ thế thụt người (né hạ người)")(score_duck)
register_scorer("knee_chamber", "Thủ co gối")(score_knee_chamber)

ALIASES.update({"nhay_tien_lui": "footwork", "thu_the_thut_nguoi": "duck", "thu_co_goi": "knee_chamber"})
