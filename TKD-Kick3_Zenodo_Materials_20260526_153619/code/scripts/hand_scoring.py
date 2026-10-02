# scripts/hand_scoring.py
"""
Đo chỉ số cho các kỹ thuật TAY không phải đấm (đỡ, gạt, chém, chọc) — cùng cách làm với punch_scoring.py:
tọa độ 3D MediaPipe (world), làm mượt 5 frame, chuẩn hóa theo chiều dài thân (vai–hông).

Mỗi kỹ thuật có một TÍN HIỆU tìm "điểm đến" (giống điểm chạm của cú đấm):
  head_up : cổ tay cao hơn mũi bao nhiêu (đỡ cao — olgul makki)
  drop    : cổ tay thấp hơn vai bao nhiêu (gạt thấp — arae makki)
  reach   : khoảng cách vai–cổ tay / chiều dài tay (chém, chọc, đỡ cạnh tay)
Điểm đến = frame đầu tiên đạt đỉnh của lần tăng tín hiệu lớn nhất trong 1 s (như punch_scoring).

Hàm chấm điểm nằm ở scorers_hands.py; file này chỉ đo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional

import numpy as np

from punch_scoring import _clamp, _interp_nan, _smooth, _straightness, _vec_angle, _yaw_deg

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
SIGNALS = ("head_up", "drop", "reach", "out", "thrust")


@dataclass
class HandMetrics:
    signal: str
    side: str
    valid: bool
    invalid_reason: str
    T: int
    fps: float
    t0: int
    t_imp: int
    t_rec: Optional[int]
    baseline_frames: int
    rise: float                 # biên độ tín hiệu trong 1 s trước điểm đến
    # hình dạng tay tại điểm đến (3D)
    elbow_flex_deg: float       # 180 = duỗi thẳng
    upper_arm_elev_deg: float   # cánh tay trên so với thân: 0 = khép sườn, 90 = ngang vai, 180 = thẳng lên
    wrist_vs_shoulder: float    # cổ tay cao hơn vai (× thân, + = cao hơn)
    wrist_vs_head: float        # cổ tay cao hơn mũi
    wrist_vs_hip: float         # cổ tay cao hơn tâm hông
    elbow_vs_shoulder: float    # khuỷu cao hơn vai
    forearm_tilt_deg: float     # cẳng tay so với mặt phẳng ngang (0 = nằm ngang, + = cổ tay cao hơn khuỷu)
    reach: float                # vai–cổ tay / chiều dài tay
    lateral: float              # cổ tay lệch về phía cùng bên so với giữa vai (× rộng vai; âm = sang bên kia)
    path_straightness: float
    # tư thế trước khi ra đòn (trong 1 s trước điểm đến)
    start_wrist_vs_shoulder: float  # cổ tay cao nhất so với vai
    start_lateral_min: float        # cổ tay sang bên kia thân nhiều nhất (× rộng vai)
    start_wrist_vs_head_min: float  # cổ tay thấp nhất so với mũi
    swing_lateral: float            # quãng quét ngang của cổ tay (× rộng vai)
    # tay còn lại tại điểm đến
    other_to_hip: float         # cổ tay còn lại cách hông cùng bên (× thân)
    other_to_chin: float
    other_to_elbow: float       # cổ tay còn lại cách khuỷu tay ra đòn (đỡ dưới khuỷu khi chọc)
    other_to_sternum: float     # cách giữa ngực (đỡ cạnh tay 2 tay)
    # chung
    recovery_ratio: Optional[float]
    stop_disp: Optional[float]  # cổ tay dịch chuyển bao xa trong 0.25 s sau điểm đến (× thân) — dừng dứt khoát
    trunk_lean_deg: float
    hip_sway: float
    shoulder_rot_deg: Optional[float]
    peak_speed: float
    peak_rot_speed: Optional[float]
    exec_time: Optional[float]
    smoothness: float
    wrist_visibility: float


def _invalid(signal, side, T, fps, reason) -> HandMetrics:
    kw = {f.name: float("nan") for f in fields(HandMetrics)}
    kw.update(signal=signal, side=side, valid=False, invalid_reason=reason, T=T, fps=fps, t0=0, t_imp=0, t_rec=None,
              baseline_frames=0, recovery_ratio=None, stop_disp=None, shoulder_rot_deg=None, peak_rot_speed=None, exec_time=None,
              wrist_visibility=0.0)
    return HandMetrics(**kw)


def technique_signals(w: np.ndarray, signal: str, Lt: float):
    """Tín hiệu tìm điểm đến cho tay trái, tay phải (w: tọa độ world đã làm mượt).
    head_up: cổ tay cao hơn mũi; drop: cổ tay thấp hơn vai; reach: vai–cổ tay / dài tay;
    out: cổ tay vươn ngang ra phía cùng bên (× rộng vai), trừ điểm khi tay thấp hơn vai quá 0.3 thân;
    thrust: reach, trừ điểm khi tay thấp hơn vai quá 0.6 thân.
    (out/thrust: tay buông xuôi bên hông cũng duỗi thẳng nên reach thuần không phân biệt được.)"""
    sh_mid = 0.5 * (w[:, LS] + w[:, RS])
    Ws = float(np.median(np.linalg.norm(w[:, LS] - w[:, RS], axis=1))) + 1e-9

    def sig_of(S, OS, Wr, E):
        below = (w[:, Wr, 1] - w[:, S, 1]) / Lt                 # world y hướng xuống: > 0 là thấp hơn vai
        if signal == "head_up":
            return (w[:, HEAD, 1] - w[:, Wr, 1]) / Lt
        if signal == "drop":
            return below
        if signal == "out":
            ax = w[:, S] - w[:, OS]
            ax = ax / (np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9)
            lat = np.einsum("ij,ij->i", w[:, Wr] - sh_mid, ax) / Ws
            return lat - 2.0 * np.maximum(0.0, below - 0.3)
        arm = np.linalg.norm(w[:, E] - w[:, S], axis=1) + np.linalg.norm(w[:, Wr] - w[:, E], axis=1)
        reach = np.linalg.norm(w[:, Wr] - w[:, S], axis=1) / np.median(arm)
        return reach - 1.5 * np.maximum(0.0, below - 0.6) if signal == "thrust" else reach
    return sig_of(LS, RS, LW, LE), sig_of(RS, LS, RW, RE)


def rep_signal(pose: Dict, signal: str, side: Optional[str] = None) -> np.ndarray:
    """Tín hiệu tách rep trong video dài: tín hiệu kỹ thuật của tay (hoặc lớn hơn của 2 tay)."""
    w = _smooth(_interp_nan(pose["world"]), 5)
    Lt = float(np.median(np.linalg.norm(0.5 * (w[:, LS] + w[:, RS]) - 0.5 * (w[:, LH] + w[:, RH]), axis=1)))
    sL, sR = technique_signals(w, signal, max(Lt, 1e-3))
    return sL if side == "L" else sR if side == "R" else np.maximum(sL, sR)


def alternating_signal(pose: Dict, *_, **__) -> np.ndarray:
    """|độ vươn tay trái − tay phải|: đấm xen kẽ 2 tay -> mỗi cú đấm là một đỉnh (độ vươn lớn nhất của 2 tay
    gần như không đổi khi một tay ra, một tay thu cùng lúc)."""
    w = _smooth(_interp_nan(pose["world"]), 5)
    Lt = float(np.median(np.linalg.norm(0.5 * (w[:, LS] + w[:, RS]) - 0.5 * (w[:, LH] + w[:, RH]), axis=1)))
    sL, sR = technique_signals(w, "reach", max(Lt, 1e-3))
    return np.abs(sL - sR)


def alternating_side(pose: Dict, t: int) -> str:
    """Tay đang ra đòn tại frame t của bài đấm xen kẽ: tay vươn xa hơn."""
    w = _smooth(_interp_nan(pose["world"]), 5)
    Lt = float(np.median(np.linalg.norm(0.5 * (w[:, LS] + w[:, RS]) - 0.5 * (w[:, LH] + w[:, RH]), axis=1)))
    sL, sR = technique_signals(w, "reach", max(Lt, 1e-3))
    return "L" if sL[t] >= sR[t] else "R"


def build_hand_metrics(kpts: np.ndarray, vis: np.ndarray, fps: float, world: np.ndarray, signal: str,
                       side: Optional[str] = None, need_rise: float = 0.4, tol: float = 0.03,
                       debug: Optional[Dict] = None) -> HandMetrics:
    if signal not in SIGNALS:
        raise ValueError(f"signal phải thuộc {SIGNALS}")
    fps = float(fps) if fps and fps > 0 else 30.0
    T = int(kpts.shape[0])
    if T < 8 or world is None or world.shape[0] != T:
        return _invalid(signal, side or "?", T, fps, "Video quá ngắn hoặc thiếu tọa độ 3D")
    w = _smooth(_interp_nan(world), 5)
    k2 = _smooth(_interp_nan(kpts), 5)
    sh_mid = 0.5 * (w[:, LS] + w[:, RS])
    hip_mid = 0.5 * (w[:, LH] + w[:, RH])
    Lt = float(np.median(np.linalg.norm(sh_mid - hip_mid, axis=1)))
    Ws = float(np.median(np.linalg.norm(w[:, LS] - w[:, RS], axis=1)))
    if not np.isfinite(Lt) or Lt < 1e-3 or Ws < 1e-3:
        return _invalid(signal, side or "?", T, fps, "Không ước lượng được kích thước cơ thể")

    sigL, sigR = technique_signals(w, signal, Lt)
    if side not in ("L", "R"):
        side = "R" if np.ptp(sigR) >= np.ptp(sigL) else "L"
    if side == "R":
        S, OS, E, Wr, OW, H, OH, sig = RS, LS, RE, RW, LW, RH, LH, sigR
    else:
        S, OS, E, Wr, OW, H, OH, sig = LS, RS, LE, LW, RW, LH, RH, sigL
    rot_sign = 1.0 if side == "R" else -1.0

    win = int(round(1.0 * fps))
    rise_t = np.array([sig[t] - np.min(sig[max(0, t - win):t + 1]) for t in range(T)])
    t_best = int(np.argmax(rise_t))
    t_max = t_best
    while t_max + 1 < T and sig[t_max + 1] >= sig[t_max]:
        t_max += 1
    t_imp = t_max
    while t_imp > 0 and sig[t_imp - 1] >= sig[t_max] - tol:
        t_imp -= 1

    sp = np.linalg.norm(np.diff(w[:, Wr], axis=0), axis=1) * fps / Lt
    sp = np.r_[sp[:1], sp]
    lo = max(0, t_imp - win)
    t_vpeak = lo + int(np.argmax(sp[lo:t_imp + 1]))
    v_peak = float(sp[t_vpeak])
    if debug is not None:
        debug.update(side=side, sig=sig.copy(), speed=sp.copy())

    t0, onset_seen = 0, False
    for t in range(t_vpeak, -1, -1):
        if sp[t] < 0.2 * v_peak:
            t0, onset_seen = t, True
            break
    if t_imp <= t0 + 1 or rise_t[t_best] < need_rise:
        return _invalid(signal, side, T, fps,
                        "Không thấy một động tác trọn vẹn: video bắt đầu/kết thúc giữa động tác, hoặc góc quay "
                        "khiến không đo được tay. Nên quay chính diện hoặc chéo ~45°.")
    b0 = max(0, t0 - int(round(0.3 * fps)))
    base = slice(b0, t0 + 1)
    baseline_frames = (t0 - b0 + 1) if onset_seen else 0

    # trục ngang của vai (từ vai bên kia sang vai ra đòn) để đo lệch ngang
    ax = w[:, S] - w[:, OS]
    ax = ax / (np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9)
    lateral = np.einsum("ij,ij->i", w[:, Wr] - sh_mid, ax) / Ws

    def up(a, b, t):  # a cao hơn b (× thân)
        return float((w[t, b, 1] - w[t, a, 1]) / Lt)

    fa = w[t_imp, Wr] - w[t_imp, E]
    tilt = float(np.degrees(np.arcsin(_clamp(-fa[1] / (np.linalg.norm(fa) + 1e-9), -1.0, 1.0))))
    arm_len = np.median(np.linalg.norm(w[:, E] - w[:, S], axis=1) + np.linalg.norm(w[:, Wr] - w[:, E], axis=1))
    pre = slice(max(0, t_imp - win), t_imp + 1)
    wvs = (w[:, S, 1] - w[:, Wr, 1]) / Lt
    wvh = (w[:, HEAD, 1] - w[:, Wr, 1]) / Lt
    chin = w[t_imp, HEAD] + np.array([0.0, 0.08 * Lt, 0.0])
    sternum = sh_mid[t_imp] + 0.35 * (hip_mid[t_imp] - sh_mid[t_imp])

    yaw_s = rot_sign * _yaw_deg(w, LS, RS)
    if onset_seen:
        rot_s = float(np.ptp(yaw_s[pre]))
        rot_speed = float(np.max(np.abs(np.diff(yaw_s[max(0, t0 - 1):t_imp + 1])))) * fps
    else:
        rot_s = rot_speed = None

    # thu về: tín hiệu quay lại mức ban đầu trong 1.2 s
    sig_base = float(np.median(sig[base])) if baseline_frames >= 3 else float(np.min(sig[pre]))
    amp = float(sig[t_imp] - sig_base)
    t_end = min(T - 1, t_imp + int(round(1.2 * fps)))
    post = sig[t_imp:t_end + 1]
    rec = _clamp(float(sig[t_imp] - np.min(post)) / (amp + 1e-9), 0.0, 1.0)
    t_rec = next((t_imp + i for i, s_ in enumerate(post) if s_ <= sig_base + 0.3 * amp), None)
    post_frames = T - 1 - t_imp
    recovery = rec if (t_rec is not None or post_frames >= int(round(0.4 * fps))) else None
    t_hold = min(T - 1, t_imp + int(round(0.25 * fps)))
    stop_disp = (float(np.max(np.linalg.norm(w[t_imp:t_hold + 1, Wr] - w[t_imp, Wr], axis=1))) / Lt
                 if t_hold - t_imp >= int(round(0.15 * fps)) else None)

    sh2 = 0.5 * (k2[:, LS] + k2[:, RS])
    hp2 = 0.5 * (k2[:, LH] + k2[:, RH])
    L2 = float(np.median(np.linalg.norm(sh2 - hp2, axis=1))) + 1e-6
    tv = sh2[t_imp] - hp2[t_imp]
    lean = float(np.degrees(np.arctan2(abs(tv[0]), -tv[1])))
    sway = float(np.max(np.std(hp2[b0:t_end + 1] / L2, axis=0)))

    spd = sp[t0:t_imp + 1]
    tv_sp = float(np.sum(np.abs(np.diff(spd)))) + abs(float(spd[0])) + abs(float(spd[-1]))
    smooth = _clamp(2.0 * v_peak / (tv_sp + 1e-9), 0.0, 1.0)
    vis_w = float(np.mean(vis[:, Wr] > 0.5)) if vis is not None and vis.shape[0] == T else 1.0

    return HandMetrics(
        signal=signal, side=side, valid=True, invalid_reason="", T=T, fps=fps,
        t0=int(t0), t_imp=int(t_imp), t_rec=t_rec, baseline_frames=int(baseline_frames),
        rise=float(rise_t[t_best]),
        elbow_flex_deg=float(_vec_angle(w[t_imp, S] - w[t_imp, E], w[t_imp, Wr] - w[t_imp, E])),
        upper_arm_elev_deg=float(_vec_angle(w[t_imp, E] - w[t_imp, S], w[t_imp, H] - w[t_imp, S])),
        wrist_vs_shoulder=up(Wr, S, t_imp), wrist_vs_head=up(Wr, HEAD, t_imp),
        wrist_vs_hip=float((hip_mid[t_imp, 1] - w[t_imp, Wr, 1]) / Lt),
        elbow_vs_shoulder=up(E, S, t_imp), forearm_tilt_deg=tilt,
        reach=float(np.linalg.norm(w[t_imp, Wr] - w[t_imp, S]) / arm_len),
        lateral=float(lateral[t_imp]),
        path_straightness=float(_straightness(w[t0:t_imp + 1, Wr])),
        start_wrist_vs_shoulder=float(np.max(wvs[pre])),
        start_lateral_min=float(np.min(lateral[pre])),
        start_wrist_vs_head_min=float(np.min(wvh[pre])),
        swing_lateral=float(np.ptp(lateral[pre])),
        other_to_hip=float(np.linalg.norm(w[t_imp, OW] - w[t_imp, OH]) / Lt),
        other_to_chin=float(np.linalg.norm(w[t_imp, OW] - chin) / Lt),
        other_to_elbow=float(np.linalg.norm(w[t_imp, OW] - w[t_imp, E]) / Lt),
        other_to_sternum=float(np.linalg.norm(w[t_imp, OW] - sternum) / Lt),
        recovery_ratio=recovery, stop_disp=stop_disp, trunk_lean_deg=lean, hip_sway=sway, shoulder_rot_deg=rot_s,
        peak_speed=v_peak, peak_rot_speed=rot_speed,
        exec_time=float((t_imp - t0) / fps) if onset_seen else None,
        smoothness=float(smooth), wrist_visibility=vis_w,
    )


def metrics_dict(m: HandMetrics) -> Dict:
    return asdict(m)
