# scripts/punch_scoring.py
"""
Chấm điểm động tác ĐẤM theo cùng khung với scoring_api.py (luật đá):
    Tổng 10 = Chính xác 4 + Biểu hiện 6  (mượn cấu trúc điểm Poomsae WT, giống tác giả bộ TKD-Kick3)

Loại đấm hỗ trợ:
- hook     : đấm móc (dollyeo jireugi / hook) — đã thử trên 24 clip "Dam moc phai"
- straight : đấm thẳng (jireugi / jab-cross)   — CHƯA có video để kiểm thử, ngưỡng là ước lượng

Kiểu thủ (guard):
- chin  : thủ đối kháng, 2 nắm tay ở cằm; tay không đấm giữ ở cằm khi ra đòn (mặc định)
- waist : kiểu Poomsae, bắt đầu nắm tay ở thắt lưng; tay không đấm kéo về hông khi ra đòn

Input:
- kpts  (T,13,2) pixel, vis (T,13), fps, world (T,13,3) mét (MediaPipe pose_world_landmarks).
  Thứ tự 13 khớp giống extract_pose_mediapipe.py. world là bắt buộc: góc khuỷu tay và độ xoay
  vai/hông khi quay chính diện bị méo rất nặng nếu chỉ dùng 2D.

Output của score_punch(): (acc4, expr6, total10, details)
- details["acc_items"], ["expr_items"]: từng hạng mục {code,label,maxi,got,deduct,reason,tip,assessable}
- Hạng mục không đánh giá được (vd video cắt trước khi thu tay) có assessable=False; điểm thành phần
  được quy đổi theo tỉ lệ trên các hạng mục còn lại.
- details["valid"] = False nếu không tìm thấy một cú đấm trọn vẹn -> không chấm.

LƯU Ý: ngưỡng điểm được đặt từ hiểu biết kỹ thuật + phân bố số đo trên clip của 1 võ sinh đai đen,
CHƯA đối chiếu với điểm của trọng tài. Xem PUNCH_SCORING.md.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
PUNCH_TYPES = ("hook", "straight")
GUARD_MODES = ("chin", "waist")


# ========== helpers ==========
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _map01(x: float, lo: float, hi: float) -> float:
    """lo -> 0, hi -> 1 (cho phép lo > hi để đảo chiều: càng nhỏ càng tốt)."""
    if x is None or not np.isfinite(x) or hi == lo:
        return 0.0
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)


def _band01(x: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    """Hình thang: 0 ngoài [lo0,hi0], 1 trong [lo1,hi1]."""
    if x is None or not np.isfinite(x):
        return 0.0
    if x < lo1:
        return _map01(x, lo0, lo1)
    if x > hi1:
        return _map01(x, hi0, hi1)
    return 1.0


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính các frame NaN theo thời gian (mỗi cột độc lập)."""
    y = x.reshape(x.shape[0], -1).astype(np.float64).copy()
    t = np.arange(y.shape[0])
    for j in range(y.shape[1]):
        ok = np.isfinite(y[:, j])
        if ok.sum() == 0:
            y[:, j] = 0.0
        elif (~ok).any():
            y[~ok, j] = np.interp(t[~ok], t[ok], y[ok, j])
    return y.reshape(x.shape)


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Trung bình trượt theo thời gian (khử rung MediaPipe & frame bị nhân đôi)."""
    if x.shape[0] < k:
        return x
    pad = k // 2
    flat = x.reshape(x.shape[0], -1)
    xp = np.concatenate([np.repeat(flat[:1], pad, 0), flat, np.repeat(flat[-1:], pad, 0)], 0)
    ker = np.ones(k) / k
    out = np.stack([np.convolve(xp[:, j], ker, "valid") for j in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def _vec_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    cos = (u * v).sum(-1) / (nu * nv + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _yaw_deg(w: np.ndarray, a: int, b: int) -> np.ndarray:
    """Góc xoay quanh trục đứng của đoạn a-b (vai hoặc hông) trong mặt phẳng x-z."""
    return np.degrees(np.unwrap(np.arctan2(w[:, a, 2] - w[:, b, 2], w[:, a, 0] - w[:, b, 0])))


def _straightness(p: np.ndarray) -> float:
    """1 - sai lệch trung bình khỏi đường thẳng nối điểm đầu-cuối (giống luật đá)."""
    if p.shape[0] < 4:
        return float("nan")
    d = p[-1] - p[0]
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return float("nan")
    u = d / n
    proj = (p - p[0]) @ u
    err = np.linalg.norm(p - (p[0] + proj[:, None] * u[None]), axis=1)
    return _clamp(1.0 - float(np.mean(err)) / n, 0.0, 1.0)


# ========== raw metrics ==========
@dataclass
class PunchMetrics:
    punch_type: str
    side: str                 # 'L' / 'R' tay đấm
    guard_mode: str
    valid: bool
    invalid_reason: str
    T: int
    fps: float
    # phases (frame index)
    t0: int                   # bắt đầu ra đòn
    t_vpeak: int              # tốc độ cổ tay lớn nhất
    t_imp: int                # điểm chạm (đến đích)
    t_rec: Optional[int]      # đã về thế thủ (None nếu không thấy)
    baseline_frames: int      # số frame thủ trước khi ra đòn
    post_frames: int          # số frame sau điểm chạm
    # accuracy
    elbow_flex_deg: float     # góc khuỷu tay (3D) tại điểm chạm, 180 = duỗi thẳng
    upper_arm_elev_deg: float # góc cánh tay trên so với thân (3D), 0 = khép sát sườn, 90 = ngang vai
    fist_height: float        # độ cao nắm tay so với vai (đơn vị chiều dài thân, + là cao hơn)
    reach: float              # khoảng cách vai-cổ tay / chiều dài tay (1 = duỗi hết)
    path_straightness: float  # độ thẳng quỹ đạo cổ tay (t0->t_imp)
    shoulder_rot_deg: Optional[float]  # xoay vai về phía đòn đấm (độ, + là đúng chiều); None nếu không thấy lúc bắt đầu
    hip_rot_deg: Optional[float]       # xoay hông về phía đòn đấm
    guard_start: Optional[float]  # khoảng cách xa nhất từ 2 nắm tay tới vị trí thủ lúc bắt đầu (None nếu không thấy)
    guard_hand_imp: float     # khoảng cách tay không đấm tới vị trí thủ tại điểm chạm
    recovery_ratio: Optional[float]
    trunk_lean_deg: float     # độ nghiêng thân (2D) tại điểm chạm
    hip_sway: float           # độ lắc tâm hông (2D, chiều dài thân)
    # expression
    peak_speed: float         # tốc độ cổ tay lớn nhất (chiều dài thân / giây)
    peak_rot_speed: Optional[float]  # tốc độ xoay vai lớn nhất (độ / giây)
    exec_time: Optional[float]  # thời gian ra đòn t0 -> t_imp (giây), None nếu video bắt đầu giữa động tác
    smoothness: float         # 2*v_peak / tổng biến thiên tốc độ (1 = một nhịp tăng-giảm gọn)
    wrist_visibility: float


def _invalid(punch_type, side, guard_mode, T, fps, reason) -> PunchMetrics:
    nan = float("nan")
    return PunchMetrics(
        punch_type=punch_type, side=side, guard_mode=guard_mode, valid=False, invalid_reason=reason,
        T=T, fps=fps, t0=0, t_vpeak=0, t_imp=0, t_rec=None, baseline_frames=0, post_frames=0,
        elbow_flex_deg=nan, upper_arm_elev_deg=nan, fist_height=nan, reach=nan, path_straightness=nan,
        shoulder_rot_deg=nan, hip_rot_deg=nan, guard_start=None, guard_hand_imp=nan, recovery_ratio=None,
        trunk_lean_deg=nan, hip_sway=nan, peak_speed=nan, peak_rot_speed=nan, exec_time=None,
        smoothness=nan, wrist_visibility=0.0)


def build_punch_metrics(kpts: np.ndarray, vis: np.ndarray, fps: float, world: np.ndarray,
                        punch_type: str = "hook", side: Optional[str] = None,
                        guard_mode: str = "chin", debug: Optional[Dict] = None) -> PunchMetrics:
    """debug: nếu truyền dict vào, hàm ghi thêm các chuỗi tín hiệu theo thời gian (để vẽ/kiểm tra)."""
    if punch_type not in PUNCH_TYPES:
        raise ValueError(f"punch_type phải là {PUNCH_TYPES}, nhận '{punch_type}'")
    if guard_mode not in GUARD_MODES:
        raise ValueError(f"guard_mode phải là {GUARD_MODES}, nhận '{guard_mode}'")
    fps = float(fps) if fps and fps > 0 else 30.0
    T = int(kpts.shape[0])
    if T < 8 or world is None or world.shape[0] != T:
        return _invalid(punch_type, side or "?", guard_mode, T, fps, "Video quá ngắn hoặc thiếu tọa độ 3D")

    w = _smooth(_interp_nan(world), 5)
    k2 = _smooth(_interp_nan(kpts), 5)

    sh_mid = 0.5 * (w[:, LS] + w[:, RS])
    hip_mid = 0.5 * (w[:, LH] + w[:, RH])
    Lt = float(np.median(np.linalg.norm(sh_mid - hip_mid, axis=1)))
    if not np.isfinite(Lt) or Lt < 1e-3:
        return _invalid(punch_type, side or "?", guard_mode, T, fps, "Không ước lượng được kích thước cơ thể")

    elevL = _vec_angle(w[:, LE] - w[:, LS], w[:, LH] - w[:, LS])
    elevR = _vec_angle(w[:, RE] - w[:, RS], w[:, RH] - w[:, RS])
    armL = np.linalg.norm(w[:, LE] - w[:, LS], axis=1) + np.linalg.norm(w[:, LW] - w[:, LE], axis=1)
    armR = np.linalg.norm(w[:, RE] - w[:, RS], axis=1) + np.linalg.norm(w[:, RW] - w[:, RE], axis=1)
    reachL = np.linalg.norm(w[:, LW] - w[:, LS], axis=1) / np.median(armL)
    reachR = np.linalg.norm(w[:, RW] - w[:, RS], axis=1) / np.median(armR)

    # ---- tay đấm ----
    if side not in ("L", "R"):
        if punch_type == "hook":
            side = "R" if np.ptp(elevR) >= np.ptp(elevL) else "L"
        else:
            side = "R" if np.ptp(reachR) >= np.ptp(reachL) else "L"
    if side == "R":
        S, E, Wr, OW = RS, RE, RW, LW
        elev, reach = elevR, reachR
    else:
        S, E, Wr, OW = LS, LE, LW, RW
        elev, reach = elevL, reachL
    rot_sign = 1.0 if side == "R" else -1.0

    # ---- điểm chạm: hook = khuỷu nâng lên, straight = tay vươn ra ----
    # Chọn cú đấm có mức tăng lớn nhất so với điểm thấp nhất trong 1s trước đó (video có thể bắt đầu
    # khi tay còn ở điểm chạm của lần đấm trước), rồi lấy frame đầu tiên đến đích.
    sig = elev if punch_type == "hook" else reach
    tol = 8.0 if punch_type == "hook" else 0.03
    win = int(round(1.0 * fps))
    rise_t = np.array([sig[t] - np.min(sig[max(0, t - win):t + 1]) for t in range(T)])
    t_best = int(np.argmax(rise_t))
    t_max = t_best
    while t_max + 1 < T and sig[t_max + 1] >= sig[t_max]:
        t_max += 1
    t_imp = t_max
    while t_imp > 0 and sig[t_imp - 1] >= sig[t_max] - tol:
        t_imp -= 1  # bỏ qua đoạn giữ nguyên tại đích

    # ---- tốc độ cổ tay (3D, gốc ở hông -> gồm cả phần xoay thân) ----
    sp = np.linalg.norm(np.diff(w[:, Wr], axis=0), axis=1) * fps / Lt
    sp = np.r_[sp[:1], sp]  # frame 0 lấy theo frame 1 (không giả định đứng yên)
    lo = max(0, t_imp - int(round(1.0 * fps)))
    t_vpeak = lo + int(np.argmax(sp[lo:t_imp + 1]))
    v_peak = float(sp[t_vpeak])

    if debug is not None:
        debug.update(side=side, sig=sig.copy(), speed=sp.copy(),
                     sig_name="Độ nâng khuỷu (°)" if punch_type == "hook" else "Độ vươn tay (× dài tay)",
                     shoulder_yaw=rot_sign * _yaw_deg(w, LS, RS))

    # ---- bắt đầu ra đòn: lùi từ đỉnh tốc độ tới khi cổ tay gần như đứng yên ----
    t0, onset_seen = 0, False
    for t in range(t_vpeak, -1, -1):
        if sp[t] < 0.2 * v_peak:
            t0, onset_seen = t, True
            break

    need = 30.0 if punch_type == "hook" else 0.25
    if t_imp <= t0 + 1 or rise_t[t_best] < need:
        return _invalid(punch_type, side, guard_mode, T, fps,
                        "Không thấy một cú đấm trọn vẹn: video bắt đầu/kết thúc giữa động tác, không có cú đấm, "
                        "hoặc góc quay (nghiêng hẳn/quay lưng) khiến không đo được tay. Nên quay chính diện hoặc chéo ~45°.")

    b0 = max(0, t0 - int(round(0.3 * fps)))
    base = slice(b0, t0 + 1)
    # chỉ tin tư thế thủ ban đầu khi thấy ít nhất 3 frame đứng yên trước lúc ra đòn
    baseline_frames = (t0 - b0 + 1) if onset_seen else 0

    # ---- hình dạng tay tại điểm chạm ----
    elbow_flex = float(_vec_angle(w[t_imp, S] - w[t_imp, E], w[t_imp, Wr] - w[t_imp, E]))
    upper_elev = float(elev[t_imp])
    fist_h = float((w[t_imp, S, 1] - w[t_imp, Wr, 1]) / Lt)  # world y hướng xuống
    straight = _straightness(w[t0:t_imp + 1, Wr])

    # ---- xoay vai / hông: biên độ xoay đúng chiều trong 1s trước điểm chạm (tính cả pha lấy đà) ----
    yaw_s = rot_sign * _yaw_deg(w, LS, RS)
    yaw_h = rot_sign * _yaw_deg(w, LH, RH)
    rw = slice(max(0, t_imp - win), t_imp + 1)
    if onset_seen:
        rot_s = float(yaw_s[t_imp] - np.min(yaw_s[rw]))
        rot_h = float(yaw_h[t_imp] - np.min(yaw_h[rw]))
        rot_speed = float(np.max(np.abs(np.diff(yaw_s[max(0, t0 - 1):t_imp + 1])))) * fps
    else:  # phần xoay có thể đã diễn ra trước khi video bắt đầu -> không đánh giá
        rot_s = rot_h = rot_speed = None

    # ---- tay thủ ----
    # vị trí thủ của mỗi cổ tay: chin = cằm (mũi hạ xuống 0.08 thân), waist = hông cùng bên
    if guard_mode == "chin":
        chin = w[:, HEAD] + np.array([0.0, 0.08 * Lt, 0.0])
        target = {LW: chin, RW: chin}
    else:
        target = {LW: w[:, LH], RW: w[:, RH]}
    guard_start = float(max(np.linalg.norm(np.median(w[base, wj] - target[wj][base], 0)) / Lt
                            for wj in (LW, RW))) if baseline_frames >= 3 else None
    g_imp = float(np.linalg.norm(w[t_imp, OW] - target[OW][t_imp]) / Lt)

    # ---- thu tay về: tay trở lại cấu hình thủ (hook: khuỷu hạ xuống; straight: tay co về) ----
    # Dùng chính tín hiệu tìm điểm chạm. Không dùng khoảng cách nắm tay-cằm vì ở đòn móc nắm tay tại
    # điểm chạm nằm ngay cạnh đầu, gần cằm không kém lúc thủ.
    pre = slice(max(0, t_imp - win), t_imp + 1)
    sig_guard = float(np.median(sig[base])) if baseline_frames >= 3 else float(np.min(sig[pre]))
    amp = float(sig[t_imp] - sig_guard)
    t_end = min(T - 1, t_imp + int(round(1.2 * fps)))
    post_sig = sig[t_imp:t_end + 1]
    rec = _clamp(float(sig[t_imp] - np.min(post_sig)) / (amp + 1e-9), 0.0, 1.0)
    t_rec = None
    for i, s in enumerate(post_sig):
        if s <= sig_guard + 0.3 * amp:
            t_rec = t_imp + i
            break
    post_frames = T - 1 - t_imp
    # video kết thúc quá sớm (<0.4s sau điểm chạm) mà chưa thấy thu tay -> không đánh giá
    recovery = rec if (t_rec is not None or post_frames >= int(round(0.4 * fps))) else None

    # ---- thăng bằng (2D) ----
    sh2 = 0.5 * (k2[:, LS] + k2[:, RS])
    hp2 = 0.5 * (k2[:, LH] + k2[:, RH])
    L2 = float(np.median(np.linalg.norm(sh2 - hp2, axis=1))) + 1e-6
    tv = sh2[t_imp] - hp2[t_imp]
    lean = float(np.degrees(np.arctan2(abs(tv[0]), -tv[1])))
    sway = float(np.max(np.std(hp2[b0:t_end + 1] / L2, axis=0)))

    # ---- nhịp ra đòn ----
    spd = sp[t0:t_imp + 1]
    tv_sp = float(np.sum(np.abs(np.diff(spd)))) + abs(float(spd[0])) + abs(float(spd[-1]))
    smooth = _clamp(2.0 * v_peak / (tv_sp + 1e-9), 0.0, 1.0)

    vis_w = float(np.mean(vis[:, Wr] > 0.5)) if vis is not None and vis.shape[0] == T else 1.0

    return PunchMetrics(
        punch_type=punch_type, side=side, guard_mode=guard_mode, valid=True, invalid_reason="",
        T=T, fps=fps, t0=int(t0), t_vpeak=int(t_vpeak), t_imp=int(t_imp), t_rec=t_rec,
        baseline_frames=int(baseline_frames), post_frames=int(post_frames),
        elbow_flex_deg=elbow_flex, upper_arm_elev_deg=upper_elev, fist_height=fist_h,
        reach=float(reach[t_imp]), path_straightness=float(straight),
        shoulder_rot_deg=rot_s, hip_rot_deg=rot_h,
        guard_start=guard_start, guard_hand_imp=g_imp, recovery_ratio=recovery,
        trunk_lean_deg=lean, hip_sway=sway,
        peak_speed=v_peak, peak_rot_speed=rot_speed,
        exec_time=float((t_imp - t0) / fps) if onset_seen else None,
        smoothness=float(smooth), wrist_visibility=vis_w,
    )


# ========== ngưỡng chấm (sửa ở đây để hiệu chỉnh) ==========
@dataclass
class PunchRanges:
    # hook — hình dạng tay (hình thang lo0, lo1, hi1, hi0)
    hook_flex: Tuple[float, float, float, float] = (55.0, 80.0, 130.0, 155.0)
    hook_elev: Tuple[float, float, float, float] = (45.0, 75.0, 135.0, 160.0)
    hook_fist_h: Tuple[float, float, float, float] = (-0.15, 0.15, 0.65, 0.90)
    # straight
    straight_flex: Tuple[float, float] = (120.0, 160.0)
    straight_path: Tuple[float, float] = (0.70, 0.93)
    straight_fist_h: Tuple[float, float, float, float] = (-0.60, -0.40, 0.30, 0.50)
    # xoay thân (độ, đúng chiều)
    # xoay quá nhiều (>90°) = vung cả người / mất kiểm soát -> giảm điểm
    hook_rot_sh: Tuple[float, float, float, float] = (10.0, 40.0, 90.0, 130.0)
    hook_rot_hip: Tuple[float, float, float, float] = (5.0, 30.0, 90.0, 130.0)
    straight_rot_sh: Tuple[float, float, float, float] = (5.0, 30.0, 75.0, 110.0)
    # thu tay
    recovery: Tuple[float, float] = (0.30, 0.80)
    # tay thủ (khoảng cách / chiều dài thân, càng nhỏ càng tốt)
    guard_start: Tuple[float, float] = (0.90, 0.55)
    guard_imp: Tuple[float, float] = (0.95, 0.55)
    lean: Tuple[float, float] = (35.0, 15.0)
    sway: Tuple[float, float] = (0.20, 0.08)
    # biểu hiện
    speed: Tuple[float, float] = (3.5, 8.0)        # chiều dài thân / giây
    rot_speed: Tuple[float, float] = (80.0, 300.0) # độ / giây
    exec_time: Tuple[float, float] = (0.90, 0.35)  # giây, càng nhanh càng tốt
    smooth: Tuple[float, float] = (0.35, 0.80)


# ========== items ==========
def _add_item(items: List[Dict], category: str, code: str, label: str, maxi: float,
              got: Optional[float], reason: str, tip: str, key: bool = False) -> None:
    assess = got is not None
    g = float(_clamp(got, 0.0, maxi)) if assess else None
    d = float(maxi - g) if assess else 0.0
    items.append({
        "category": category, "code": code, "label": label, "maxi": float(maxi),
        "got": g, "deduct": d, "reason": reason, "tip": tip, "key": bool(key),
        "assessable": assess,
        "severity": "n/a" if not assess else ("high" if d > 0.6 else ("mid" if d > 0.3 else "low")),
    })


def _component(items: List[Dict], full: float) -> float:
    maxi = sum(it["maxi"] for it in items if it["assessable"])
    got = sum(it["got"] for it in items if it["assessable"])
    return _clamp(full * got / maxi, 0.0, full) if maxi > 0 else 0.0


def score_punch(m: PunchMetrics, ranges: Optional[PunchRanges] = None):
    """Trả về (acc4, expr6, total10, details). Nếu m.valid=False -> điểm None."""
    if not m.valid:
        return None, None, None, {"valid": False, "reason": m.invalid_reason, "raw": asdict(m)}
    r = ranges or PunchRanges()
    acc: List[Dict] = []
    expr: List[Dict] = []
    hand = "phải" if m.side == "R" else "trái"

    # ---------- Chính xác (4) ----------
    if m.punch_type == "hook":
        f01 = _band01(m.elbow_flex_deg, *r.hook_flex)
        e01 = _band01(m.upper_arm_elev_deg, *r.hook_elev)
        h01 = _band01(m.fist_height, *r.hook_fist_h)
        _add_item(acc, "accuracy", "HK_A1", "Hình dạng tay khi chạm (khuỷu gập, khuỷu ngang vai)", 1.6,
                  (0.40 * f01 + 0.35 * e01 + 0.25 * h01) * 1.6,
                  f"góc khuỷu≈{m.elbow_flex_deg:.0f}°, nâng cánh tay≈{m.upper_arm_elev_deg:.0f}°, "
                  f"nắm tay cao hơn vai≈{m.fist_height:.2f}",
                  "Khuỷu tay gập khoảng 90°, nâng khuỷu ngang vai, nắm tay đến tầm đầu đối phương; "
                  "không duỗi thẳng tay, không để khuỷu thấp.", key=True)
        rot_ok = m.shoulder_rot_deg is not None
        a2 = (0.60 * _band01(m.shoulder_rot_deg, *r.hook_rot_sh)
              + 0.40 * _band01(m.hip_rot_deg, *r.hook_rot_hip)) * 1.2 if rot_ok else None
        _add_item(acc, "accuracy", "HK_A2", "Xoay hông – vai theo đòn", 1.2, a2,
                  f"xoay vai≈{m.shoulder_rot_deg:.0f}°, xoay hông≈{m.hip_rot_deg:.0f}°" if rot_ok
                  else "video bắt đầu giữa động tác — không đánh giá",
                  "Đòn móc lấy lực từ xoay hông và vai; xoay gót chân sau, đừng chỉ vung tay.", key=True)
    else:
        f01 = _map01(m.elbow_flex_deg, *r.straight_flex)
        l01 = _map01(m.path_straightness, *r.straight_path)
        _add_item(acc, "accuracy", "ST_A1", "Duỗi tay & đường đấm thẳng", 1.6,
                  (0.60 * f01 + 0.40 * l01) * 1.6,
                  f"góc khuỷu≈{m.elbow_flex_deg:.0f}°, độ thẳng quỹ đạo≈{m.path_straightness:.2f}",
                  "Đấm theo đường thẳng ngắn nhất tới mục tiêu, duỗi tay gần hết khi chạm (không khóa cứng khớp).",
                  key=True)
        h01 = _band01(m.fist_height, *r.straight_fist_h)
        if m.shoulder_rot_deg is not None:
            a2 = (0.50 * _band01(m.shoulder_rot_deg, *r.straight_rot_sh) + 0.50 * h01) * 1.2
            rot_txt = f"xoay vai≈{m.shoulder_rot_deg:.0f}°"
        else:
            a2, rot_txt = h01 * 1.2, "xoay vai: không đánh giá"
        _add_item(acc, "accuracy", "ST_A2", "Xoay vai & tầm đấm", 1.2, a2,
                  f"{rot_txt}, nắm tay so với vai≈{m.fist_height:.2f}",
                  "Đẩy vai theo đòn; nắm tay ở tầm ngực (momtong) hoặc mặt, không quá thấp/cao.", key=True)

    rec_got = None if m.recovery_ratio is None else _map01(m.recovery_ratio, *r.recovery) * 0.8
    rec_reason = ("video kết thúc trước khi thu tay — không đánh giá" if m.recovery_ratio is None
                  else f"tay {hand} trở về thế thủ≈{m.recovery_ratio * 100:.0f}%")
    _add_item(acc, "accuracy", "PU_A3", "Thu tay về thế thủ", 0.8, rec_got, rec_reason,
              "Đấm xong thu tay về ngay theo đường cũ, không để tay treo ở điểm chạm.", key=True)

    # A4: tay thủ lúc chạm 50%, thủ ban đầu 25% (bỏ nếu không quan sát được), thăng bằng 25%
    parts = [(0.50, _map01(m.guard_hand_imp, *r.guard_imp)),
             (0.25, 0.5 * _map01(m.trunk_lean_deg, *r.lean) + 0.5 * _map01(m.hip_sway, *r.sway))]
    if m.guard_start is not None:
        parts.append((0.25, _map01(m.guard_start, *r.guard_start)))
    a4 = sum(wt * v for wt, v in parts) / sum(wt for wt, _ in parts)
    where = "cằm" if m.guard_mode == "chin" else "hông"
    gs_txt = "không thấy" if m.guard_start is None else f"{m.guard_start:.2f}"
    _add_item(acc, "accuracy", "PU_A4", "Tay thủ & thăng bằng", 0.4, a4 * 0.4,
              f"tay còn lại cách vị trí thủ≈{m.guard_hand_imp:.2f}, thủ ban đầu≈{gs_txt}, "
              f"nghiêng thân≈{m.trunk_lean_deg:.0f}°",
              f"Tay không đấm giữ ở {where} suốt đòn; thân thẳng, không đổ người theo cú đấm.")

    # ---------- Biểu hiện (6) ----------
    _add_item(expr, "expression", "PX_E1", "Tốc độ", 2.0, _map01(m.peak_speed, *r.speed) * 2.0,
              f"tốc độ cổ tay≈{m.peak_speed:.1f} thân/giây",
              "Thả lỏng vai rồi bộc phát; tay đi nhanh nhất ngay trước điểm chạm.", key=True)
    _add_item(expr, "expression", "PX_E2", "Lực (xoay thân truyền lực)", 1.6,
              None if m.peak_rot_speed is None else _map01(m.peak_rot_speed, *r.rot_speed) * 1.6,
              "video bắt đầu giữa động tác — không đánh giá" if m.peak_rot_speed is None
              else f"tốc độ xoay vai≈{m.peak_rot_speed:.0f}°/s",
              "Lực đến từ chân – hông – vai; xoay thân dứt khoát cùng lúc ra đòn.", key=True)
    _add_item(expr, "expression", "PX_E3", "Dứt khoát (thời gian ra đòn)", 1.4,
              None if m.exec_time is None else _map01(m.exec_time, *r.exec_time) * 1.4,
              "video bắt đầu giữa động tác — không đánh giá" if m.exec_time is None
              else f"từ lúc ra đòn tới điểm chạm≈{m.exec_time:.2f}s",
              "Ra đòn gọn, không lấy đà quá dài làm lộ ý đồ.")
    _add_item(expr, "expression", "PX_E4", "Độ mượt", 1.0, _map01(m.smoothness, *r.smooth) * 1.0,
              f"độ mượt≈{m.smoothness:.2f}",
              "Một nhịp tăng tốc liền mạch, không khựng giữa chừng.")

    acc4 = _component(acc, 4.0)
    expr6 = _component(expr, 6.0)
    total10 = _clamp(acc4 + expr6, 0.0, 10.0)

    scored = [it for it in acc + expr if it["assessable"]]
    top = sorted(scored, key=lambda d: d["deduct"], reverse=True)
    top = [d for d in top if d["deduct"] > 1e-6][:6]
    warnings = []
    if m.baseline_frames < 3:
        warnings.append("Video bắt đầu gần như ngay khi ra đòn — tư thế thủ ban đầu đo không chắc chắn.")
    if m.exec_time is None:
        warnings.append("Video bắt đầu giữa động tác — các mục xoay thân/thời gian ra đòn không tính.")
    if m.shoulder_rot_deg is not None and m.shoulder_rot_deg > 100:
        warnings.append("Thân xoay rất nhiều (có thể quay lưng lại camera) — MediaPipe dễ nhầm trái/phải, số đo kém tin cậy.")
    if m.recovery_ratio is None:
        warnings.append("Video kết thúc trước khi thu tay — hạng mục thu tay không tính, điểm Chính xác quy đổi theo tỉ lệ.")
    if m.wrist_visibility < 0.6:
        warnings.append("Cổ tay đấm bị che khuất nhiều — số đo kém tin cậy.")
    details = {
        "valid": True,
        "acc_items": acc,
        "expr_items": expr,
        "top_issues": top,
        "tips": [d["tip"] for d in top][:5],
        "warnings": warnings,
        "raw": asdict(m),
    }
    return acc4, expr6, total10, details
