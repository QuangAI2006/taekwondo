from __future__ import annotations

"""
Beginner-friendly Taekwondo kick scoring (Poomsae-style, 10 points total).

Design goals:
- Total = Accuracy 4 + Expression 6 (matches your Poomsae rubric).
- Key technical points carry higher weights; secondary points lower.
- Detailed, UI-ready deduction items with clear "reason" and "tip".
- Beginner-friendly calibration so typical novice performances land >= 6.0,
  without giving high scores to clearly invalid motions.

Input:
- RawMetrics (precomputed) OR (kpts, vis, fps) OR dict {"kpts","vis","fps"}.
  kpts: (T,13,2) float32 pixel coords
  vis : (T,13) float32 visibility in [0,1]

Output:
- acc4, expr6, total10, details(dict)
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Dict, List, Tuple, Any

import numpy as np
import math


# ========== 输入维度兼容（评分规则只用二维 x,y） ==========
def _ensure_xy2(kpts: np.ndarray) -> np.ndarray:
    """
    评分规则体系仅定义在图像平面 (x,y) 上，期望 kpts 为 (T,13,2)。
    若输入为 (T,13,3) 或更多维度（含 z），自动丢弃 z，仅保留前两维。
    """
    k = np.asarray(kpts)
    if k.ndim != 3:
        raise ValueError(f"kpts must be 3D (T,K,C), got shape={k.shape}")
    if k.shape[-1] == 2:
        return k.astype(np.float32, copy=False)
    if k.shape[-1] >= 3:
        return k[..., :2].astype(np.float32, copy=False)
    raise ValueError(f"kpts last dim must be >=2, got shape={k.shape}")


# ========== 关键点索引（与 extract_pose_mediapipe 的 13 点一致） ==========
HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12


# ========== 枚举 ==========
class KickType(Enum):
    front = "front"  # 前踢
    roundhouse = "roundhouse"  # 横踢
    side = "side"  # 侧踢
    axe = "axe"  # 下劈


def _parse_ktype(x: Union[str, KickType]) -> KickType:
    if isinstance(x, KickType):
        return x
    s = str(x).strip().lower()
    for k in KickType:
        if s == k.value:
            return k
    # allow Chinese mapping
    zh_map = {"前踢": "front", "横踢": "roundhouse", "侧踢": "side", "下劈": "axe"}
    if s in zh_map:
        s = zh_map[s]
        for k in KickType:
            if s == k.value:
                return k
    raise ValueError(f"Unknown kick type: {x}")


# ========== Raw metrics ==========
@dataclass
class RawMetrics:
    # phases
    t0: int
    t_peak: int
    t1: int

    # validity
    T: int
    valid_ratio: float
    has_enough: bool
    kick_side: str  # 'L' or 'R'

    # orientation axes
    lr_axis_quality: float

    # joint / motion
    straightness: float
    smoothness: float
    stability: float
    recovery_ratio: float
    recovery_knee_bend_deg: float

    # landing & forward/back
    landing_forward: Optional[float]  # +in-front, -behind
    landing_hard: float

    # hands
    hand_guard_drop: float

    # speed / power / height
    speed_raw: float
    power_raw: float
    height_raw: float
    down_speed: float  # 下劈下行速度

    # —— 准确度相关（通用） ——
    support_knee_angle_med: float
    kick_height_over_hip: float
    kick_knee_angle_peak: float
    roundhouse_align_dev_deg: Optional[float]

    knee_lift_vertical: float
    knee_lift_lateral: float


# ========== Scoring config ==========
@dataclass
class ScoreConfig:
    level: str = "beginner"  # beginner / intermediate / advanced
    vth: float = 0.2  # visibility threshold
    min_valid_ratio: float = 0.35
    # weights
    w_straight: float = 1.0
    w_smooth: float = 1.0
    w_power: float = 1.0
    w_height: float = 1.0
    w_stability: float = 1.0
    w_guard: float = 1.0


@dataclass
class ScoreRanges:
    # accuracy (4)
    acc_straight: Tuple[float, float] = (0.70, 0.95)
    acc_smooth: Tuple[float, float] = (0.45, 0.85)
    acc_stable: Tuple[float, float] = (0.35, 0.80)
    acc_recover: Tuple[float, float] = (0.25, 0.75)

    # expression (6)
    expr_speed: Tuple[float, float] = (0.20, 0.75)
    expr_power: Tuple[float, float] = (0.18, 0.70)
    expr_height: Tuple[float, float] = (0.15, 0.65)
    expr_downspeed: Tuple[float, float] = (0.20, 0.90)

    # technique cues
    knee_lift_v: Tuple[float, float] = (0.03, 0.11)
    knee_lift_lat: Tuple[float, float] = (0.03, 0.10)

    @staticmethod
    def default(level: str = "beginner") -> "ScoreRanges":
        # can adjust levels later
        return ScoreRanges()


# ========== small helpers ==========
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _map01(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if hi <= lo:
        return 0.0
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)


def _unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps or not np.isfinite(n):
        return v * 0.0
    return v / n


def _ema(arr: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # angle ABC
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = _clamp(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def _nanmedian(x: np.ndarray, axis=None) -> float:
    with np.errstate(all="ignore"):
        v = np.nanmedian(x, axis=axis)
    if np.isscalar(v):
        return float(v)
    return float(np.nanmedian(v))


def _safe_median(arr: np.ndarray) -> np.ndarray:
    """
    Safely calculate median along axis 0.
    If all values in a slice are NaN (e.g. occlusion), return 0.0 instead of warning.
    """
    if arr.size == 0:
        return np.zeros((arr.shape[1] if arr.ndim > 1 else 1,), dtype=np.float32)

    # 检查是否全部是 NaN
    if np.all(np.isnan(arr)):
        return np.zeros(arr.shape[1], dtype=np.float32)

    with np.errstate(all="ignore", invalid="ignore"):
        m = np.nanmedian(arr, axis=0)

    # 如果只有部分列是 NaN，nanmedian 结果里可能还有 NaN，补 0
    m[np.isnan(m)] = 0.0
    return m


def _nanmean(x: np.ndarray, axis=None) -> float:
    with np.errstate(all="ignore"):
        v = np.nanmean(x, axis=axis)
    if np.isscalar(v):
        return float(v)
    return float(np.nanmean(v))


def _nanstd(x: np.ndarray, axis=None) -> float:
    with np.errstate(all="ignore"):
        v = np.nanstd(x, axis=axis)
    if np.isscalar(v):
        return float(v)
    return float(np.nanstd(v))


# ========== kick side / indices ==========
def _kick_side_indices(xy: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    Determine kicking side (left or right) by max ankle distance from hip center.
    Returns indices: A, K, H, AS, KS, HS
    """
    hip_ctr = 0.5 * (xy[:, LH] + xy[:, RH])
    distL = np.linalg.norm(xy[:, LA] - hip_ctr, axis=1)
    distR = np.linalg.norm(xy[:, RA] - hip_ctr, axis=1)
    if float(np.nanmax(distL)) >= float(np.nanmax(distR)):
        # kick left
        return LA, LK, LH, RA, RK, RH
    else:
        return RA, RK, RH, LA, LK, LH


# ========== phase split ==========
def _split_phases(xy: np.ndarray, A: int, hip_ctr: np.ndarray) -> Tuple[int, int, int]:
    """
    t0: start of kick (ankle begins moving away from hip),
    t_peak: max extension,
    t1: end (recovery / after landing)
    """
    T = xy.shape[0]
    distA = np.linalg.norm(xy[:, A] - hip_ctr, axis=1)

    # Safe argmax
    if np.any(np.isfinite(distA)):
        t_peak = int(np.nanargmax(distA))
    else:
        t_peak = 0

    base_n = max(1, int(0.1 * T))

    # Fix for RuntimeWarning: All-NaN slice encountered in median
    slice_base = distA[:base_n]
    if np.all(np.isnan(slice_base)):
        base = float(distA[0]) if T > 0 and np.isfinite(distA[0]) else 0.0
    else:
        base = np.nanmedian(slice_base)

    if not np.isfinite(base):
        base = float(distA[0]) if T > 0 else 0.0

    max_dist = float(np.nanmax(distA)) if np.any(np.isfinite(distA)) else 0.0
    thr = base + 0.12 * (max_dist - base + 1e-6)

    t0 = 0
    for i in range(T):
        if np.isfinite(distA[i]) and float(distA[i]) > thr:
            t0 = i
            break

    # end: when distance falls back near base
    t1 = T - 1
    thr2 = base + 0.10 * (max_dist - base + 1e-6)
    for i in range(t_peak, T):
        if np.isfinite(distA[i]) and float(distA[i]) < thr2:
            t1 = i
            break
    return int(t0), int(t_peak), int(t1)


# ========== metrics builder ==========
def build_metrics_from_sequence(
        kpts: np.ndarray,
        vis: np.ndarray,
        fps: float,
        ktype_hint: Optional[Union[str, KickType]] = None
) -> RawMetrics:
    """
    关键设计：
    - 以肩线(优先)或髋线估计“左右轴” e_lr(t)，EMA 平滑；
    - 前后轴 e_fwd(t) = rotate90(e_lr)，其正负由“起始→峰值的踝位移投影”自动判定；
    - 用前/后轴做落脚“在前/在后”的投影差（侧视更稳）；
    - 起始/峰值/回收阶段分割更严格；手位基准来自起始阶段。
    """
    T = kpts.shape[0]
    xy = _ensure_xy2(kpts)
    v = vis.astype(np.float32)

    # mask invalid
    xy = xy.copy()
    xy[v < 0.2] = np.nan

    # hip center
    hip_ctr = 0.5 * (xy[:, LH] + xy[:, RH])

    # choose kick side
    A, K, H, AS, KS, HS = _kick_side_indices(xy)

    # phases
    t0, t_peak, t1 = _split_phases(xy, A, hip_ctr)

    # estimate LR axis (shoulder preferred)
    lr_raw = xy[:, LS] - xy[:, RS]
    use_sh = np.all(np.isfinite(lr_raw), axis=1)
    lr2_raw = xy[:, LH] - xy[:, RH]
    use_hip = np.all(np.isfinite(lr2_raw), axis=1)

    e_lr_raw = np.zeros((T, 2), np.float32)
    e_lr_raw[:] = np.nan
    e_lr_raw[use_sh] = lr_raw[use_sh]
    e_lr_raw[~use_sh & use_hip] = lr2_raw[~use_sh & use_hip]

    qual = float(np.mean(np.isfinite(e_lr_raw[:, 0]))) if T > 0 else 0.0
    for i in range(T):
        if np.any(np.isnan(e_lr_raw[i])):
            # fallback
            e_lr_raw[i] = e_lr_raw[i - 1] if i > 0 else np.array([1.0, 0.0], np.float32)

    e_lr_raw = np.stack([_unit(e_lr_raw[i]) for i in range(T)], axis=0)
    e_lr = _ema(e_lr_raw, alpha=0.25)

    # forward axis = rotate lr by +90deg
    e_fwd_raw = np.stack([np.array([-e_lr[i, 1], e_lr[i, 0]], np.float32) for i in range(T)], axis=0)

    base_n = max(1, int(0.1 * T))

    # Safe median for start position
    a_start = _safe_median(xy[:base_n, A])
    a_peak = xy[t_peak, A]

    # Handle NaN in a_peak if the specific frame is invalid
    if np.any(np.isnan(a_peak)):
        a_peak = a_start  # Fallback to avoid crash

    disp = a_peak - a_start

    e_fwd_mean = _unit(np.nanmean(e_fwd_raw, axis=0))
    if not np.all(np.isfinite(e_fwd_mean)):
        e_fwd_mean = np.array([0.0, 1.0], np.float32)
    sign = 1.0 if float(np.dot(disp, e_fwd_mean)) >= 0.0 else -1.0
    e_fwd = _ema(e_fwd_raw * sign, alpha=0.25)

    # straightness: how line-like ankle trajectory is around kick
    seg0 = max(0, t0)
    seg1 = min(T, max(t_peak + 1, t0 + 5))
    ankle = xy[seg0:seg1, A]
    if ankle.shape[0] >= 5 and np.all(np.isfinite(ankle), axis=1).sum() >= 5:
        pts = ankle[np.all(np.isfinite(ankle), axis=1)]
        p0 = pts[0]
        p1 = pts[-1]
        dist_p0_p1 = float(np.linalg.norm(p1 - p0))
        if dist_p0_p1 > 1e-6:
            dirv = _unit(p1 - p0)
            proj = (pts - p0) @ dirv
            recon = p0 + proj[:, None] * dirv[None, :]
            err = np.linalg.norm(pts - recon, axis=1)
            straight = 1.0 - float(np.nanmean(err) / dist_p0_p1)
            straight = _clamp(straight, 0.0, 1.0)
        else:
            straight = 0.5
    else:
        straight = 0.5

    # smoothness: inverse of jerkiness (acceleration variance) around kick
    ankle_full = xy[:, A]
    ok = np.all(np.isfinite(ankle_full), axis=1)
    if ok.sum() >= 8:
        pts = ankle_full[ok]
        dt = 1.0 / (fps + 1e-6)
        vels = np.diff(pts, axis=0) / dt
        accs = np.diff(vels, axis=0) / dt
        j = float(np.nanstd(accs))
        smooth = float(1.0 / (1.0 + 0.05 * j))
        smooth = _clamp(smooth, 0.0, 1.0)
    else:
        smooth = 0.5

    # stability: trunk sway (hip center movement) after normalization by body scale
    w1 = np.linalg.norm(xy[:, LS] - xy[:, RS], axis=1)
    w2 = np.linalg.norm(xy[:, LH] - xy[:, RH], axis=1)

    # Robust scale calculation to avoid RuntimeWarning: All-NaN slice encountered
    # stack w1 and w2
    w_stack = np.stack([w1, w2], axis=1)
    # Check which rows have at least one valid value
    has_data = np.any(np.isfinite(w_stack), axis=1)

    scale_arr = np.full(T, np.nan, dtype=np.float32)
    if np.any(has_data):
        scale_arr[has_data] = np.nanmax(w_stack[has_data], axis=1)

    if np.all(np.isnan(scale_arr)):
        scale = 100.0
    else:
        scale = float(np.nanmedian(scale_arr))

    if not np.isfinite(scale) or scale < 1e-3:
        scale = 100.0

    hip_ok = np.all(np.isfinite(hip_ctr), axis=1)
    if hip_ok.sum() >= 6:
        hc = hip_ctr[hip_ok]
        sway = float(np.nanstd(hc / scale))
        stable = float(1.0 / (1.0 + 8.0 * sway))
        stable = _clamp(stable, 0.0, 1.0)
    else:
        stable = 0.5

    # recovery: how close ankle returns to start
    # Fix: use safe median
    a0 = _safe_median(xy[:base_n, A])
    a_end_pts = _safe_median(xy[-base_n:, A])

    # Check if we have valid baselines
    if np.any(np.isnan(a0)) or np.any(np.isnan(a_end_pts)) or np.any(np.isnan(a_peak)):
        rec_ratio = 0.5
    else:
        dist0 = float(np.linalg.norm(a_peak - a0) + 1e-6)
        dist_back = float(np.linalg.norm(a_end_pts - a0))
        rec_ratio = _clamp(1.0 - dist_back / dist0, 0.0, 1.0)

    # knee bend at recovery
    def knee_angle_series(hip, knee, ank) -> np.ndarray:
        ang = np.zeros((T,), np.float32)
        ang[:] = np.nan
        for t in range(T):
            if np.all(np.isfinite(xy[t, hip])) and np.all(np.isfinite(xy[t, knee])) and np.all(np.isfinite(xy[t, ank])):
                ang[t] = _angle_deg(xy[t, hip], xy[t, knee], xy[t, ank])
        return ang

    if A == LA:
        ang = knee_angle_series(LH, LK, LA)
    else:
        ang = knee_angle_series(RH, RK, RA)

    # during recovery segment
    seg_rec = ang[t_peak:t1] if t1 > t_peak else ang[t_peak:]
    if np.any(np.isfinite(seg_rec)):
        min_ang = float(np.nanmin(seg_rec))
        bend_deg = _clamp(180.0 - min_ang, 0.0, 120.0)
    else:
        bend_deg = 0.0

    # landing forward/back: compare ankle projection vs hip along e_fwd at end
    if t1 <= 0:
        landing_fwd = None
        landing_hard = 0.0
    else:
        # use last 10% frames median
        tail_n = max(1, int(0.10 * T))

        # SAFE calculations using _safe_median
        a_end_tail = _safe_median(xy[-tail_n:, A])
        h_end_tail = _safe_median(hip_ctr[-tail_n:])
        ef_tail = _safe_median(e_fwd[-tail_n:])

        if np.any(np.isnan(a_end_tail)) or np.any(np.isnan(h_end_tail)):
            landing_fwd = None
        else:
            # Normalize ef_tail
            ef_tail = _unit(ef_tail)
            landing_fwd = float(np.dot(a_end_tail - h_end_tail, ef_tail) / (scale + 1e-6))

        # landing hardness: ankle speed drop around landing
        if ok.sum() >= 8:
            t_land = min(T - 2, max(t_peak + 1, t1))
            t_land = _clamp(t_land, 2, T - 2)
            t_land = int(t_land)
            # speed before landing vs after
            v_pre = float(np.linalg.norm(ankle_full[t_land] - ankle_full[t_land - 1]))
            v_post = float(np.linalg.norm(ankle_full[t_land + 1] - ankle_full[t_land]))
            landing_hard = _clamp((v_pre - v_post) / (v_pre + 1e-6), 0.0, 1.0)
        else:
            landing_hard = 0.0

    # hand guard drop: wrist height drop relative to shoulder baseline
    # FIX: Use _safe_median to prevent RuntimeWarning: All-NaN slice encountered
    slice_idx = base_n if t0 > 0 else max(1, int(0.1 * T))
    baseL = _safe_median(xy[:slice_idx, LW])
    baseR = _safe_median(xy[:slice_idx, RW])

    sh_mid0 = _safe_median(0.5 * (xy[:slice_idx, LS] + xy[:slice_idx, RS]))

    # Helper for drop calc
    def get_drop(base_wrist, wrist_idx):
        if np.all(np.isfinite(base_wrist)) and np.all(np.isfinite(sh_mid0)):
            # Check if we have valid data in range
            y_vals = xy[t0:t1, wrist_idx, 1]
            if np.any(np.isfinite(y_vals)):
                return float((base_wrist[1] - np.nanmin(y_vals)) / (scale + 1e-6))
        return 0.0

    dropL = get_drop(baseL, LW)
    dropR = get_drop(baseR, RW)
    guard_drop = _clamp(max(dropL, dropR), 0.0, 0.25)

    # speed / power / height (raw)
    if ok.sum() >= 8:
        pts = ankle_full[ok]
        dt = 1.0 / (fps + 1e-6)
        vels = np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt
        speed_raw = float(np.nanmedian(vels) / (scale + 1e-6))
        power_raw = float(np.nanmax(vels) / (scale + 1e-6))
    else:
        speed_raw = 0.0
        power_raw = 0.0

    # height: max ankle above hip (negative y is up? but pixel y down; use hip - ankle)
    if np.any(np.isfinite(xy[:, A, 1])) and np.any(np.isfinite(hip_ctr[:, 1])):
        height_raw = float(np.nanmax((hip_ctr[:, 1] - xy[:, A, 1]) / (scale + 1e-6)))
    else:
        height_raw = 0.0

    # down speed for axe: downward velocity after peak
    if ok.sum() >= 8 and t_peak + 2 < T:
        seg = ankle_full[t_peak:t1] if t1 > t_peak else ankle_full[t_peak:]
        seg_ok = np.all(np.isfinite(seg), axis=1)
        if seg_ok.sum() >= 5:
            pts2 = seg[seg_ok]
            dt = 1.0 / (fps + 1e-6)
            vy = np.diff(pts2[:, 1]) / dt
            down_speed = float(np.nanmax(vy) / (scale + 1e-6))
        else:
            down_speed = 0.0
    else:
        down_speed = 0.0

    # support knee angle median (support leg)
    if A == LA:
        ang_sup = knee_angle_series(RH, RK, RA)
    else:
        ang_sup = knee_angle_series(LH, LK, LA)

    # Fix for RuntimeWarning: All-NaN slice encountered in median
    if np.any(np.isfinite(ang_sup)):
        sup_slice = ang_sup[:base_n]
        if np.all(np.isnan(sup_slice)):
            sup_knee_med = 165.0
        else:
            sup_knee_med = float(np.nanmedian(sup_slice))
    else:
        sup_knee_med = 165.0

    # kick height over hip
    kick_height = float(np.nanmax((hip_ctr[:, 1] - xy[:, A, 1]) / (scale + 1e-6))) if np.any(
        np.isfinite(xy[:, A, 1])) else 0.0

    # kick knee angle peak (max extension -> max angle)
    if np.any(np.isfinite(ang)):
        knee_peak = float(np.nanmax(ang[t0:t1])) if t1 > t0 else float(np.nanmax(ang))
    else:
        knee_peak = 150.0

    # roundhouse alignment deviation: difference between shin direction and lr axis at peak
    if ktype_hint is not None:
        kt = _parse_ktype(ktype_hint)
    else:
        kt = KickType.front

    if kt is KickType.roundhouse:
        # shin vector knee->ankle
        if np.all(np.isfinite(xy[t_peak, K])) and np.all(np.isfinite(xy[t_peak, A])) and np.all(
                np.isfinite(e_lr[t_peak])):
            shin = _unit(xy[t_peak, A] - xy[t_peak, K])
            lr = _unit(e_lr[t_peak])
            # want shin align with +/-lr (horizontal)
            cosv = abs(float(np.dot(shin, lr)))
            dev = float(np.degrees(np.arccos(_clamp(cosv, -1.0, 1.0))))
            rh_align = dev
        else:
            rh_align = None
    else:
        rh_align = None

    # knee lift cues (front/side): from start to t0 segment
    kpos = xy[:, K]
    if np.all(np.isfinite(kpos[:base_n])) and np.any(np.isfinite(kpos[t0:t_peak])):
        k0 = np.nanmedian(kpos[:base_n], axis=0)
        k_peak = xy[t_peak, K]
        knee_lift_vertical = float((k0[1] - k_peak[1]) / (scale + 1e-6))
        knee_lift_lateral = float(abs(k_peak[0] - k0[0]) / (scale + 1e-6))
    else:
        knee_lift_vertical = 0.0
        knee_lift_lateral = 0.0

    # validity ratio
    valid_ratio = float(np.mean(np.isfinite(xy[:, A, 0]))) if T > 0 else 0.0
    has_enough = valid_ratio >= 0.35 and T >= 20

    kick_side = "L" if A == LA else "R"

    return RawMetrics(
        t0=t0, t_peak=t_peak, t1=t1,
        T=T, valid_ratio=valid_ratio, has_enough=has_enough,
        kick_side=kick_side,
        lr_axis_quality=qual,
        straightness=float(straight),
        smoothness=float(smooth),
        stability=float(stable),
        recovery_ratio=float(rec_ratio),
        recovery_knee_bend_deg=float(bend_deg),
        landing_forward=landing_fwd,
        landing_hard=float(landing_hard),
        hand_guard_drop=float(guard_drop),
        speed_raw=float(speed_raw),
        power_raw=float(power_raw),
        height_raw=float(height_raw),
        down_speed=float(down_speed),
        support_knee_angle_med=float(sup_knee_med),
        kick_height_over_hip=float(kick_height),
        kick_knee_angle_peak=float(knee_peak),
        roundhouse_align_dev_deg=rh_align,
        knee_lift_vertical=float(knee_lift_vertical),
        knee_lift_lateral=float(knee_lift_lateral),
    )


# ========== UI item builder ==========
def _add_item(items: List[Dict], category: str, code: str, label: str, maxi: float, got: float,
              reason: str, tip: str, key: bool = False):
    got = float(_clamp(got, 0.0, maxi))
    items.append({
        "category": category,
        "code": code,
        "label": label,
        "maxi": float(maxi),
        "got": got,
        "deduct": float(maxi - got),
        "reason": reason,
        "tip": tip,
        "key": bool(key),
        "severity": "high" if (maxi - got) > 0.6 else ("mid" if (maxi - got) > 0.3 else "low")
    })


def _summarize_top_issues(items: List[Dict], topk: int = 6) -> List[Dict]:
    s = sorted(items, key=lambda d: float(d.get("deduct", 0.0)), reverse=True)
    return [d for d in s if float(d.get("deduct", 0.0)) > 1e-6][:topk]


# ========== main scoring ==========
def score_one_video(
        raw_or_seq: Union[RawMetrics, Tuple[np.ndarray, np.ndarray, float], Dict, Any],
        ktype: Union[str, KickType],
        ranges: Optional[ScoreRanges] = None,
        config: Optional[ScoreConfig] = None
):
    """
    返回: (acc4, expr6, total10, details_dict)
    details_dict: {"acc_items":..., "expr_items":..., "top_issues":..., "tips":..., "raw":...}
    """
    k = _parse_ktype(ktype)
    cfg = config or ScoreConfig()

    # build metrics
    if not isinstance(raw_or_seq, RawMetrics):
        # 兼容性修复：处理 NpzFile (numpy.lib.npyio.NpzFile) 或类似字典的对象
        # 如果 raw_or_seq 具有 'files' 属性 (NpzFile) 或者可以直接作为字典访问
        is_dict_like = isinstance(raw_or_seq, dict) or hasattr(raw_or_seq, 'files') or hasattr(raw_or_seq, 'keys')

        if is_dict_like:
            try:
                # 尝试安全获取
                kpts = raw_or_seq['kpts'] if 'kpts' in raw_or_seq else raw_or_seq['keypoints']
                # vis 可选
                if 'vis' in raw_or_seq:
                    vis = raw_or_seq['vis']
                else:
                    vis = np.ones(kpts.shape[:2], dtype=np.float32)
                # fps 可选
                fps = float(raw_or_seq['fps']) if 'fps' in raw_or_seq else 30.0
            except (KeyError, IndexError, TypeError):
                # 如果 dict 访问失败，回退到 tuple 解包尝试
                kpts, vis, fps = raw_or_seq
        else:
            # 尝试标准解包 (如果是 tuple/list)
            kpts, vis, fps = raw_or_seq

        # 评分规则只支持二维 (x,y)，统一丢弃 z
        kpts = _ensure_xy2(kpts)
        raw = build_metrics_from_sequence(kpts, vis, fps, ktype_hint=k)
    else:
        raw = raw_or_seq

    ranges = ranges or ScoreRanges.default(level=cfg.level)

    acc_items: List[Dict] = []
    expr_items: List[Dict] = []

    # ---------- common normalization ----------
    straight = raw.straightness
    smooth = raw.smoothness
    stable = raw.stability
    rec = raw.recovery_ratio
    guard = raw.hand_guard_drop

    # expression raw
    speed = raw.speed_raw
    power = raw.power_raw
    height = raw.height_raw

    # map to 0..1
    straight01 = _map01(straight, *ranges.acc_straight)
    smooth01 = _map01(smooth, *ranges.acc_smooth)
    stable01 = _map01(stable, *ranges.acc_stable)
    rec01 = _map01(rec, *ranges.acc_recover)

    speed01 = _map01(speed, *ranges.expr_speed)
    power01 = _map01(power, *ranges.expr_power)
    height01 = _map01(height, *ranges.expr_height)
    down01 = _map01(raw.down_speed, *ranges.expr_downspeed)

    # technique cues
    knee_v01 = _map01(raw.knee_lift_vertical, *ranges.knee_lift_v)
    knee_lat01 = _map01(raw.knee_lift_lateral, *ranges.knee_lift_lat)

    # support knee angle
    sup_knee01 = _map01(raw.support_knee_angle_med, 150.0, 175.0)

    # kick knee angle peak
    k01 = _map01(raw.kick_knee_angle_peak, 135.0, 175.0)

    # landing (weak)
    lf = float(raw.landing_forward if raw.landing_forward is not None else 0.0)
    landing01 = _map01(lf, -0.08, 0.08)

    # ---------- Accuracy (4) ----------
    if k is KickType.front:
        # A1
        lift01 = _map01(raw.knee_lift_vertical, 0.04, 0.12)
        line01 = _map01(straight, 0.72, 0.96)
        got = (0.60 * lift01 + 0.40 * line01) * 1.6
        _add_item(acc_items, category="accuracy", code="FR_A1",
                  label="提膝先行与路线直",
                  maxi=1.6, got=got,
                  reason=f"提膝≈{raw.knee_lift_vertical:.2f}，直线度≈{straight:.2f}",
                  tip="先提膝再踢出；膝盖朝前，脚沿直线送出。",
                  key=True)

        # A2
        got = (0.55 * height01 + 0.45 * k01) * 1.2
        _add_item(acc_items, category="accuracy", code="FR_A2",
                  label="高度与膝伸展",
                  maxi=1.2, got=got,
                  reason=f"踝高≈{raw.kick_height_over_hip:.2f}，最大膝角≈{raw.kick_knee_angle_peak:.0f}°",
                  tip="高度逐步提高；膝伸展要快但别抛腿，到位后能回收。",
                  key=True)

        # A3
        got = rec01 * 0.8
        _add_item(acc_items, category="accuracy", code="FR_A3",
                  label="回收还原",
                  maxi=0.8, got=got,
                  reason=f"回收比例≈{raw.recovery_ratio:.2f}，再弯膝≈{raw.recovery_knee_bend_deg:.0f}°",
                  tip="踢出后先收膝再落脚，回到起始准备状态。",
                  key=True)

        # A4
        got = (0.65 * stable01 + 0.35 * (1.0 - guard / 0.25)) * 0.4
        _add_item(acc_items, category="accuracy", code="FR_A4",
                  label="重心与手位",
                  maxi=0.4, got=got,
                  reason=f"稳定≈{stable:.2f}，手位下沉≈{guard:.2f}",
                  tip="支撑脚稳、髋不要晃；手位保持护手，不要大幅下垂。",
                  key=False)

    elif k is KickType.roundhouse:
        # A1: alignment (important)
        if raw.roundhouse_align_dev_deg is None:
            align01 = 0.55
            dev = float("nan")
        else:
            dev = float(raw.roundhouse_align_dev_deg)
            align01 = _map01(30.0 - dev, 5.0, 25.0)
        got = (0.70 * align01 + 0.30 * straight01) * 1.6
        _add_item(acc_items, category="accuracy", code="RH_A1",
                  label="摆腿平面与击打方向",
                  maxi=1.6, got=got,
                  reason=f"对齐偏差≈{dev:.1f}°，直线度≈{straight:.2f}",
                  tip="横踢摆腿要在水平面；脚背/胫骨方向与击打方向一致。",
                  key=True)

        # A2
        got = (0.55 * height01 + 0.45 * k01) * 1.2
        _add_item(acc_items, category="accuracy", code="RH_A2",
                  label="高度与膝伸展",
                  maxi=1.2, got=got,
                  reason=f"踝高≈{raw.kick_height_over_hip:.2f}，最大膝角≈{raw.kick_knee_angle_peak:.0f}°",
                  tip="抬腿高度稳步增加；腿到位再甩出，注意回收。",
                  key=True)

        # A3
        got = rec01 * 0.8
        _add_item(acc_items, category="accuracy", code="RH_A3",
                  label="回收还原",
                  maxi=0.8, got=got,
                  reason=f"回收比例≈{raw.recovery_ratio:.2f}，再弯膝≈{raw.recovery_knee_bend_deg:.0f}°",
                  tip="踢出后先收膝再落脚，保持能连贯衔接下一动作。",
                  key=True)

        # A4
        got = (0.60 * stable01 + 0.40 * landing01) * 0.4
        _add_item(acc_items, category="accuracy", code="RH_A4",
                  label="重心与落脚控制",
                  maxi=0.4, got=got,
                  reason=f"稳定≈{stable:.2f}，落脚前后≈{lf:.2f}",
                  tip="重心别前冲；落脚要轻、稳，能立刻进入下一步。",
                  key=False)

    elif k is KickType.side:
        # A1: chamber (knee lift + lateral)
        got = (0.55 * knee_v01 + 0.45 * knee_lat01) * 1.6
        _add_item(acc_items, category="accuracy", code="SD_A1",
                  label="收腿与出腿路线",
                  maxi=1.6, got=got,
                  reason=f"提膝≈{raw.knee_lift_vertical:.2f}，侧向准备≈{raw.knee_lift_lateral:.2f}",
                  tip="侧踢先收腿（膝靠胸），脚跟朝目标，沿直线推出。",
                  key=True)

        # A2
        got = (0.55 * height01 + 0.45 * k01) * 1.2
        _add_item(acc_items, category="accuracy", code="SD_A2",
                  label="高度与腿到位",
                  maxi=1.2, got=got,
                  reason=f"踝高≈{raw.kick_height_over_hip:.2f}，最大膝角≈{raw.kick_knee_angle_peak:.0f}°",
                  tip="脚跟顶出去；到位后能快速收回。",
                  key=True)

        # A3
        got = rec01 * 0.8
        _add_item(acc_items, category="accuracy", code="SD_A3",
                  label="回收还原",
                  maxi=0.8, got=got,
                  reason=f"回收比例≈{raw.recovery_ratio:.2f}，再弯膝≈{raw.recovery_knee_bend_deg:.0f}°",
                  tip="踢完立刻收腿再落脚，保持“收—出—收”的节奏。",
                  key=True)

        # A4
        got = (0.65 * stable01 + 0.35 * (1.0 - guard / 0.25)) * 0.4
        _add_item(acc_items, category="accuracy", code="SD_A4",
                  label="重心与手位",
                  maxi=0.4, got=got,
                  reason=f"稳定≈{stable:.2f}，手位下沉≈{guard:.2f}",
                  tip="身体略侧但别塌腰；护手保持，支撑脚稳。",
                  key=False)

    else:  # axe
        # A1: upward path & height
        got = (0.55 * height01 + 0.45 * straight01) * 1.6
        _add_item(acc_items, category="accuracy", code="AX_A1",
                  label="抬腿高度与路径",
                  maxi=1.6, got=got,
                  reason=f"踝高≈{raw.kick_height_over_hip:.2f}，直线度≈{straight:.2f}",
                  tip="先抬高再下劈；抬腿轨迹稳定、别乱晃。",
                  key=True)

        # A2: down speed (important for axe)
        got = down01 * 1.2
        _add_item(acc_items, category="accuracy", code="AX_A2",
                  label="下劈下行速度",
                  maxi=1.2, got=got,
                  reason=f"下行速度≈{raw.down_speed:.2f}",
                  tip="下劈要有“落下去”的速度，但注意控制不要猛砸伤膝。",
                  key=True)

        # A3: recovery
        got = rec01 * 0.8
        _add_item(acc_items, category="accuracy", code="AX_A3",
                  label="回收还原",
                  maxi=0.8, got=got,
                  reason=f"回收比例≈{raw.recovery_ratio:.2f}，再弯膝≈{raw.recovery_knee_bend_deg:.0f}°",
                  tip="劈完也要能收回；别直接甩腿落地。",
                  key=True)

        # A4: stability & guard
        got = (0.60 * stable01 + 0.40 * (1.0 - guard / 0.25)) * 0.4
        _add_item(acc_items, category="accuracy", code="AX_A4",
                  label="重心与手位",
                  maxi=0.4, got=got,
                  reason=f"稳定≈{stable:.2f}，手位下沉≈{guard:.2f}",
                  tip="抬腿时骨盆别歪；手位别散。",
                  key=False)

    acc4 = float(sum([it["got"] for it in acc_items]))
    acc4 = _clamp(acc4, 0.0, 4.0)

    # ---------- Expression (6) ----------
    # E1 speed
    got = speed01 * 2.0
    _add_item(expr_items, category="expression", code="EX_E1",
              label="速度",
              maxi=2.0, got=got,
              reason=f"速度指标≈{speed:.2f}",
              tip="加快踢出速度：先放松、再瞬间发力；注意不要牺牲路线。",
              key=True)

    # E2 power
    got = power01 * 1.6
    _add_item(expr_items, category="expression", code="EX_E2",
              label="力量感",
              maxi=1.6, got=got,
              reason=f"爆发指标≈{power:.2f}",
              tip="力量来自髋部与支撑脚；到位要紧、落点要清晰。",
              key=True)

    # E3 height
    got = height01 * 1.4
    _add_item(expr_items, category="expression", code="EX_E3",
              label="高度",
              maxi=1.4, got=got,
              reason=f"高度指标≈{height:.2f}",
              tip="逐步提高高度，不要硬拉；保持上身直立，支撑脚稳。",
              key=False)

    # E4 smoothness
    got = smooth01 * 1.0
    _add_item(expr_items, category="expression", code="EX_E4",
              label="流畅性",
              maxi=1.0, got=got,
              reason=f"流畅≈{smooth:.2f}",
              tip="动作不要卡顿；多做慢—快结合训练，轨迹更顺。",
              key=False)

    expr6 = float(sum([it["got"] for it in expr_items]))
    expr6 = _clamp(expr6, 0.0, 6.0)

    total10 = float(acc4 + expr6)
    total10 = _clamp(total10, 0.0, 10.0)

    all_items = acc_items + expr_items
    top_issues = _summarize_top_issues(all_items, topk=6)
    tips = [d["tip"] for d in top_issues if d.get("tip")]

    details = {
        "acc_items": acc_items,
        "expr_items": expr_items,
        "top_issues": top_issues,
        "tips": tips[:5],
        "raw": raw.__dict__,
    }
    return acc4, expr6, total10, details