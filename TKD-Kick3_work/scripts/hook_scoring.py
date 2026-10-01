# -*- coding: utf-8 -*-
"""
hook_scoring.py
Rule-based scoring for the right hook punch (hook_right), same 10-point layout as
scoring_api.py: Accuracy 4 + Expression 6, with UI-ready deduction items.

Assumptions (2D image plane, like scoring_api.py):
- performer roughly faces the camera; right wrist = attacking hand
- kpts (T,13,2) pixel coords, vis (T,13), fps; typically the wrist-window punch view
  produced by hook_pose.py

Geometry is normalized by the median torso length (shoulder center -> hip center) and
expressed relative to the shoulder center, so metrics are resolution-independent.

"power" here is a speed/acceleration proxy (explosiveness), NOT biomechanical power.
Score ranges live in outputs/hook/hook_score_ranges.json (see calibrate_ranges); they are
provisional and need expert confirmation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
VTH = 0.2

ROOT = Path(__file__).resolve().parents[1]
RANGES_PATH = ROOT / "outputs" / "hook" / "hook_score_ranges.json"


# ----------------------------
# Raw metrics
# ----------------------------
@dataclass
class HookRaw:
    T: int
    fps: float
    valid_ratio: float
    t_impact: int
    right_left_path_ratio: float  # >1 means the right wrist is the more active one
    elbow_angle_deg: float        # right elbow angle at impact (shoulder-elbow-wrist)
    elbow_level: float            # (shoulder_y - elbow_y)/L at impact; ~0 = elbow at shoulder height
    forearm_tilt: float           # |elbow_y - wrist_y|/L at impact; ~0 = horizontal forearm
    fist_height: float            # (shoulder_y - wrist_y)/L at impact; >0 = fist above shoulder line
    reach: float                  # max wrist displacement from guard / L
    shoulder_turn: float          # 1 - min projected shoulder width / initial (torso rotation proxy)
    hip_turn: float               # same for hips
    guard_dist: float             # median |left wrist - head| / L (lower = better guard)
    recovery: float               # 1 - final displacement / max displacement
    base_sway: float              # hip-center travel range / L (lower = more stable)
    speed: float                  # p90 wrist speed (torso lengths/s), relative to shoulder center
    power: float                  # 0.7*speed + 0.3*accel/10  (explosiveness proxy)
    smoothness: float             # 1 for a single-peak speed profile up to impact


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linear fill of NaNs along time for (T,) or (T,C)."""
    a = a.astype(np.float32).copy()
    flat = a.reshape(a.shape[0], -1)
    t = np.arange(a.shape[0])
    for c in range(flat.shape[1]):
        m = np.isfinite(flat[:, c])
        if m.sum() == 0:
            flat[:, c] = 0.0
        elif (~m).any():
            flat[~m, c] = np.interp(t[~m], t[m], flat[m, c])
    return flat.reshape(a.shape)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def compute_raw(kpts: np.ndarray, vis: np.ndarray, fps: float) -> HookRaw:
    k = np.asarray(kpts, np.float32)[..., :2].copy()
    v = np.asarray(vis, np.float32)
    T = k.shape[0]
    if T < 4:
        raise ValueError("sequence too short for hook scoring")
    fps = float(fps) if fps and fps > 0 else 30.0

    k[v < VTH] = np.nan
    valid_ratio = float(np.mean(v[:, [RS, RE, RW, LS, LW, HEAD]] > VTH))
    k = _interp_nan(k)

    n0 = min(3, T)
    sc = 0.5 * (k[:, LS] + k[:, RS])
    # scale = torso length (shoulder center -> hip center): unlike shoulder width it does
    # not collapse when the performer turns sideways during the hook
    L = float(np.median(np.linalg.norm(sc - 0.5 * (k[:, LH] + k[:, RH]), axis=1))) or 1.0
    sw_t = np.linalg.norm(k[:, LS] - k[:, RS], axis=1)
    hw_t = np.linalg.norm(k[:, LH] - k[:, RH], axis=1)
    sw0 = max(float(np.max(sw_t[:n0])), 1e-6)
    hw0 = max(float(np.max(hw_t[:n0])), 1e-6)

    rel = (k - sc[:, None, :]) / L  # shoulder-centered, torso-length units; y down

    rw = rel[:, RW]
    guard_pos = np.median(rw[:n0], axis=0)
    disp = np.linalg.norm(rw - guard_pos, axis=1)
    ti = int(np.argmax(_smooth(disp)))
    win = slice(max(0, ti - 1), min(T, ti + 2))

    def at(j: int) -> np.ndarray:
        return np.median(rel[win, j], axis=0)

    elbow = _angle(at(RS), at(RE), at(RW))
    elbow_level = float(at(RS)[1] - at(RE)[1])
    forearm_tilt = float(abs(at(RE)[1] - at(RW)[1]))
    fist_height = float(at(RS)[1] - at(RW)[1])

    upto = slice(0, min(T, ti + 4))
    shoulder_turn = float(np.clip(1.0 - np.min(sw_t[upto]) / sw0, 0.0, 1.0))
    hip_turn = float(np.clip(1.0 - np.min(hw_t[upto]) / hw0, 0.0, 1.0))

    guard_dist = float(np.median(np.linalg.norm(rel[:, LW] - rel[:, HEAD], axis=1)))
    recovery = float(np.clip(1.0 - disp[-1] / max(disp[ti], 1e-6), 0.0, 1.0))

    hip_c = 0.5 * (k[:, LH] + k[:, RH]) / L
    base_sway = float(np.linalg.norm(np.percentile(hip_c, 95, axis=0) - np.percentile(hip_c, 5, axis=0)))

    sp = _smooth(np.linalg.norm(np.diff(rw, axis=0), axis=1) * fps)
    ac = _smooth(np.abs(np.diff(sp)) * fps) if len(sp) > 1 else np.zeros(1)
    speed = float(np.percentile(sp, 90))
    power = float(0.7 * speed + 0.3 * np.percentile(ac, 90) / 10.0)

    sp_strike = sp[:max(2, min(len(sp), ti + 2))]
    rises = float(np.sum(np.abs(np.diff(sp_strike))))
    smoothness = float(np.clip(2.0 * (sp_strike.max() - 0.5 * (sp_strike[0] + sp_strike[-1])) / max(rises, 1e-6), 0, 1))

    def path(j: int) -> float:
        return float(np.sum(np.linalg.norm(np.diff(rel[:, j], axis=0), axis=1)))

    return HookRaw(
        T=T, fps=fps, valid_ratio=valid_ratio, t_impact=ti,
        right_left_path_ratio=path(RW) / max(path(LW), 1e-6),
        elbow_angle_deg=elbow, elbow_level=elbow_level, forearm_tilt=forearm_tilt,
        fist_height=fist_height, reach=float(disp[ti]), shoulder_turn=shoulder_turn, hip_turn=hip_turn,
        guard_dist=guard_dist, recovery=recovery, base_sway=base_sway,
        speed=speed, power=power, smoothness=smoothness,
    )


# ----------------------------
# Ranges
# ----------------------------
# Technique targets (fixed, from hook technique; provisional until expert-confirmed).
# Units: torso lengths (L). Pairs are [value for 0 credit, value for full credit].
FIXED_RANGES = {
    "elbow_ideal": [75.0, 115.0],   # full credit inside, linear falloff over `elbow_margin`
    "elbow_margin": 45.0,
    "elbow_level": [-0.45, -0.1],   # elbow at least near shoulder height
    "forearm_tilt": [0.6, 0.2],     # forearm close to horizontal
    "guard_dist": [0.8, 0.4],       # left wrist stays near the head
    "valid_ratio_min": 0.6,
}
# Calibrated on reference (instructor) clips: full credit at the reference median,
# zero credit at `zero_factor` x median (for base_sway lower is better).
CALIBRATION = {
    "speed": 0.5, "power": 0.5, "smoothness": 0.5,
    "shoulder_turn": 0.0, "hip_turn": 0.0, "recovery": 0.0,
    "base_sway": 2.0,
}


def calibrate_ranges(raws: List[HookRaw]) -> Dict[str, Any]:
    cal = {}
    for key, zero_factor in CALIBRATION.items():
        med = float(np.median([getattr(r, key) for r in raws]))
        cal[key] = [zero_factor * med, med]
    return {"fixed": FIXED_RANGES, "calibrated": cal, "calibration_rule": CALIBRATION,
            "n_reference": len(raws),
            "note": "provisional thresholds; full credit = median of reference clips; needs expert confirmation"}


def load_ranges(path: Path = RANGES_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python scripts/hook_scoring.py --calibrate` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _map01(x: float, a: float, b: float) -> float:
    """Linear 0..1 from a (0) to b (1); works for decreasing ranges (a > b)."""
    if not np.isfinite(x):
        return 0.0
    return float(np.clip((x - a) / (b - a), 0.0, 1.0)) if abs(b - a) > 1e-9 else 0.0


# ----------------------------
# Scoring
# ----------------------------
def _add(items: List[Dict], category: str, code: str, label: str, maxi: float, got: float,
         reason: str, tip: str, key: bool) -> None:
    got = float(np.clip(got, 0.0, maxi))
    d = maxi - got
    items.append({"category": category, "code": code, "label": label, "maxi": float(maxi), "got": got,
                  "deduct": float(d), "reason": reason, "tip": tip, "key": key,
                  "severity": "high" if d > 0.6 else ("mid" if d > 0.3 else "low")})


def score_hook(seq: Any, ranges: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float, Dict]:
    """
    seq: npz / dict with kpts, vis, fps  -- or a (kpts, vis, fps) tuple -- or HookRaw.
    Returns (acc4, expr6, total10, details) like scoring_api.score_one_video.
    """
    if isinstance(seq, HookRaw):
        raw = seq
    elif isinstance(seq, tuple):
        raw = compute_raw(*seq)
    else:
        raw = compute_raw(seq["kpts"], seq["vis"], float(seq["fps"]) if "fps" in seq else 30.0)

    rg = ranges or load_ranges()
    fx, cal = rg["fixed"], rg["calibrated"]

    # --- Accuracy (4) ---
    acc: List[Dict] = []
    lo, hi = fx["elbow_ideal"]
    m = fx["elbow_margin"]
    e = raw.elbow_angle_deg
    elbow01 = 1.0 if lo <= e <= hi else _map01(e, lo - m, lo) if e < lo else _map01(e, hi + m, hi)
    level01 = _map01(raw.elbow_level, *fx["elbow_level"])
    tilt01 = _map01(raw.forearm_tilt, *fx["forearm_tilt"])
    _add(acc, "accuracy", "HK_A1", "Góc khuỷu tay & tay ngang", 1.4,
         (0.5 * elbow01 + 0.25 * level01 + 0.25 * tilt01) * 1.4,
         f"góc khuỷu≈{e:.0f}°, khuỷu so với vai≈{raw.elbow_level:+.2f}, độ nghiêng cẳng tay≈{raw.forearm_tilt:.2f}",
         "Giữ khuỷu gập khoảng 90°, khuỷu nâng ngang vai, cẳng tay song song mặt đất khi chạm mục tiêu.", True)

    sh01 = _map01(raw.shoulder_turn, *cal["shoulder_turn"])
    hp01 = _map01(raw.hip_turn, *cal["hip_turn"])
    _add(acc, "accuracy", "HK_A2", "Xoay vai – hông", 1.2, (0.6 * sh01 + 0.4 * hp01) * 1.2,
         f"xoay vai≈{raw.shoulder_turn:.2f}, xoay hông≈{raw.hip_turn:.2f}",
         "Lực đấm móc đến từ xoay hông và vai; xoay chân sau, đừng chỉ vung tay.", True)

    guard01 = _map01(raw.guard_dist, *fx["guard_dist"])
    _add(acc, "accuracy", "HK_A3", "Tay trái thủ", 0.8, guard01 * 0.8,
         f"khoảng cách tay trái–đầu≈{raw.guard_dist:.2f}",
         "Tay trái luôn giữ sát cằm/má để bảo vệ trong lúc tay phải đấm.", True)

    rec01 = _map01(raw.recovery, *cal["recovery"])
    sway01 = _map01(raw.base_sway, *cal["base_sway"])
    _add(acc, "accuracy", "HK_A4", "Thu tay & giữ trụ", 0.6, (0.6 * rec01 + 0.4 * sway01) * 0.6,
         f"thu tay≈{raw.recovery:.2f}, dao động trọng tâm≈{raw.base_sway:.2f}",
         "Đấm xong thu tay về thế thủ ngay; giữ trọng tâm ổn định, không đổ người theo cú đấm.", False)

    # --- Expression (6) ---
    expr: List[Dict] = []
    sp01 = _map01(raw.speed, *cal["speed"])
    pw01 = _map01(raw.power, *cal["power"])
    sm01 = _map01(raw.smoothness, *cal["smoothness"])
    _add(expr, "expression", "EX_E1", "Tốc độ", 2.2, sp01 * 2.2, f"tốc độ nắm tay≈{raw.speed:.2f} (chiều dài thân/giây)",
         "Thả lỏng rồi bung nhanh; đường móc ngắn, gọn.", True)
    _add(expr, "expression", "EX_E2", "Độ bộc phát", 1.8, pw01 * 1.8, f"chỉ số bộc phát≈{raw.power:.2f}",
         "Tăng tốc dứt khoát ở đoạn cuối, siết nắm tay lúc chạm.", True)
    _add(expr, "expression", "EX_E3", "Phối hợp xoay người", 1.0, (0.5 * sh01 + 0.5 * hp01) * 1.0,
         f"xoay vai≈{raw.shoulder_turn:.2f}, xoay hông≈{raw.hip_turn:.2f}",
         "Hông – vai – tay xoay liền một nhịp.", False)
    _add(expr, "expression", "EX_E4", "Độ mượt", 1.0, sm01 * 1.0, f"độ mượt≈{raw.smoothness:.2f}",
         "Tránh khựng giữa chừng; tập chậm rồi tăng tốc dần.", False)

    acc4 = float(np.clip(sum(i["got"] for i in acc), 0, 4))
    expr6 = float(np.clip(sum(i["got"] for i in expr), 0, 6))
    total = float(np.clip(acc4 + expr6, 0, 10))

    warnings = []
    if raw.valid_ratio < fx["valid_ratio_min"]:
        warnings.append(f"keypoints of the arms/head poorly visible (valid≈{raw.valid_ratio:.2f})")
    if raw.right_left_path_ratio < 1.0:
        warnings.append("left wrist moved more than right wrist: check that this is a RIGHT hook")

    top = sorted([i for i in acc + expr if i["deduct"] > 1e-6], key=lambda d: d["deduct"], reverse=True)[:6]
    details = {"acc_items": acc, "expr_items": expr, "top_issues": top,
               "tips": [d["tip"] for d in top][:5], "warnings": warnings, "raw": asdict(raw)}
    return acc4, expr6, total, details


# ----------------------------
# CLI: calibrate ranges on the reference clips and score them
# ----------------------------
def main() -> None:
    import argparse
    import csv

    ap = argparse.ArgumentParser(description="Calibrate / score right-hook sequences.")
    ap.add_argument("--calibrate", action="store_true", help="(re)write hook_score_ranges.json from --ref")
    ap.add_argument("--ref", type=str, default=str(ROOT / "data" / "sequences_hook"),
                    help="folder of reference hook npz files (searched recursively)")
    args = ap.parse_args()

    files = sorted(Path(args.ref).rglob("*.npz"))
    if not files:
        raise SystemExit(f"[ERR] no npz under {args.ref}")
    raws = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        raws.append(compute_raw(z["kpts"], z["vis"], float(z["fps"])))

    if args.calibrate:
        RANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
        RANGES_PATH.write_text(json.dumps(calibrate_ranges(raws), indent=2), encoding="utf-8")
        print(f"[SAVE] {RANGES_PATH}")

    ranges = load_ranges()
    rows = []
    for f, r in zip(files, raws):
        acc4, expr6, total, d = score_hook(r, ranges)
        rows.append({"npz": str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f),
                     "accuracy_4": round(acc4, 3), "expression_6": round(expr6, 3), "total_10": round(total, 3),
                     **{i["code"]: round(i["got"], 3) for i in d["acc_items"] + d["expr_items"]},
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d["raw"].items()}})
        print(f"{f.stem:20s} total={total:5.2f}  (acc {acc4:.2f}/4, expr {expr6:.2f}/6)")
    out = ROOT / "outputs" / "hook" / "reference_clip_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tot = np.array([r["total_10"] for r in rows])
    print(f"[INFO] n={len(rows)} total mean={tot.mean():.2f} min={tot.min():.2f} max={tot.max():.2f}")
    print(f"[SAVE] {out}")


if __name__ == "__main__":
    main()
