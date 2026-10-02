# scripts/scorers_stances.py
"""
Luật chấm TẤN (thế đứng): đứng trung bình tấn (juchum seogi), tấn trước dài (ap kubi).

Tấn là tư thế giữ yên nên không có "điểm chạm"; vẫn giữ khung 10 = Chính xác 4 + Biểu hiện 6 của tác giả
để so sánh được với đá/đấm:
  Chính xác 4 = A1 1.6 (gối) + A2 1.2 (độ rộng/dài tấn) + A3 0.8 (thân thẳng) + A4 0.4 (cân đối / tay)
  Biểu hiện 6 = E1 2.0 độ vững (giữ yên) + E2 1.6 trọng tâm thấp + E3 1.4 giữ tấn liên tục
              + E4 1.0 thân trên ổn định
Đo trên tọa độ 3D MediaPipe (world), trong đoạn đứng yên dài nhất của clip. Ngưỡng trong StanceRanges.

stance_at(): đo nhanh độ rộng tấn / độ khuỵu gối tại 1 frame — dùng cho các kỹ thuật tay có bước mở chân.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from action_scorers import ALIASES, ScoreResult, register_scorer
from punch_scoring import _add_item, _band01, _component, _interp_nan, _map01, _smooth, _vec_angle

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)


@dataclass
class StanceRanges:
    # trung bình tấn
    horse_knee: Tuple[float, float, float, float] = (90.0, 105.0, 145.0, 165.0)
    horse_width: Tuple[float, float, float, float] = (1.3, 1.7, 2.9, 3.5)   # khoảng cách 2 cổ chân / rộng vai
    horse_depth: Tuple[float, float] = (0.95, 0.80)                          # chiều cao hông / dài chân (thấp = tốt)
    knees_out: Tuple[float, float] = (0.65, 0.85)                            # khoảng cách 2 gối / 2 cổ chân
    # tấn trước dài
    front_knee: Tuple[float, float, float, float] = (80.0, 95.0, 135.0, 155.0)
    rear_knee: Tuple[float, float] = (145.0, 168.0)
    front_length: Tuple[float, float, float, float] = (1.0, 1.5, 3.0, 3.6)  # dọc theo hướng mặt, / rộng vai (3D)
    front_length_2d: Tuple[float, float, float, float] = (0.7, 1.05, 2.1, 2.5)  # quay nghiêng: / chiều dài thân (2D)
    front_lateral: Tuple[float, float, float, float] = (0.1, 0.3, 1.2, 1.6)  # bề ngang 2 chân / rộng vai (3D)
    side_on: float = 0.45                                                    # rộng vai / dài thân (2D) dưới mức này = quay nghiêng
    front_depth: Tuple[float, float] = (0.97, 0.85)
    # chung
    lean: Tuple[float, float] = (20.0, 7.0)                # độ nghiêng thân so với phương thẳng đứng
    asym: Tuple[float, float] = (25.0, 8.0)                # |gối trái - gối phải| (trung bình tấn)
    sway: Tuple[float, float] = (0.06, 0.015)              # lắc tâm hông (× thân) khi giữ tấn
    hold_frac: Tuple[float, float] = (0.4, 0.9)            # tỉ lệ thời gian đạt tấn trong clip
    trunk_std: Tuple[float, float] = (6.0, 1.5)            # dao động góc thân (độ)
    # bước mở chân (kỹ thuật tay)
    open_width: Tuple[float, float] = (1.1, 1.8)
    open_knee: Tuple[float, float] = (172.0, 145.0)


SR = StanceRanges()


def _prep(world: np.ndarray) -> np.ndarray:
    return _smooth(_interp_nan(world), 5)


def _knee(w, H, K, A):
    return _vec_angle(w[:, H] - w[:, K], w[:, A] - w[:, K])


def _frame_metrics(w: np.ndarray) -> Dict[str, np.ndarray]:
    sh = 0.5 * (w[:, LS] + w[:, RS])
    hip = 0.5 * (w[:, LH] + w[:, RH])
    Ws = np.median(np.linalg.norm(w[:, LS] - w[:, RS], axis=1))
    leg = np.median(np.linalg.norm(w[:, LH] - w[:, LK], axis=1) + np.linalg.norm(w[:, LK] - w[:, LA], axis=1))
    kL, kR = _knee(w, LH, LK, LA), _knee(w, RH, RK, RA)
    width = np.linalg.norm((w[:, LA] - w[:, RA])[:, [0, 2]], axis=1) / Ws
    knee_gap = np.linalg.norm((w[:, LK] - w[:, RK])[:, [0, 2]], axis=1) / (Ws * width + 1e-9)
    trunk = sh - hip
    lean = np.degrees(np.arccos(np.clip(-trunk[:, 1] / (np.linalg.norm(trunk, axis=1) + 1e-9), -1, 1)))
    depth = (0.5 * (w[:, LA, 1] + w[:, RA, 1]) - hip[:, 1]) / leg   # world y hướng xuống
    # hướng mặt (ngang): vuông góc trục vai, cùng phía với mũi
    ax = (w[:, LS] - w[:, RS])[:, [0, 2]]
    fwd = np.stack([-ax[:, 1], ax[:, 0]], 1)
    nose = (w[:, HEAD] - sh)[:, [0, 2]]
    fwd *= np.sign(np.sum(fwd * nose, axis=1, keepdims=True) + 1e-9)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    dfeet = (w[:, LA] - w[:, RA])[:, [0, 2]]
    length = np.abs(np.sum(dfeet * fwd, axis=1)) / Ws        # khoảng cách 2 chân theo hướng trước–sau
    return {"kL": kL, "kR": kR, "width": width, "knee_gap": knee_gap, "lean": lean, "depth": depth,
            "length": length, "Ws": Ws}


def stance_at(world: Optional[np.ndarray], kpts: np.ndarray, t: int) -> Dict:
    """Độ rộng tấn & khuỵu gối quanh frame t (±3 frame). open01: mức 'bước mở chân, khuỵu gối'."""
    if world is None or world.shape[0] == 0:
        return {"width": float("nan"), "knee_min": float("nan"), "open01": 0.0}
    w = _prep(world)
    f = _frame_metrics(w)
    sl = slice(max(0, t - 3), min(w.shape[0], t + 4))
    width = float(np.median(f["width"][sl]))
    knee_min = float(np.median(np.minimum(f["kL"], f["kR"])[sl]))
    open01 = 0.5 * _map01(width, *SR.open_width) + 0.5 * _map01(knee_min, *SR.open_knee)
    return {"width": width, "knee_min": knee_min, "open01": float(open01)}


def _hold_run(w: np.ndarray, fps: float, thr: float = 0.6) -> Tuple[int, int]:
    """Đoạn đứng yên dài nhất: tốc độ trung bình các khớp < thr (× thân / giây)."""
    sh = 0.5 * (w[:, LS] + w[:, RS])
    hip = 0.5 * (w[:, LH] + w[:, RH])
    Lt = float(np.median(np.linalg.norm(sh - hip, axis=1))) + 1e-9
    sp = np.r_[0.0, np.mean(np.linalg.norm(np.diff(w, axis=0), axis=2), axis=1) * fps / Lt]
    still = sp < thr
    best, cur = (0, -1), None
    for t, s in enumerate(still):
        if s and cur is None:
            cur = t
        if (not s or t == len(still) - 1) and cur is not None:
            e = t if s else t - 1
            if e - cur > best[1] - best[0]:
                best = (cur, e)
            cur = None
    return best


def _score_stance(pose: Dict, action: str, **_) -> ScoreResult:
    world, kpts, fps = pose.get("world"), pose["kpts"], float(pose["fps"])
    T = kpts.shape[0]
    common = dict(action=action, phase_labels=("Bắt đầu giữ", "Giữa", "Kết thúc giữ"),
                  phase_names=("Vào tấn", "Giữ tấn", "GIỮ TẤN", "Giữ tấn", "Ra tấn"), highlight=(LH, LK, LA, RH, RK, RA))
    if world is None or T < 8:
        return ScoreResult(valid=False, reason="Video quá ngắn hoặc thiếu tọa độ 3D", **common)
    w = _prep(world)
    f = _frame_metrics(w)
    s, e = _hold_run(w, fps)
    if e - s + 1 < int(0.5 * fps):
        return ScoreResult(valid=False, reason="Không thấy đoạn giữ tấn (đứng yên ≥ 0.5 s).", **common)
    sl = slice(s, e + 1)
    med = {k: float(np.median(v[sl])) for k, v in f.items() if k != "Ws"}
    # trung bình tấn phải quay chính diện: quay nghiêng thì 2 chân chồng nhau theo chiều sâu, MediaPipe đo sai
    if action == "horse_stance":
        sw2 = np.linalg.norm(kpts[sl, LS] - kpts[sl, RS], axis=1)
        tr2 = np.linalg.norm(0.5 * (kpts[sl, LS] + kpts[sl, RS]) - 0.5 * (kpts[sl, LH] + kpts[sl, RH]), axis=1)
        if float(np.nanmedian(sw2 / (tr2 + 1e-6))) < 0.40:
            return ScoreResult(valid=False, reason="Người quay nghiêng — trung bình tấn cần quay chính diện để đo.",
                               **common)
    # đứng thẳng / đã ra khỏi tấn (vd lúc nghỉ giữa bài) -> không chấm, thay vì chấm như một thế tấn kém
    kmin = min(med["kL"], med["kR"])
    if kmin > (SR.horse_knee[3] if action == "horse_stance" else SR.front_knee[3]) + 5:
        return ScoreResult(valid=False, reason=f"Không ở thế tấn (gối khuỵu nhất ≈{kmin:.0f}°).", **common)
    hip2 = 0.5 * (kpts[:, LH] + kpts[:, RH])
    L2 = float(np.nanmedian(np.linalg.norm(0.5 * (kpts[:, LS] + kpts[:, RS]) - hip2, axis=1))) + 1e-6
    sway = float(np.max(np.nanstd(hip2[sl] / L2, axis=0)))
    trunk_std = float(np.std(f["lean"][sl]))  # (độ dao động: 3D đủ dùng ở cả hai góc quay)
    acc: List[Dict] = []
    expr: List[Dict] = []
    if action == "horse_stance":
        k_ok = 0.5 * _band01(med["kL"], *SR.horse_knee) + 0.5 * _band01(med["kR"], *SR.horse_knee)
        in_stance = ((np.minimum(f["kL"], f["kR"]) < SR.horse_knee[3]) & (f["width"] > SR.horse_width[0]))
        _add_item(acc, "accuracy", "HS_A1", "Độ khuỵu hai gối", 1.6, k_ok * 1.6,
                  f"gối trái≈{med['kL']:.0f}°, gối phải≈{med['kR']:.0f}°",
                  "Khuỵu gối sâu (đùi gần song song mặt đất), hai gối đẩy ra ngoài theo hướng mũi chân.", key=True)
        a2 = 0.6 * _band01(med["width"], *SR.horse_width) + 0.4 * _map01(med["knee_gap"], *SR.knees_out)
        _add_item(acc, "accuracy", "HS_A2", "Độ rộng tấn & gối mở", 1.2, a2 * 1.2,
                  f"hai chân rộng≈{med['width']:.1f}× vai, khoảng cách gối/cổ chân≈{med['knee_gap']:.2f}",
                  "Hai bàn chân song song, rộng khoảng 2 lần vai; không để gối khép vào trong.", key=True)
        _add_item(acc, "accuracy", "HS_A3", "Thân thẳng", 0.8, _map01(med["lean"], *SR.lean) * 0.8,
                  f"thân nghiêng≈{med['lean']:.0f}°", "Lưng thẳng, không chúi người ra trước hay ngả sau.")
        _add_item(acc, "accuracy", "HS_A4", "Cân đối hai bên", 0.4,
                  _map01(abs(med["kL"] - med["kR"]), *SR.asym) * 0.4,
                  f"chênh lệch hai gối≈{abs(med['kL'] - med['kR']):.0f}°", "Trọng tâm chia đều hai chân.")
        depth_rng = SR.horse_depth
    else:  # front_stance
        # quay nghiêng (tấn trước thường quay từ bên cạnh): 2D chính xác trong mặt phẳng dọc, còn 3D của MediaPipe
        # sai độ sâu -> dùng góc gối, độ nghiêng, chiều dài tấn 2D; bề ngang 2 chân (theo chiều sâu) không đo được.
        k2 = _smooth(_interp_nan(kpts), 5)
        tr2 = np.linalg.norm(0.5 * (k2[:, LS] + k2[:, RS]) - 0.5 * (k2[:, LH] + k2[:, RH]), axis=1) + 1e-6
        side_on = float(np.median((np.linalg.norm(k2[:, LS] - k2[:, RS], axis=1) / tr2)[sl])) < SR.side_on
        if side_on:
            kLs, kRs = _knee(k2, LH, LK, LA), _knee(k2, RH, RK, RA)
            tv = 0.5 * (k2[:, LS] + k2[:, RS]) - 0.5 * (k2[:, LH] + k2[:, RH])
            lean_s = np.degrees(np.arctan2(np.abs(tv[:, 0]), -tv[:, 1]))
            length = float(np.median((np.abs(k2[:, LA, 0] - k2[:, RA, 0]) / tr2)[sl]))
            len_rng, len_txt = SR.front_length_2d, f"chiều dài tấn≈{length:.1f}× thân (2D, quay nghiêng)"
        else:
            kLs, kRs, lean_s = f["kL"], f["kR"], f["lean"]
            length = med["length"]
            len_rng, len_txt = SR.front_length, f"chiều dài tấn≈{length:.1f}× vai"
        mkL, mkR, lean_m = (float(np.median(x[sl])) for x in (kLs, kRs, lean_s))
        med.update(kL=mkL, kR=mkR, lean=lean_m, length=length, side_on=float(side_on))
        front_is_L = mkL < mkR
        kf, kr = (mkL, mkR) if front_is_L else (mkR, mkL)
        kfs, krs = (kLs, kRs) if front_is_L else (kRs, kLs)
        in_stance = (kfs < SR.front_knee[3]) & (krs > SR.rear_knee[0])
        a1 = 0.6 * _band01(kf, *SR.front_knee) + 0.4 * _map01(kr, *SR.rear_knee)
        _add_item(acc, "accuracy", "FS_A1", "Gối trước khuỵu, chân sau thẳng", 1.6, a1 * 1.6,
                  f"gối trước≈{kf:.0f}°, gối sau≈{kr:.0f}°" + (" (2D)" if side_on else ""),
                  "Gối trước khuỵu ngay trên cổ chân, chân sau duỗi thẳng, gót sau chạm đất.", key=True)
        _add_item(acc, "accuracy", "FS_A2", "Độ dài tấn", 1.2, _band01(length, *len_rng) * 1.2, len_txt,
                  "Bước dài khoảng 1,5–2 bước chân thường (~2 lần rộng vai).", key=True)
        _add_item(acc, "accuracy", "FS_A3", "Thân thẳng", 0.8, _map01(lean_m, *SR.lean) * 0.8,
                  f"thân nghiêng≈{lean_m:.0f}°", "Thân thẳng đứng, không đổ người ra trước.")
        lateral = float(np.sqrt(max(med["width"] ** 2 - med["length"] ** 2, 0.0))) if not side_on else None
        _add_item(acc, "accuracy", "FS_A4", "Hai chân không trên một đường", 0.4,
                  None if lateral is None else _band01(lateral, *SR.front_lateral) * 0.4,
                  "quay nghiêng — không đo được bề ngang 2 chân" if lateral is None
                  else f"bề ngang 2 chân≈{lateral:.1f}× vai",
                  "Hai chân cách nhau theo chiều ngang khoảng một nắm tay đến một vai, không đứng trên một đường.")
        depth_rng = SR.front_depth
    hold = float(np.mean(in_stance))
    _add_item(expr, "expression", "SX_E1", "Độ vững (giữ yên)", 2.0, _map01(sway, *SR.sway) * 2.0,
              f"lắc tâm hông≈{sway:.3f} thân", "Giữ yên trọng tâm, không nhấp nhô hay lắc ngang.", key=True)
    _add_item(expr, "expression", "SX_E2", "Trọng tâm thấp", 1.6, _map01(med["depth"], *depth_rng) * 1.6,
              f"chiều cao hông≈{med['depth']:.2f} dài chân", "Hạ thấp trọng tâm bằng cách khuỵu gối, không cúi lưng.",
              key=True)
    _add_item(expr, "expression", "SX_E3", "Giữ tấn liên tục", 1.4, _map01(hold, *SR.hold_frac) * 1.4,
              f"đạt tấn≈{hold * 100:.0f}% thời gian", "Giữ nguyên tấn suốt bài, không đứng lên nghỉ giữa chừng.")
    _add_item(expr, "expression", "SX_E4", "Thân trên ổn định", 1.0, _map01(trunk_std, *SR.trunk_std) * 1.0,
              f"dao động góc thân≈{trunk_std:.1f}°", "Thân trên tĩnh, vai thả lỏng.")
    acc4, expr6 = _component(acc, 4.0), _component(expr, 6.0)
    top = sorted([d for d in acc + expr if d["deduct"] > 1e-6], key=lambda d: -d["deduct"])[:6]
    mid = (s + e) // 2
    return ScoreResult(
        valid=True, total=acc4 + expr6, acc=acc4, expr=expr6, acc_items=acc, expr_items=expr, top_issues=top,
        phases={"t0": s, "t_peak": mid, "t_end": e},
        series={"sig": np.minimum(f["kL"], f["kR"]), "sig_name": "Góc gối nhỏ nhất (°)",
                "speed": f["width"], "speed_name": "Độ rộng tấn (× vai)"},
        side_text="", status=f"gối {med['kL']:.0f}°/{med['kR']:.0f}°, rộng {med['width']:.1f}× vai",
        raw=dict(med, sway=sway, trunk_std=trunk_std, hold_frac=hold, hold_s=(e - s + 1) / fps), **common)


_score_stance.window_s = 3.0       # score_segments.py: chia đoạn giữ tấn dài thành các cửa sổ 3 s
register_scorer("horse_stance", "Juchum seogi (đứng trung bình tấn)")(_score_stance)
register_scorer("front_stance", "Ap kubi (tấn trước dài)")(_score_stance)

ALIASES.update({"juchum_seogi": "horse_stance", "trung_binh_tan": "horse_stance", "ap_kubi": "front_stance",
                "tan_truoc_dai": "front_stance"})
