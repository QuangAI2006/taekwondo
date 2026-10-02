# scripts/scorers_kicks_extra.py
"""
Luật chấm cho các đòn ĐÁ chưa có trong luật của tác giả, viết theo đúng cách của scoring_api.py:

  - đoạn cắt  : action_window(..., "ankle") — đúng hàm cắt của tác giả (như _score_kick)
  - chỉ số    : build_metrics_from_sequence() của tác giả (2D, chuẩn hóa theo bề rộng vai/hông)
  - thang điểm: Chính xác 4 = A1 1.6 (kỹ thuật riêng của đòn) + A2 1.2 + A3 0.8 (thu chân) + A4 0.4
                Biểu hiện 6 = EX_E1..E4 của tác giả, GIỮ NGUYÊN công thức và ngưỡng (ScoreRanges)
  - ánh xạ    : điểm mục = điểm tối đa × map(chỉ số, ngưỡng dưới, ngưỡng trên) (_map01 của tác giả)

Chỗ khác tác giả (ghi rõ trong từng hạng mục): vài chỉ số cần hướng 3D (đá ra sau hay ra trước) lấy từ tọa
độ world của MediaPipe; mọi ngưỡng khác lấy lại từ luật đá của tác giả. Xem OTHER_TECHNIQUES_SCORING.md.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from action_scorers import ALIASES, ScoreResult, _torso_len, kick_input, make_item, register_scorer
from punch_scoring import _band01
from scoring_api import ScoreRanges, _map01, build_metrics_from_sequence

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
R = ScoreRanges()


def _reason(**kv) -> str:
    return ", ".join(f"{k}≈{v}" for k, v in kv.items())


# ---------- phần dùng chung với luật của tác giả ----------
def author_expression(raw) -> List[Dict]:
    """EX_E1..E4 — chép đúng scoring_api.score_one_video (cùng ngưỡng ScoreRanges)."""
    return [
        make_item("expression", "EX_E1", "Tốc độ", 2.0, _map01(raw.speed_raw, *R.expr_speed) * 2.0,
                  f"chỉ số tốc độ≈{raw.speed_raw:.2f}",
                  "Tăng tốc độ đá: thả lỏng rồi bộc phát; không hy sinh đường đá."),
        make_item("expression", "EX_E2", "Lực", 1.6, _map01(raw.power_raw, *R.expr_power) * 1.6,
                  f"chỉ số bùng nổ≈{raw.power_raw:.2f}",
                  "Lực đến từ hông và chân trụ; tới đích phải dứt khoát."),
        make_item("expression", "EX_E3", "Độ cao", 1.4, _map01(raw.height_raw, *R.expr_height) * 1.4,
                  f"chỉ số độ cao≈{raw.height_raw:.2f}",
                  "Tăng dần độ cao, đừng gồng ép; giữ thân thẳng, chân trụ vững."),
        make_item("expression", "EX_E4", "Độ mượt", 1.0, _map01(raw.smoothness, *R.acc_smooth) * 1.0,
                  f"độ mượt≈{raw.smoothness:.2f}",
                  "Động tác không bị khựng; tập xen kẽ chậm–nhanh để quỹ đạo mượt hơn."),
    ]


def author_recovery(code: str, raw) -> Dict:
    return make_item("accuracy", code, "Thu chân về", 0.8, _map01(raw.recovery_ratio, *R.acc_recover) * 0.8,
                     f"tỉ lệ thu về≈{raw.recovery_ratio:.2f}, gập gối lại≈{raw.recovery_knee_bend_deg:.0f}°",
                     "Đá xong thu gối trước rồi mới hạ chân, trở về tư thế chuẩn bị.")


def author_balance_guard(code: str, raw) -> Dict:
    guard = raw.hand_guard_drop
    got = (0.65 * _map01(raw.stability, *R.acc_stable) + 0.35 * (1.0 - guard / 0.25)) * 0.4
    return make_item("accuracy", code, "Trọng tâm & tay thủ", 0.4, got,
                     f"ổn định≈{raw.stability:.2f}, tay thủ hạ≈{guard:.2f}",
                     "Chân trụ vững, hông không lắc; giữ tay thủ, không buông tay xuống.")


def height01(raw) -> float:
    return _map01(raw.height_raw, *R.expr_height)


def knee01(raw) -> float:
    return _map01(raw.kick_knee_angle_peak, 135.0, 175.0)


def facing_cos(world: np.ndarray, t_ref: int, t_peak: int, A: int) -> Optional[float]:
    """cos giữa hướng mặt (lúc bắt đầu) và hướng đá (hông -> cổ chân lúc đỉnh), trong mặt phẳng ngang (x, z)
    của tọa độ world MediaPipe. +1 = đá ra trước mặt, -1 = đá ra sau lưng. None nếu thiếu 3D."""
    if world is None or world.shape[0] <= max(t_ref, t_peak):
        return None
    w = world
    sh = 0.5 * (w[t_ref, LS] + w[t_ref, RS])
    face = (w[t_ref, HEAD] - sh)[[0, 2]]
    kick = (w[t_peak, A] - 0.5 * (w[t_peak, LH] + w[t_peak, RH]))[[0, 2]]
    nf, nk = float(np.linalg.norm(face)), float(np.linalg.norm(kick))
    if not np.isfinite(nf * nk) or nf < 1e-3 or nk < 1e-3:
        return None
    return float(np.dot(face, kick) / (nf * nk))


# ---------- khung chung cho 1 đòn đá ----------
def score_kick_with(pose: Dict, action: str, accuracy: Callable, kick_name: str, kick_window: str = "ankle",
                    fix_leg: bool = False) -> ScoreResult:
    """accuracy(raw, ctx) -> 4 hạng mục Chính xác. ctx có k, v, world, s, e (đoạn cắt), A (cổ chân đá).
    kick_window / fix_leg: như _score_kick (action_scorers.kick_input)."""
    k, v, fps = pose["kpts"], pose["vis"], float(pose["fps"])
    world = pose.get("world")
    if k.shape[0] < 8:
        return ScoreResult(action=action, valid=False, reason="Video quá ngắn")
    s, e, k_in = kick_input(k, v, fps, kick_window, fix_leg)
    raw = build_metrics_from_sequence(k_in[s:e + 1], v[s:e + 1], fps, ktype_hint="front")
    A, H, K = (LA, LH, LK) if raw.kick_side == "L" else (RA, RH, RK)
    ctx = {"k": k, "v": v, "world": None if world is None else world[s:e + 1], "s": s, "e": e, "A": A, "fps": fps}
    acc_items = accuracy(raw, ctx)
    expr_items = author_expression(raw)
    acc4 = float(np.clip(sum(it["got"] for it in acc_items), 0, 4))
    expr6 = float(np.clip(sum(it["got"] for it in expr_items), 0, 6))
    top = sorted([x for x in acc_items + expr_items if x["deduct"] > 1e-6], key=lambda x: -x["deduct"])[:6]

    hip = 0.5 * (k[:, LH] + k[:, RH])
    L = _torso_len(k)
    dist = np.linalg.norm(k[:, A] - hip, axis=1) / L
    dist[v[:, A] < 0.2] = np.nan
    sp = np.r_[np.nan, np.linalg.norm(np.diff(k[:, A], axis=0), axis=1) * fps / L]
    warnings = [] if raw.has_enough else ["Cổ chân bị che/khó thấy trong nhiều frame — điểm kém tin cậy."]
    warnings += ctx.get("warnings", [])
    return ScoreResult(
        action=action, valid=True, total=acc4 + expr6, acc=acc4, expr=expr6,
        acc_items=acc_items, expr_items=expr_items, top_issues=top, warnings=warnings,
        phases={"t0": s + raw.t0, "t_peak": s + raw.t_peak, "t_end": s + raw.t1, "win_s": s, "win_e": e},
        phase_labels=("Bắt đầu", "Đỉnh đá", "Kết thúc"),
        phase_names=("Chuẩn bị", "Ra chân", "ĐỈNH ĐÁ", "Thu chân", "Kết thúc"),
        series={"sig": dist, "sig_name": "Cổ chân cách hông (× thân)", "speed": sp,
                "speed_name": "Tốc độ cổ chân (thân/s)"},
        highlight=(H, K, A), side_text="chân trái" if raw.kick_side == "L" else "chân phải",
        status=f"{kick_name}: gối duỗi {raw.kick_knee_angle_peak:.0f}°, cao {raw.kick_height_over_hip:.2f}\n"
               f"thu về {raw.recovery_ratio * 100:.0f}%",
        raw=dict(raw.__dict__, **{k_: v_ for k_, v_ in ctx.items() if k_ in ("back_cos",)}),
    )


# ========== ĐÁ SAU (Dwit chagi) ==========
def _back_accuracy(raw, ctx) -> List[Dict]:
    straight01 = _map01(raw.straightness, 0.72, 0.96)          # ngưỡng FR_A1 của tác giả
    c = facing_cos(ctx["world"], 0, raw.t_peak, ctx["A"])
    ctx["back_cos"] = c
    if c is None:
        a1, why = straight01, f"độ thẳng≈{raw.straightness:.2f}, hướng đá: không đo được (thiếu 3D)"
    else:
        back01 = _map01(-c, 0.0, 0.6)                           # -c = 0.6 <=> lệch ≤ ~53° so với phía sau
        a1, why = 0.6 * straight01 + 0.4 * back01, f"độ thẳng≈{raw.straightness:.2f}, đá ra sau≈{-c:+.2f}"
        if c > 0.3:
            ctx.setdefault("warnings", []).append("Chân đá đi ra phía trước mặt — có thể không phải đá sau.")
    return [
        make_item("accuracy", "BK_A1", "Đẩy gót thẳng ra sau", 1.6, a1 * 1.6, why,
                  "Thu gối sát ngực, nhìn qua vai, đẩy gót chân theo đường thẳng ra phía sau; mũi chân hướng xuống."),
        make_item("accuracy", "BK_A2", "Độ cao & duỗi gối", 1.2, (0.55 * height01(raw) + 0.45 * knee01(raw)) * 1.2,
                  f"cổ chân cao≈{raw.kick_height_over_hip:.2f}, góc gối lớn nhất≈{raw.kick_knee_angle_peak:.0f}°",
                  "Duỗi hết gối khi tới đích, gót chân ngang hông trở lên."),
        author_recovery("BK_A3", raw),
        author_balance_guard("BK_A4", raw),
    ]


def score_back(pose: Dict, action: str, kick_window: str = "ankle", fix_leg: bool = False, **_) -> ScoreResult:
    return score_kick_with(pose, action, _back_accuracy, "đá sau", kick_window, fix_leg)


# ========== ĐÁ MÓC GẬP (Huryeo chagi — "đá gập vòng") ==========
def _hook_kick_accuracy(raw, ctx) -> List[Dict]:
    fold01 = _map01(raw.recovery_knee_bend_deg, 30.0, 90.0)
    # độ cong quỹ đạo: công thức "arc" của tác giả cho đá vòng cầu (scoring_rules.py)
    arc01 = float(np.clip((0.9 - raw.straightness) / 0.4, 0, 1))
    return [
        make_item("accuracy", "HKK_A1", "Duỗi chân rồi gập móc gót", 1.6, (0.5 * knee01(raw) + 0.5 * fold01) * 1.6,
                  f"góc gối lớn nhất≈{raw.kick_knee_angle_peak:.0f}°, gập gối lại≈{raw.recovery_knee_bend_deg:.0f}°",
                  "Duỗi thẳng chân lệch qua mục tiêu rồi gập gối thật nhanh, kéo gót chân về như lưỡi móc."),
        make_item("accuracy", "HKK_A2", "Độ cao & quỹ đạo vòng", 1.2, (0.7 * height01(raw) + 0.3 * arc01) * 1.2,
                  f"cổ chân cao≈{raw.kick_height_over_hip:.2f}, độ thẳng≈{raw.straightness:.2f}",
                  "Đưa gót chân lên tầm đầu, quét theo vòng cung ngang chứ không đá thẳng."),
        author_recovery("HKK_A3", raw),
        author_balance_guard("HKK_A4", raw),
    ]


def score_hook_kick(pose: Dict, action: str, kick_window: str = "ankle", fix_leg: bool = False, **_) -> ScoreResult:
    return score_kick_with(pose, action, _hook_kick_accuracy, "đá móc gập", kick_window, fix_leg)


# ========== ĐÁ ĐẨY (Mireo chagi — "đá ức", "đá ức ngang", "ngả người – đá ức chân") ==========
def shin_tilt(world: Optional[np.ndarray], t: int, A: int) -> Optional[float]:
    """Góc cẳng chân (gối -> cổ chân) so với mặt phẳng ngang tại frame t (3D). 0 = nằm ngang."""
    if world is None or world.shape[0] <= t:
        return None
    K = LK if A == LA else RK
    v = world[t, A] - world[t, K]
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-3:
        return None
    return float(np.degrees(np.arcsin(np.clip(-v[1] / n, -1, 1))))  # world y hướng xuống


def _push_accuracy(raw, ctx) -> List[Dict]:
    lift01 = _map01(raw.knee_lift_vertical, 0.04, 0.12)      # FR_A1 của tác giả
    line01 = _map01(raw.straightness, 0.72, 0.96)
    tilt = shin_tilt(ctx["world"], raw.t_peak, ctx["A"])
    if tilt is None:
        flat01, ttxt = 0.5, "không đo được (thiếu 3D)"
    else:
        flat01, ttxt = _map01(abs(tilt), 45.0, 15.0), f"{tilt:+.0f}°"
    return [
        make_item("accuracy", "PK_A1", "Co gối cao rồi đẩy thẳng", 1.6, (0.6 * lift01 + 0.4 * line01) * 1.6,
                  f"nâng gối≈{raw.knee_lift_vertical:.2f}, độ thẳng≈{raw.straightness:.2f}",
                  "Co gối sát ngực trước, rồi đẩy lòng bàn chân thẳng tới mục tiêu, hông đẩy theo."),
        make_item("accuracy", "PK_A2", "Chân duỗi ngang khi đẩy", 1.2, (0.45 * knee01(raw) + 0.55 * flat01) * 1.2,
                  f"góc gối lớn nhất≈{raw.kick_knee_angle_peak:.0f}°, cẳng chân so với phương ngang≈{ttxt}",
                  "Khi tới đích chân duỗi gần thẳng, cẳng chân nằm ngang tầm bụng–ngực, không hất lên."),
        author_recovery("PK_A3", raw),
        author_balance_guard("PK_A4", raw),
    ]


def score_push(pose: Dict, action: str, kick_window: str = "ankle", fix_leg: bool = False, **_) -> ScoreResult:
    return score_kick_with(pose, action, _push_accuracy, "đá đẩy", kick_window, fix_leg)


# ========== ĐÁ THẤP: luật đá ngang / vòng cầu của tác giả, đổi mục độ cao thành "đúng tầm thấp" ==========
# Đòn đá thấp cố ý (vào đùi/gối) sẽ bị luật gốc trừ điểm "độ cao" (SD_A2/RH_A2, EX_E3). Hai luật dưới đây giữ
# nguyên mọi hạng mục khác của tác giả và chỉ thay phần độ cao bằng mức đạt tầm thấp.
LOW_BAND = (-1.6, -1.0, 0.05, 0.45)   # cổ chân so với hông lúc đỉnh (× bề rộng vai/hông 2D, như kick_height_over_hip)


def _score_low(pose: Dict, action: str, base: str, a2_code: str, **kw) -> ScoreResult:
    from action_scorers import _score_kick
    res = _score_kick(pose, base, **kw)
    if not res.valid:
        return res
    res.action = action
    raw = res.raw
    low01 = _band01(raw["kick_height_over_hip"], *LOW_BAND)
    k01 = _map01(raw["kick_knee_angle_peak"], 135.0, 175.0)
    for it in res.acc_items + res.expr_items:
        if it["code"] == a2_code:
            g = (0.55 * low01 + 0.45 * k01) * 1.2
            it.update(label="Đúng tầm thấp & duỗi gối", got=g, deduct=1.2 - g, label_en="Low target & knee extension",
                      reason=f"cổ chân so với hông≈{raw['kick_height_over_hip']:+.2f}, "
                             f"góc gối lớn nhất≈{raw['kick_knee_angle_peak']:.0f}°",
                      tip="Đá thấp: cổ chân ngang gối–hông đối phương, duỗi hết gối khi chạm.")
        if it["code"] == "EX_E3":
            g = low01 * 1.4
            it.update(label="Đúng tầm (đá thấp)", got=g, deduct=1.4 - g, label_en="Target height (low kick)",
                      reason=f"cổ chân so với hông≈{raw['kick_height_over_hip']:+.2f}",
                      tip="Giữ đòn ở tầm thấp có kiểm soát, không để chân rơi chạm đất.")
    res.acc = float(np.clip(sum(it["got"] for it in res.acc_items), 0, 4))
    res.expr = float(np.clip(sum(it["got"] for it in res.expr_items), 0, 6))
    res.total = res.acc + res.expr
    res.top_issues = sorted([x for x in res.acc_items + res.expr_items if x["deduct"] > 1e-6],
                            key=lambda x: -x["deduct"])[:6]
    return res


def score_side_low(pose: Dict, action: str, **kw) -> ScoreResult:
    return _score_low(pose, action, "side", "SD_A2", **kw)


def score_roundhouse_low(pose: Dict, action: str, **kw) -> ScoreResult:
    return _score_low(pose, action, "roundhouse", "RH_A2", **kw)


for _name, _vi, _fn in [("back", "Dwit chagi (đá sau)", score_back),
                        ("hook_kick", "Huryeo chagi (đá móc gập)", score_hook_kick),
                        ("push", "Mireo chagi (đá đẩy / đá ức)", score_push),
                        ("side_low", "Yeop chagi thấp (đá cạnh thấp)", score_side_low),
                        ("roundhouse_low", "Dollyo chagi thấp (đá vòng cầu thấp)", score_roundhouse_low)]:
    _fn.rep_signal = "ankle"
    register_scorer(_name, _vi)(_fn)

# tên thư mục / lớp khác -> luật (xem ACTIONS.md, bảng ALIASES)
ALIASES.update({"dwit_chagi": "back", "da_sau": "back", "huryeo_chagi": "hook_kick", "da_gap_vong": "hook_kick",
                "mireo_chagi": "push", "da_day": "push", "da_uc": "push", "da_uc_chan": "push",
                "da_canh_thap": "side_low", "da_vong_cau_thap": "roundhouse_low",
                "da_truoc": "front", "da_vong_cau": "roundhouse", "da_ngang": "side", "da_bo": "axe"})
