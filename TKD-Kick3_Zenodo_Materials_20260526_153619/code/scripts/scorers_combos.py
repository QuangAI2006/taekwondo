# scripts/scorers_combos.py
"""
Luật chấm cho ĐÒN PHỐI HỢP và đấm trong tấn, ghép từ các luật đã có (không đặt ngưỡng mới):

  block_punch : gạt thấp + đấm thẳng   = 0.5 × luật gạt thấp (scorers_hands) + 0.5 × luật đấm thẳng (punch_scoring)
  block_kick  : gạt thấp + đá trước    = 0.5 × luật gạt thấp + 0.5 × luật đá trước của tác giả (scoring_api)
  horse_punch : đấm thẳng trong tấn rộng = luật đấm thẳng (thủ ở hông) + kiểm tra mở tấn ở mục PU_A4

Mỗi phần vẫn là thang 4 + 6, nên tổng có trọng số vẫn là Chính xác 4 + Biểu hiện 6. Các hạng mục hiển thị
được nhân trọng số (điểm tối đa 0.8 thay vì 1.6, ...). Tham số side = tay GẠT (mặc định trái, như bài tập).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from action_scorers import ALIASES, ScoreResult, _score_kick, _score_punch, register_scorer
from hand_scoring import alternating_side, alternating_signal, rep_signal
from scorers_hands import score_hand
from scorers_stances import stance_at

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)


def _scaled(items: List[Dict], w: float, prefix: str) -> List[Dict]:
    out = []
    for it in items:
        it = dict(it)
        it["maxi"] *= w
        if it.get("got") is not None:
            it["got"] *= w
            it["deduct"] = it["maxi"] - it["got"]
        it["label"] = f"{prefix}: {it['label']}"
        out.append(it)
    return out


def _combine(action: str, parts, phase_names) -> ScoreResult:
    """parts: [(tên phần, ScoreResult, trọng số)] theo thứ tự thực hiện."""
    bad = [(n, r) for n, r, _ in parts if not r.valid]
    if bad:
        n, r = bad[0]
        return ScoreResult(action=action, valid=False, reason=f"Phần '{n}' không chấm được: {r.reason}")
    acc_items, expr_items, warnings = [], [], []
    for n, r, w in parts:
        acc_items += _scaled(r.acc_items, w, n)
        expr_items += _scaled(r.expr_items, w, n)
        warnings += [f"{n}: {x}" for x in r.warnings]
    acc = float(sum(w * r.acc for _, r, w in parts))
    expr = float(sum(w * r.expr for _, r, w in parts))
    first, last = parts[0][1], parts[-1][1]
    if first.phases["t_peak"] > last.phases["t_peak"]:
        warnings.append(f"Thứ tự chưa đúng: '{parts[-1][0]}' tới đích trước '{parts[0][0]}'.")
    top = sorted([x for x in acc_items + expr_items if x.get("got") is not None and x["deduct"] > 1e-6],
                 key=lambda x: -x["deduct"])[:6]
    hl = tuple(dict.fromkeys(first.highlight + last.highlight))
    return ScoreResult(
        action=action, valid=True, total=acc + expr, acc=acc, expr=expr, acc_items=acc_items,
        expr_items=expr_items, top_issues=top, warnings=warnings,
        phases={"t0": first.phases["t0"], "t_peak": last.phases["t_peak"], "t_end": last.phases.get("t_end")},
        phase_labels=("Bắt đầu gạt", parts[-1][0], "Kết thúc"), phase_names=phase_names,
        series=last.series, highlight=hl, side_text=" + ".join(r.side_text for _, r, _ in parts if r.side_text),
        status=f"{parts[0][0]} {first.total:.1f} · {parts[-1][0]} {last.total:.1f}",
        raw={n: r.raw for n, r, _ in parts},
    )


def score_block_punch(pose: Dict, action: str, side: Optional[str] = None, **_) -> ScoreResult:
    bside = side or "L"
    block = score_hand(pose, "low_block", side=bside)
    punch = _score_punch(pose, "straight", guard="waist", side="R" if bside == "L" else "L")
    return _combine(action, [("Gạt thấp", block, 0.5), ("Đấm thẳng", punch, 0.5)],
                    ("Chuẩn bị", "Gạt", "ĐẤM", "Thu tay", "Kết thúc"))


def score_block_kick(pose: Dict, action: str, side: Optional[str] = None, kick_window: str = "ankle",
                     fix_leg: bool = False, **_) -> ScoreResult:
    block = score_hand(pose, "low_block", side=side or "L")
    kick = _score_kick(pose, "front", kick_window=kick_window, fix_leg=fix_leg)
    return _combine(action, [("Gạt thấp", block, 0.5), ("Đá trước", kick, 0.5)],
                    ("Chuẩn bị", "Gạt", "ĐÁ", "Thu chân", "Kết thúc"))


def score_horse_punch(pose: Dict, action: str, side: Optional[str] = None, **_) -> ScoreResult:
    res = _score_punch(pose, "straight", guard="waist", side=side)
    if not res.valid:
        return res
    res.action = action
    st = stance_at(pose.get("world"), pose["kpts"], res.phases["t_peak"])
    for it in res.acc_items:   # "rộng chân": chân mở rộng, gối chùng (stance_at.open01), không đòi trung bình tấn sâu
        if it["code"] == "PU_A4":
            g = 0.5 * it["got"] + 0.5 * st["open01"] * 0.4
            it.update(code="HP_A4", label="Tay kéo về hông, thăng bằng & tấn rộng", got=g, deduct=0.4 - g,
                      label_en="Pull-back hand, balance & wide stance",
                      reason=it["reason"] + f", tấn rộng≈{st['width']:.1f}× vai, gối≈{st['knee_min']:.0f}°",
                      tip="Giữ chân rộng (~2 vai), gối chùng suốt bài; tay không đấm kéo về hông.")
    assess = [it for it in res.acc_items if it.get("got") is not None]
    res.acc = float(4.0 * sum(it["got"] for it in assess) / sum(it["maxi"] for it in assess))
    res.total = res.acc + res.expr
    res.top_issues = sorted([x for x in res.acc_items + res.expr_items if x.get("got") is not None
                             and x["deduct"] > 1e-6], key=lambda x: -x["deduct"])[:6]
    res.raw = dict(res.raw, stance_width=st["width"], stance_knee_min=st["knee_min"])
    return res


# tách rep trong video dài: mỗi lần gạt (tay gạt hạ xuống) là 1 rep; đòn đấm/đá theo sau nằm trong post_s
score_block_punch.rep_signal_fn = lambda pose, action, side=None, **_: rep_signal(pose, "drop", side or "L")
score_block_punch.rep_cfg = {"min_gap_s": 1.2, "pre_s": 1.0, "post_s": 1.6}
score_block_kick.rep_signal_fn = lambda pose, action, side=None, **_: rep_signal(pose, "drop", side or "L")
score_block_kick.rep_cfg = {"min_gap_s": 1.2, "pre_s": 0.8, "post_s": 2.4, "clip": False}
score_horse_punch.rep_signal_fn = alternating_signal
score_horse_punch.rep_cfg = {"min_gap_s": 0.5, "pre_s": 0.8, "post_s": 1.0, "clip": False, "side_fn": alternating_side}
register_scorer("block_punch", "Gạt thấp – đấm thẳng")(score_block_punch)
register_scorer("block_kick", "Gạt thấp – đá trước")(score_block_kick)
register_scorer("horse_punch", "Rộng chân – đấm thẳng (đấm thẳng trong tấn rộng)")(score_horse_punch)

ALIASES.update({"gat_thap_dam_thang": "block_punch", "gat_thap_da_truoc": "block_kick",
                "rong_chan_dam_thang": "horse_punch", "juchum_seogi_jireugi": "horse_punch"})
